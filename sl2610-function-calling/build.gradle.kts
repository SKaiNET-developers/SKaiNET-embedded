import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import java.io.File
import java.net.HttpURLConnection
import java.net.URI

plugins {
    alias(libs.plugins.kotlinMultiplatform)
}

group = "sk.coral.voicecc"
version = "0.1.0"

kotlin {
    jvmToolchain(21)

    // Host dev target: fast iteration + A/B reference harness.
    jvm {
        @OptIn(ExperimentalKotlinGradlePluginApi::class)
        mainRun { mainClass.set("voicecc.MainKt") }
    }

    // The shipped board binary (Astra Machina SL2610, aarch64 Linux).
    // Cross-compiled from the x64 host: linkReleaseExecutableLinuxArm64.
    linuxArm64 {
        binaries {
            executable {
                entryPoint = "voicecc.main"
            }
        }
    }

    sourceSets {
        // commonMain (App + actions) is pure Kotlin — the SKaiNET model deps now
        // live in :llm (board LLM) and jvmMain (host export tooling), not here.
        commonTest.dependencies {
            implementation(kotlin("test"))
        }
        // The board binary wires: the published gemma-iree runtime (GemmaDecoder
        // + IreeRuntime + CompactCodec, from SKaiNET-transformers), plus the
        // app-local :asr (Moonshine ASR on the NPU) and :vad (Silero).
        linuxArm64Main.dependencies {
            implementation(project.dependencies.platform(libs.skainet.transformers.bom))
            implementation(libs.skainet.transformers.runtime.gemma.iree)
            implementation(project(":asr"))
            implementation(project(":vad"))
        }
        // JVM-only host model-compiler tooling (export/: DAG -> StableHLO trace).
        // The FFM-accelerated backend runs the trace; never ships in the native binary.
        jvmMain.dependencies {
            implementation(project.dependencies.platform(libs.skainet.transformers.bom))
            implementation(libs.skainet.lang.core)
            implementation(libs.skainet.backend.native.cpu)
            implementation(libs.skainet.compile.hlo)
            implementation(libs.skainet.compile.dag)
            implementation(libs.skainet.transformers.core) // transformer Modules (MHA/RoPE/FFN) for trace tooling
        }
    }
}

// Build the release aarch64 binary and push+run it on the board.
//   ./gradlew deployBoard            (set BOARD=root@<ip> if not the default)
tasks.register<Exec>("deployBoard") {
    group = "deploy"
    description = "Cross-compile the linuxArm64 release binary and deploy+run it on the SL2610 board."
    dependsOn("linkReleaseExecutableLinuxArm64")
    commandLine("bash", "scripts/deploy.sh", "--run")
    environment("BOARD", System.getenv("BOARD") ?: "root@192.168.3.26")
}

// Host model-compiler bridge: build a small SKaiNET DAG, export StableHLO MLIR.
// Downstream: scripts/iree-compile-cpu.sh runs iree-compile (llvm-cpu+NEON).
tasks.register<JavaExec>("hloExport") {
    group = "bridge"
    description = "Export a sample SKaiNET DAG to StableHLO MLIR (host tooling)."
    dependsOn("jvmMainClasses")
    val main = kotlin.jvm().compilations.getByName("main")
    classpath(main.output.allOutputs, main.runtimeDependencyFiles)
    mainClass.set("voicecc.export.HloBridgeKt")
    val out = layout.buildDirectory.dir("mlir").get().asFile
    doFirst { out.mkdirs() }
    args(out.resolve("addrelu.mlir").path)
}

// Fetch the FunctionGemma-270M GGUF (~248 MB) from Hugging Face into models/.
// The gguf is gitignored (not committed) — run this once before the board pipeline.
//   ./gradlew downloadModel            (set HF_TOKEN=… if the repo ever becomes gated)
val functionGemmaRepo = "BrinqAI/functiongemma-270m-physical-ai"
val functionGemmaFile = "functiongemma-physical-ai-v10-Q5_K_M.gguf"

tasks.register("downloadModel") {
    group = "setup"
    description = "Download the FunctionGemma-270M GGUF (~248 MB) from Hugging Face into models/."
    val dest = layout.projectDirectory.dir("models").file(functionGemmaFile).asFile
    outputs.file(dest)
    doLast {
        if (dest.exists() && dest.length() > 0) {
            logger.lifecycle("FunctionGemma already present: ${dest.path} (${dest.length() / 1_000_000} MB)")
            return@doLast
        }
        dest.parentFile.mkdirs()
        val token = System.getenv("HF_TOKEN")
        var url = URI("https://huggingface.co/$functionGemmaRepo/resolve/main/$functionGemmaFile?download=true").toURL()
        val tmp = File(dest.path + ".part")
        // Follow HF's 302 to the LFS CDN by hand (it's a cross-host redirect, and the
        // auth header must NOT be forwarded to the presigned CDN URL or it 400s).
        var redirects = 0
        while (true) {
            val conn = (url.openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = false
                connectTimeout = 30_000
                readTimeout = 60_000
                setRequestProperty("User-Agent", "sl2610-function-calling-gradle")
                if (token != null && url.host.endsWith("huggingface.co")) {
                    setRequestProperty("Authorization", "Bearer $token")
                }
            }
            when (val code = conn.responseCode) {
                in 300..399 -> {
                    val loc = conn.getHeaderField("Location") ?: error("redirect ($code) without Location")
                    conn.disconnect()
                    if (++redirects > 8) error("too many redirects fetching $functionGemmaFile")
                    url = URI(loc).toURL()
                }
                200 -> {
                    val total = conn.contentLengthLong
                    logger.lifecycle(
                        "downloading $functionGemmaFile " +
                            "(${if (total > 0) "${total / 1_000_000} MB" else "unknown size"}) -> ${dest.path}",
                    )
                    conn.inputStream.use { input -> tmp.outputStream().use { out -> input.copyTo(out, 1 shl 20) } }
                    conn.disconnect()
                    if (!tmp.renameTo(dest)) { tmp.copyTo(dest, overwrite = true); tmp.delete() }
                    logger.lifecycle("done: ${dest.path} (${dest.length() / 1_000_000} MB)")
                    return@doLast
                }
                else -> { conn.disconnect(); error("HTTP $code fetching $url") }
            }
        }
    }
}

// "Trace SDPA first": record a 1-block gemma3 forward and report the op node
// types (is scaledDotProductAttention atomic?) + which lack StableHLO converters.
tasks.register<JavaExec>("sdpaTrace") {
    group = "bridge"
    description = "Trace a gemma3 transformer block and inspect recorded ops vs StableHLO converters."
    dependsOn("jvmMainClasses")
    val main = kotlin.jvm().compilations.getByName("main")
    classpath(main.output.allOutputs, main.runtimeDependencyFiles)
    mainClass.set("voicecc.export.SdpaTraceKt")
}
