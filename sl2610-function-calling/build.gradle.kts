import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi

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
        // The board binary wires the reusable library modules: :llm (FunctionGemma
        // decode + codec), :asr (Moonshine ASR on the NPU), :vad (Silero segmenter),
        // and :runtime (the IREE vmfb runner, used by Main's iree-smoke/gemma/gen).
        linuxArm64Main.dependencies {
            implementation(project(":runtime"))
            implementation(project(":llm"))
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
