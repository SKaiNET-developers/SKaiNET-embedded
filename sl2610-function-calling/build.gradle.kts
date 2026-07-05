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
        commonMain.dependencies {
            implementation(libs.kotlinx.coroutines)
            // The gemma3 graph, GGUF loader (incl. Q5_K dequant), KV-cache and
            // tokenizer are all commonMain (native-capable). We reuse those and
            // run them eagerly via the pure-Kotlin CPU backend — works on both
            // jvm (host A/B) and linuxArm64 (the board binary).
            implementation(project.dependencies.platform(libs.skainet.transformers.bom))
            implementation(libs.skainet.transformers.inference.gemma)
            // Reusable Gemma-on-IREE runtime: CompactCodec (commonMain) +
            // IreeRuntime/GemmaDecoder (nativeMain) — replaces the demo-local copies.
            implementation(libs.skainet.transformers.runtime.gemma.iree)
            implementation(libs.skainet.lang.core)
            implementation(libs.skainet.backend.cpu)
            implementation(libs.skainet.io.core)
            implementation(libs.skainet.io.gguf)
        }
        commonTest.dependencies {
            implementation(kotlin("test"))
        }
        // JVM gets the FFM-accelerated backend + the host-only model-compiler
        // tooling (DAG -> StableHLO). These never ship in the native binary.
        jvmMain.dependencies {
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
