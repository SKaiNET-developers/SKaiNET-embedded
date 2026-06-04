plugins {
    alias(libs.plugins.kotlinMultiplatform)
}

group = "sk.coral.voicecc"
version = "0.1.0"

// Reusable FunctionGemma LLM module:
//   commonMain  — CompactCodec + ToolCall (pure Kotlin, no deps): the tool-call
//                 grammar, usable on any target / in any app.
//   linuxArm64  — GemmaDecoder: the on-device greedy decode loop driving OUR
//                 DSL->StableHLO->IREE f16 vmfb (via :runtime) + the GGUF
//                 tokenizer. Give it a vmfb + .irpa + gguf and a prompt; get
//                 back the tool-call text + parsed ToolCalls.
kotlin {
    jvmToolchain(21)
    jvm()
    linuxArm64()

    sourceSets {
        linuxArm64Main.dependencies {
            implementation(project(":runtime"))
            // GGUFTokenizer + the gemma3 graph/loader (commonMain, native-capable).
            implementation(project.dependencies.platform(libs.skainet.transformers.bom))
            implementation(libs.skainet.transformers.inference.gemma)
            implementation(libs.skainet.lang.core)
            implementation(libs.skainet.backend.cpu)
            implementation(libs.skainet.io.core)
            implementation(libs.skainet.io.gguf)
        }
    }
}
