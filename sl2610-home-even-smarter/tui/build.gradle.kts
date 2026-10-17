// The board app: a terminal UI (Mosaic) over AppViewModel. Kotlin/Native executables — linuxArm64 for the board,
// linuxX64 for running the same binary on the development host. No JVM on the board.
plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.composeCompiler)
}

kotlin {
    listOf(linuxX64(), linuxArm64()).forEach { target ->
        target.binaries.executable {
            baseName = "hes-tui"
            entryPoint = "sk.ainet.examples.smarthome.tui.main"
        }
    }
    // Board check of the real NLU without the UI: hes-nlu-check <pack-dir> [sentence…]
    linuxArm64().binaries.executable("nluCheck") {
        baseName = "hes-nlu-check"
        entryPoint = "sk.ainet.examples.smarthome.tui.nlucheck.main"
    }
    // Board check of the real ASR without the UI: hes-asr-check [--fast] <pack-dir> <file.wav>…
    linuxArm64().binaries.executable("asrCheck") {
        baseName = "hes-asr-check"
        entryPoint = "sk.ainet.examples.smarthome.tui.asrcheck.main"
    }
    // Board check of the whole app (AppViewModel, installed cartridges, real engines) without the terminal UI.
    linuxArm64().binaries.executable("appCheck") {
        baseName = "hes-app-check"
        entryPoint = "sk.ainet.examples.smarthome.tui.appcheck.main"
    }

    sourceSets {
        commonMain.dependencies {
            implementation(project(":presentation"))
            implementation(libs.mosaic.runtime)
        }
    }
}
