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

    sourceSets {
        commonMain.dependencies {
            implementation(project(":presentation"))
            implementation(libs.mosaic.runtime)
        }
    }
}
