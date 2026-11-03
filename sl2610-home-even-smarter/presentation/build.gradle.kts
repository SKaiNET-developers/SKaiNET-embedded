import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// App state without a UI toolkit: AppViewModel and AppEnvironment (engines, cartridges, pipeline, golden set, log).
// The terminal UI in :tui renders it; any other front-end could drive the same view model.
plugins {
    alias(libs.plugins.kotlinMultiplatform)
}

kotlin {
    jvm { compilerOptions { jvmTarget.set(JvmTarget.JVM_17) } }
    linuxX64()
    linuxArm64()

    sourceSets {
        commonMain.dependencies {
            api(project(":core"))
            api(project(":cartridges"))
        }
    }
}
