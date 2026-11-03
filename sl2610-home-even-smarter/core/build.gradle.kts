import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// Domain and pipeline: the home model, the tool → command mapping, the voice pipeline state machine and the engine
// contracts the cartridges are adapted to. Pure Kotlin — no UI, no cartridge dependency — so it runs in fast JVM
// tests with fake engines and natively on the board.
plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.kotlinSerialization)
}

kotlin {
    jvm { compilerOptions { jvmTarget.set(JvmTarget.JVM_17) } }
    linuxX64()
    linuxArm64()

    sourceSets {
        commonMain.dependencies {
            api(libs.kotlinx.coroutinesCore)
            api(libs.kotlinx.serializationJson)
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.kotlinx.coroutinesTest)
        }
    }
}
