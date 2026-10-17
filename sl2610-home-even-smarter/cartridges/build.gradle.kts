import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// The bridge between the app and the cartridges: locating materialized pack_dirs on the board, downloading them from
// the cartridge server, and creating engines from them. On the board (linuxArm64) the engines run on the in-process
// IREE runtime (:iree); the linuxX64 and JVM factories exist for host runs and tests and report it unavailable.
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
            api(project(":core"))
            implementation(libs.kotlinx.serializationJson)
            api(libs.kotlinx.ioCore)                 // kotlinx.io.files.Path is part of the public API (CartridgeStore.root)
            implementation(libs.ktor.clientCore)
            implementation(libs.ktor.clientCio)
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.kotlinx.coroutinesTest)
            implementation(libs.ktor.clientMock)
        }
        // The board: real engines over the in-process IREE runtime (:iree) and the blueprint modules' engine code.
        linuxArm64Main.dependencies {
            implementation(project(":iree"))
            implementation(libs.cartridge.nlu.functiongemma)       // FunctionGemmaEngine, KvBackend, PackDir (included build)
            implementation(libs.cartridge.asr.moonshine)           // StreamingAsrCartridge, PackDir, Language (included build)
            implementation(project.dependencies.platform(libs.transformers.bom))
            implementation(libs.transformers.core)                 // TokenizerFactory
            implementation(libs.skainet.io.gguf)                   // StreamingGGUFReader, createRandomAccessSource (pread)
        }
    }
}
