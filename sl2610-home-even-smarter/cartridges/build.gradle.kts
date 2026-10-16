import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// The bridge between the app and the cartridges: locating materialized pack_dirs on the board, downloading them from
// the cartridge server, and creating engines from them. The Linux engine factory reports the runtime unavailable
// until the native IREE (Torq) binding lands; the JVM one exists for host tests only.
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
    }
}
