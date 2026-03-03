plugins {
    kotlin("multiplatform") version "2.3.0"
    id("com.gradleup.shadow") version "9.0.0-beta12"
}

group = "sk.ainet.embedded"
version = "0.1.0"

repositories {
    mavenCentral()
    google()
}

kotlin {
    jvm()

    sourceSets {
        commonMain.dependencies {
            implementation("sk.ainet:skainet-lang-core:0.13.0")
            implementation("sk.ainet:skainet-compile-dag:0.13.0")
            implementation("sk.ainet:skainet-compile-opt:0.13.0")
        }

        commonTest.dependencies {
            implementation(kotlin("test"))
        }

        jvmMain.dependencies {
            // CLI argument parsing could go here if needed
        }
    }
}

// Configure the existing shadowJar task from the shadow plugin
tasks.named<com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar>("shadowJar") {
    archiveBaseName.set("coralnpu-codegen")
    archiveClassifier.set("all")
    from(kotlin.jvm().compilations["main"].output)
    configurations = listOf(
        project.configurations.getByName("jvmRuntimeClasspath")
    )
    manifest {
        attributes("Main-Class" to "sk.ainet.embedded.coralnpu.codegen.MainKt")
    }
    mergeServiceFiles()
}
