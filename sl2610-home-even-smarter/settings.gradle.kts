rootProject.name = "sl2610-home-even-smarter"

pluginManagement {
    repositories {
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
        mavenCentral()
        // Mosaic's Compose runtime depends on androidx KMP artifacts (collection, annotation, lifecycle) published
        // only on Google's repository.
        google { mavenContent { includeGroupAndSubgroups("androidx") } }
    }
}

plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}

// :core = home domain, voice pipeline, engine contracts; :cartridges = cartridge store/downloader + engine factory;
// :presentation = AppViewModel (no UI toolkit); :tui = the board app (Kotlin/Native, Mosaic);
// :server = cartridge server (JVM) the board downloads its NPU cartridges from;
// :iree = in-process IREE runtime (CPU + Torq NPU) for the board, over native/iree-torq-runtime.
include(":core", ":cartridges", ":presentation", ":tui", ":server", ":iree")
