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

// The cartridge blueprints (Kotlin API + build recipe, no weights) from a sibling checkout of
// SKaiNET-cartridge-blueprints, as an included build. Override the location with -PblueprintsDir=<path>.
val blueprintsDir = providers.gradleProperty("blueprintsDir").getOrElse("../../SKaiNET-cartridge-blueprints")
includeBuild(blueprintsDir) {
    dependencySubstitution {
        substitute(module("sk.ainet.cartridge:nlu-functiongemma-270m-iree"))
            .using(project(":blueprints:nlu-functiongemma-270m-iree"))
        substitute(module("sk.ainet.cartridge:asr-moonshine-v2-streaming-iree"))
            .using(project(":blueprints:asr-moonshine-v2-streaming-iree"))
    }
}

// :core = home domain, voice pipeline, engine contracts; :cartridges = cartridge store/downloader + engine factory;
// :presentation = AppViewModel (no UI toolkit); :tui = the board app (Kotlin/Native, Mosaic);
// :server = cartridge server (JVM) the board downloads its NPU cartridges from;
// :iree = in-process IREE runtime (CPU + Torq NPU) for the board, over native/iree-torq-runtime.
include(":core", ":cartridges", ":presentation", ":tui", ":server", ":iree")
