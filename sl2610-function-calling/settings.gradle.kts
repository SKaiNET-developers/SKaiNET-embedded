pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

// DEFAULT: resolve the whole SKaiNET stack from published Maven Central artifacts,
// so a fresh `git clone` of just this directory builds with one Gradle command and
// no sibling checkouts. Versions are pinned by the transformers BOM (libs.versions.toml).
//
// OPT-IN composite build for developing the stack + this demo together — needs the
// sibling `../../SKaiNET` and `../../SKaiNET-transformers` source checkouts present:
//   ./gradlew <task> -PuseLocalStack=true
// It substitutes sk.ainet.core:* (SKaiNET) and sk.ainet.transformers:* coordinates
// with the local projects (the transformers modules publish as skainet-transformers-*
// but are named gemma/llm-core/llm-bom, so they're mapped explicitly).
if (providers.gradleProperty("useLocalStack").orNull == "true") {
    includeBuild("../../SKaiNET")
    includeBuild("../../SKaiNET-transformers") {
        dependencySubstitution {
            substitute(module("sk.ainet.transformers:skainet-transformers-bom")).using(project(":llm-bom"))
            substitute(module("sk.ainet.transformers:skainet-transformers-core")).using(project(":llm-core"))
            substitute(module("sk.ainet.transformers:skainet-transformers-inference-gemma")).using(project(":llm-inference:gemma"))
            substitute(module("sk.ainet.transformers:skainet-transformers-runtime-gemma-iree")).using(project(":llm-runtime:gemma-iree"))
        }
    }
}

rootProject.name = "sl2610-voice-cc-kt"
