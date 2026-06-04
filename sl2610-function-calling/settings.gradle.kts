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

rootProject.name = "sl2610-function-calling"

// The LLM runtime (Gemma decode + IREE vmfb driver + tool-call codec) was
// extracted into SKaiNET-transformers as :llm-runtime:gemma-iree, consumed here
// via composite build. The board binary builds against transformers' pinned
// SKaiNET release as-is; add -PuseLocalSkainet=true only when also building the
// host gemma-export tooling, which needs StableHLO converters not yet released.
includeBuild("../../SKaiNET-transformers") {
    // Map the published coordinate to the local project explicitly — Gradle's
    // automatic substitution doesn't see vanniktech's lazily-set POM artifactId.
    dependencySubstitution {
        substitute(module("sk.ainet.transformers:skainet-transformers-runtime-gemma-iree"))
            .using(project(":llm-runtime:gemma-iree"))
    }
}

// App-local modules: :asr = Moonshine ASR on the Torq NPU; :vad = Silero speech
// segmenter. (The former :runtime + :llm now live in gemma-iree, above.)
include(":asr", ":vad")
