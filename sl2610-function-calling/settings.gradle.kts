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

// The LLM runtime (Gemma decode + IREE vmfb driver + tool-call codec) lives in
// SKaiNET-transformers as :llm-runtime:gemma-iree, consumed here as a published
// Maven artifact (sk.ainet.transformers:skainet-transformers-runtime-gemma-iree,
// pinned via the transformers BOM). No composite build needed.

// App-local modules: :asr = Moonshine ASR on the Torq NPU; :vad = Silero speech
// segmenter. (The former :runtime + :llm now live in the published gemma-iree.)
include(":asr", ":vad")
