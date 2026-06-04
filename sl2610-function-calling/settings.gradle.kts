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

// Reusable library modules. :runtime = the generic IREE vmfb runner; :llm =
// FunctionGemma decode + compact tool-call codec; :asr = Moonshine ASR on the
// Torq NPU; :vad = Silero speech segmenter (drop into any KMP app).
include(":runtime", ":llm", ":asr", ":vad")
