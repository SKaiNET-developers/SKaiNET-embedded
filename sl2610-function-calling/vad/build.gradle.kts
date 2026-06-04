plugins {
    alias(libs.plugins.kotlinMultiplatform)
}

group = "sk.coral.voicecc"
version = "0.1.0"

// Silero VAD speech segmenter. The board target streams utterances from the
// light capture+VAD helper (scripts/vad_capture.py via the sample venv — no
// Moonshine, low RAM) as a subprocess. Pure platform/posix, no SKaiNET deps,
// so it is reusable. jvm target is a no-op placeholder for KMP consumers.
kotlin {
    jvmToolchain(21)
    jvm()
    linuxArm64()
}
