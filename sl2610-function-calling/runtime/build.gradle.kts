plugins {
    alias(libs.plugins.kotlinMultiplatform)
}

group = "sk.coral.voicecc"
version = "0.1.0"

// Generic IREE vmfb runner. The SL2610 ships only a statically-linked
// `iree-run-module` (no shared libiree C API), so the board target drives it as
// a subprocess. Pure platform/posix — no SKaiNET deps — so it is reusable for
// ANY vmfb (LLM, ASR, VAD). jvm target is a no-op placeholder so the module can
// be consumed from a KMP app that also has a jvm target.
kotlin {
    jvmToolchain(21)
    jvm()
    linuxArm64()
}
