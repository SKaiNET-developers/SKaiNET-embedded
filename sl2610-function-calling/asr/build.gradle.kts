plugins {
    alias(libs.plugins.kotlinMultiplatform)
}

group = "sk.coral.voicecc"
version = "0.1.0"

// Moonshine ASR on the Torq NPU. The board target drives the prebuilt Synaptics
// vmfbs via the version-matched torq.runtime as a subprocess (scripts/
// moonshine_npu.py) — pure platform/posix, no SKaiNET deps, so it is reusable.
// jvm target is a no-op placeholder so a KMP app with a jvm target can consume it.
kotlin {
    jvmToolchain(21)
    jvm()
    linuxArm64()
}
