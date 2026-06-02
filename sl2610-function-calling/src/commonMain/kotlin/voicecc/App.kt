package voicecc

import voicecc.actions.Intent
import voicecc.actions.defaultRouter

/**
 * Shared entry. Phase 0 = a smoke run that wires the [defaultRouter] and
 * dispatches a few intents end-to-end (proves the common code path on both
 * the jvm host and the linuxArm64 board binary). Later phases add the real
 * mic -> VAD -> ASR -> Gemma -> codec pipeline ahead of this dispatch.
 */
fun runApp(args: Array<String>) {
    println("sl2610-voice-cc-kt 0.1.0 — native voice command-and-control (scaffold)")
    val router = defaultRouter()
    println("registered tools: ${router.tools}")

    val demo = listOf(
        Intent("set_lights", mapOf("color" to "red", "state" to "on")),
        Intent("get_system_status", mapOf("metric" to "all")),
        Intent("respond", mapOf("message" to "hello from kotlin/native")),
    )
    for (r in router.dispatchAll(demo)) println("  $r")
}
