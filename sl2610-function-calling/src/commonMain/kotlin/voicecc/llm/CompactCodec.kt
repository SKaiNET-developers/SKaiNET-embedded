package voicecc.llm

import voicecc.actions.Intent

/**
 * Decodes FunctionGemma's compact tool-call format into [Intent]s — the Kotlin
 * port of vendor/function_calling/compact_codec.py (v10 Octopus-v2 named-arg
 * style, which our fine-tune emits):
 *   <tool_0>(state="on")<end>          -> Intent("set_lights", {state:on})
 *   <tool_4>(metric="all")<end>        -> Intent("get_system_status", {metric:all})
 *   <tool_5>(message="hi")<end>        -> Intent("respond", {message:hi})
 */
public object CompactCodec {
    private val TOKEN_TO_NAME: Map<String, String?> = mapOf(
        "0" to "set_lights", "1" to "play_buzzer", "2" to "set_alarm",
        "3" to "cancel_alarm", "4" to "get_system_status", "5" to "respond",
        "none" to null,
    )
    private val CALL_RE = Regex("""<tool_(\d+|none)>\(([^)]*)\)(?:<end>)?""")
    private val NAMED_ARG_RE = Regex("""(\w+)\s*=\s*"([^"]*)"""")

    /** Parse all tool calls in [raw] (special-token text from the model). */
    public fun parse(raw: String): List<Intent> =
        CALL_RE.findAll(raw).mapNotNull { m ->
            val name = TOKEN_TO_NAME[m.groupValues[1]] ?: return@mapNotNull null
            val args = NAMED_ARG_RE.findAll(m.groupValues[2])
                .associate { it.groupValues[1] to it.groupValues[2] }
            Intent(name, args)
        }.toList()
}
