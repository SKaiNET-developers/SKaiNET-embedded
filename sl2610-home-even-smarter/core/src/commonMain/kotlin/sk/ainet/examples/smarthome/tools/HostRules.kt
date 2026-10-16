package sk.ainet.examples.smarthome.tools

import sk.ainet.examples.smarthome.engine.NluOutcome

/** A tool call the host rewrote before acting on it, and why. */
data class Correction(val from: NluOutcome.Call, val to: NluOutcome.Call, val reason: String)

/**
 * Cheap, precise rules that run after the language model and before the action: the bottom rung of the escalation
 * ladder. They never invent a call — they only repair a call the model made when the transcript plainly says
 * otherwise. Today's single rule: a 270M model likes to put "twenty one degrees" into a light's brightness.
 */
object HostRules {
    private val temperatureWords = Regex("""\b(degrees?|temperature|thermostat|heating|warmer|colder|cooler|celsius)\b""")
    private val wholeHome = Regex("""\b(whole|entire|every|all|everywhere|house|home)\b""")

    fun apply(transcript: String, call: NluOutcome.Call): Correction? {
        val t = transcript.lowercase()
        if (call.name == HomeTools.SET_LIGHT && temperatureWords.containsMatchIn(t)) {
            val number = call.args[HomeTools.Args.BRIGHTNESS]?.let { IntentMapper.number(it) } ?: numberIn(transcript) ?: return null
            val args = buildMap {
                put(HomeTools.Args.TEMPERATURE, formatNumber(number))
                val room = call.args[HomeTools.Args.ROOM]
                if (room != null && !wholeHome.containsMatchIn(t)) put(HomeTools.Args.ROOM, room)
            }
            return Correction(call, call.copy(name = HomeTools.SET_THERMOSTAT, args = args), "transcript names a temperature, not a light level")
        }
        return null
    }

    /** The first number in free text: digits, or the longest run of up to three number words ("twenty two", "nineteen and a half"). */
    fun numberIn(text: String): Double? {
        Regex("""-?\d+(\.\d+)?""").find(text)?.let { return it.value.toDouble() }
        val words = text.lowercase().replace(Regex("[^a-z0-9. ]+"), " ").split(' ').filter { it.isNotEmpty() }
        for (i in words.indices) for (len in minOf(4, words.size - i) downTo 1) {
            IntentMapper.number(words.subList(i, i + len).joinToString(" "))?.let { return it }
        }
        return null
    }

    private fun formatNumber(d: Double): String = if (d == d.toLong().toDouble()) d.toLong().toString() else d.toString()
}
