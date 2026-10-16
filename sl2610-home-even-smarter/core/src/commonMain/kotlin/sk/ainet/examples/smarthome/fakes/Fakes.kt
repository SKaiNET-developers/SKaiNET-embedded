package sk.ainet.examples.smarthome.fakes

import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import sk.ainet.examples.smarthome.engine.AsrEngine
import sk.ainet.examples.smarthome.engine.AsrSession
import sk.ainet.examples.smarthome.engine.AudioSource
import sk.ainet.examples.smarthome.engine.NluEngine
import sk.ainet.examples.smarthome.engine.NluOutcome
import sk.ainet.examples.smarthome.tools.HomeTools

/**
 * A speech engine that "hears" [script]: every feed reveals one more word of the utterance as a partial, and
 * `finish()` returns the whole sentence. Lets the UI and the pipeline run on a laptop without any cartridge.
 */
class FakeAsrEngine(private val script: () -> String, override val language: String = "en") : AsrEngine {
    override val id: String = "fake-asr"
    override fun open(): AsrSession = Session()
    override fun close() {}

    private inner class Session : AsrSession {
        private var words: List<String> = emptyList()
        private var revealed = 0
        private var started = false
        private fun start() { if (!started) { words = script().split(' ').filter { it.isNotBlank() }; started = true } }
        override fun feed(pcm: FloatArray): String? {
            start()
            if (revealed >= words.size) return null
            revealed++
            return words.take(revealed).joinToString(" ")
        }
        override fun finish(): String { start(); val text = words.joinToString(" "); reset(); return text }
        override fun reset() { words = emptyList(); revealed = 0; started = false }
    }
}

/** A rule-based stand-in for the language model: regexes over the six home functions. Same outcome type, so the rest of the app cannot tell. */
class KeywordNluEngine(private val simulatedLatencyMs: Long = 0) : NluEngine {
    override val id: String = "keyword-nlu"
    override val toolNames: Set<String> = HomeTools.names
    override fun warmUp() {}
    override fun close() {}

    override fun resolve(transcript: String, budgetMs: Long): NluOutcome {
        val t = transcript.lowercase()
        val timing = NluOutcome.Timing(1, 0, simulatedLatencyMs / 2, simulatedLatencyMs / 2, 8)
        fun call(name: String, vararg args: Pair<String, String?>): NluOutcome {
            val a = args.mapNotNull { (k, v) -> v?.let { k to it } }.toMap()
            return NluOutcome.Call(name, a, "$name(${a.entries.joinToString { "${it.key}=\"${it.value}\"" }})", timing)
        }
        val room = ROOM_WORDS.entries.firstOrNull { (k, _) -> k in t }?.value
        val number = Regex("""\d+(\.\d+)?""").find(t)?.value
        return when {
            Regex("""\b(status|state|how is|what.s)\b""").containsMatchIn(t) -> call(HomeTools.GET_STATUS, "room" to room)
            Regex("""\bscene\b|\bmovie\b|\bgood night\b|\bgood morning\b|\bleaving\b|\baway\b""").containsMatchIn(t) -> {
                val scene = when { "movie" in t -> "movie"; "night" in t -> "night"; "morning" in t -> "morning"; else -> "away" }
                call(HomeTools.SET_SCENE, "scene" to scene)
            }
            Regex("""\b(lock|unlock)\b""").containsMatchIn(t) -> {
                val door = if ("back" in t) "back_door" else "front_door"
                call(HomeTools.SET_LOCK, "door" to door, "state" to if ("unlock" in t) "unlocked" else "locked")
            }
            Regex("""\bblinds?\b|\bshades?\b|\bcurtains?\b""").containsMatchIn(t) -> {
                val pos = when { "half" in t -> "half"; "open" in t || "up" in t -> "open"; else -> "closed" }
                call(HomeTools.SET_BLINDS, "room" to room, "position" to pos)
            }
            Regex("""\bdegrees?\b|\btemperature\b|\bheat\b|\bthermostat\b|\bwarmer\b|\bcolder\b""").containsMatchIn(t) ->
                call(HomeTools.SET_THERMOSTAT, "room" to room, "temperature" to (number ?: "21"))
            Regex("""\blights?\b|\blamp\b""").containsMatchIn(t) -> {
                val off = Regex("""\b(off|out)\b""").containsMatchIn(t)
                call(HomeTools.SET_LIGHT, "room" to room, "state" to if (off) "off" else "on", "brightness" to number?.takeIf { "percent" in t || "%" in t })
            }
            else -> NluOutcome.NoCall("I did not find a home function for that.", timing)
        }
    }

    companion object {
        private val ROOM_WORDS = linkedMapOf(
            "living" to "living_room", "lounge" to "living_room", "kitchen" to "kitchen", "bedroom" to "bedroom",
            "bath" to "bathroom", "hall" to "hallway", "corridor" to "hallway",
        )
    }
}

/** Frames of a fixed duration (useful with [FakeAsrEngine]: one frame reveals one word). */
class SyntheticAudioSource(private val frames: Int, private val frameSamples: Int = 1280, private val frameDelayMs: Long = 80, private val amplitude: Float = 0.1f) : AudioSource {
    override fun frames(): Flow<FloatArray> = flow {
        repeat(frames) { i ->
            val f = FloatArray(frameSamples) { n -> amplitude * kotlin.math.sin((i * frameSamples + n) * 0.05f) }
            emit(f)
            if (frameDelayMs > 0) delay(frameDelayMs)
        }
    }
}

/** Replays a 16-bit PCM mono WAV (16 kHz) in 80 ms frames, optionally in real time. */
class ReplayAudioSource(private val wavBytes: ByteArray, private val realTime: Boolean = true, private val frameSamples: Int = 1280) : AudioSource {
    private val pcm: FloatArray = Wav.decode16BitMono(wavBytes)
    override fun frames(): Flow<FloatArray> = flow {
        var i = 0
        while (i < pcm.size) {
            val n = minOf(frameSamples, pcm.size - i)
            emit(pcm.copyOfRange(i, i + n))
            i += n
            if (realTime) delay(n * 1000L / sampleRateHz)
        }
    }
}

/** Minimal RIFF/WAVE reader for 16-bit little-endian PCM. Multi-channel input is averaged to mono. */
object Wav {
    fun decode16BitMono(bytes: ByteArray): FloatArray {
        require(bytes.size > 44 && bytes.decodeToString(0, 4) == "RIFF" && bytes.decodeToString(8, 12) == "WAVE") { "not a RIFF/WAVE file" }
        var pos = 12
        var channels = 1
        var bits = 16
        while (pos + 8 <= bytes.size) {
            val id = bytes.decodeToString(pos, pos + 4)
            val size = le32(bytes, pos + 4)
            val body = pos + 8
            when (id) {
                "fmt " -> { channels = le16(bytes, body + 2); bits = le16(bytes, body + 14) }
                "data" -> {
                    require(bits == 16) { "only 16-bit PCM is supported, got $bits" }
                    val frames = minOf(size, bytes.size - body) / (2 * channels)
                    return FloatArray(frames) { f ->
                        var acc = 0
                        for (c in 0 until channels) acc += le16s(bytes, body + (f * channels + c) * 2)
                        (acc.toFloat() / channels) / 32768f
                    }
                }
            }
            pos = body + size + (size and 1)
        }
        error("no data chunk")
    }

    private fun le16(b: ByteArray, i: Int): Int = (b[i].toInt() and 0xFF) or ((b[i + 1].toInt() and 0xFF) shl 8)
    private fun le16s(b: ByteArray, i: Int): Int = le16(b, i).toShort().toInt()
    private fun le32(b: ByteArray, i: Int): Int = le16(b, i) or (le16(b, i + 2) shl 16)
}
