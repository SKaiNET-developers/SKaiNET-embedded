package sk.ainet.examples.smarthome.engine

import kotlinx.coroutines.flow.Flow

/**
 * Streaming speech-to-text, as the app sees it. Mirrors the ASR cartridge contract one-to-one so the adapter is
 * trivial: 16 kHz mono float PCM in, cumulative partials out, one exact final on [AsrSession.finish].
 */
interface AsrEngine : AutoCloseable {
    val id: String
    val language: String
    val sampleRateHz: Int get() = 16_000
    fun open(): AsrSession
}

interface AsrSession : AutoCloseable {
    /** Feed audio in [-1, 1]. Returns the updated cumulative partial, or `null` when nothing changed. */
    fun feed(pcm: FloatArray): String?
    /** End of utterance: the final transcript (empty when nothing was recognized). The session is reusable. */
    fun finish(): String
    /** Abort the current utterance. */
    fun reset()
    override fun close() = reset()
}

/** One transcript in, one outcome out. Mirrors the NLU cartridge contract (function name + string arguments). */
interface NluEngine : AutoCloseable {
    val id: String
    val toolNames: Set<String>
    /** Loads the runtime and prefills the catalog prefix. Expensive; idempotent. */
    fun warmUp()
    fun resolve(transcript: String, budgetMs: Long): NluOutcome
}

sealed interface NluOutcome {
    data class Call(val name: String, val args: Map<String, String>, val raw: String, val timing: Timing) : NluOutcome
    /** The model answered without calling a function (prose, refusal, unparsable call). */
    data class NoCall(val text: String, val timing: Timing) : NluOutcome
    data class Failed(val reason: String, val cause: Throwable? = null, val timing: Timing? = null) : NluOutcome

    /** Wall-clock breakdown of one resolve, in milliseconds — the pipeline panel draws these bars. */
    data class Timing(val tokenizeMs: Long, val restoreMs: Long, val chunkMs: Long, val decodeMs: Long, val decodeTokens: Int) {
        val totalMs: Long get() = tokenizeMs + restoreMs + chunkMs + decodeMs
        companion object { val NONE = Timing(0, 0, 0, 0, 0) }
    }
}

/** Microphone or replay: frames of mono float PCM at [sampleRateHz]. The flow completes when the source ends. */
interface AudioSource {
    val sampleRateHz: Int get() = 16_000
    fun frames(): Flow<FloatArray>
}

/** Lifecycle of one engine as shown on the cartridge screen. */
sealed interface EngineStatus {
    data object Absent : EngineStatus
    data class Missing(val reason: String) : EngineStatus
    data class Loading(val phase: String, val elapsedMs: Long) : EngineStatus
    data class Ready(val warmUpMs: Long, val detail: String = "") : EngineStatus
    data class Failed(val reason: String) : EngineStatus
}
