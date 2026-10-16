package sk.ainet.examples.smarthome.pipeline

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * Energy-based utterance end detection on top of push-to-talk: the flow of frames completes after [silenceMs] of
 * quiet once at least [minSpeechMs] of speech was heard, or when the upstream completes (button released).
 * The ASR cartridge itself never decides where an utterance ends — that is the host's call, by design.
 */
class Endpointer(
    val sampleRateHz: Int = 16_000,
    val thresholdRms: Float = 0.015f,
    val silenceMs: Long = 700,
    val minSpeechMs: Long = 300,
    val maxUtteranceMs: Long = 15_000,
) {
    fun bound(frames: Flow<FloatArray>): Flow<FloatArray> = flow {
        var speechMs = 0L
        var silence = 0L
        var total = 0L
        try {
            frames.collect { frame ->
                val frameMs = frame.size * 1000L / sampleRateHz
                total += frameMs
                emit(frame)
                if (VoicePipeline.rms(frame) >= thresholdRms) { speechMs += frameMs; silence = 0 } else silence += frameMs
                if ((speechMs >= minSpeechMs && silence >= silenceMs) || total >= maxUtteranceMs) throw EndOfUtterance
            }
        } catch (_: EndOfUtterance) {
        }
    }

    private object EndOfUtterance : RuntimeException() {
        private fun readResolve(): Any = EndOfUtterance
    }
}
