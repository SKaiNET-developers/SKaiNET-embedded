package sk.ainet.examples.smarthome.pipeline

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import sk.ainet.examples.smarthome.actions.ActionResult
import sk.ainet.examples.smarthome.actions.ActionRouter
import sk.ainet.examples.smarthome.actions.Intent
import sk.ainet.examples.smarthome.engine.AsrEngine
import sk.ainet.examples.smarthome.engine.NluEngine
import sk.ainet.examples.smarthome.engine.NluOutcome
import sk.ainet.examples.smarthome.hybrid.EscalationResult
import sk.ainet.examples.smarthome.hybrid.IntentEscalation
import sk.ainet.examples.smarthome.tools.Correction
import sk.ainet.examples.smarthome.tools.HostRules
import kotlin.math.sqrt
import kotlin.time.TimeSource

/**
 * mic frames → ASR (partials) → final transcript → NLU → action router → home, with every step published as a
 * [PipelineEvent]. One utterance at a time; the caller decides where an utterance ends by completing the frame flow
 * (push-to-talk release, or an endpointer). [runText] skips the ASR stage for typed input and the golden set.
 */
class VoicePipeline(
    private val asr: AsrEngine?,
    private val nlu: NluEngine,
    private val router: ActionRouter,
    private val escalation: IntentEscalation = IntentEscalation.None,
    private val nluBudgetMs: Long = 15_000,
    private val asrDispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val nluDispatcher: CoroutineDispatcher = Dispatchers.Default,
) {
    private val _events = MutableSharedFlow<PipelineEvent>(extraBufferCapacity = 256)
    val events: SharedFlow<PipelineEvent> = _events.asSharedFlow()

    private val _state = MutableStateFlow(PipelineState.IDLE)
    val state: StateFlow<PipelineState> = _state.asStateFlow()

    private val clock = TimeSource.Monotonic.markNow()
    private fun now(): Long = clock.elapsedNow().inWholeMilliseconds

    private fun emit(e: PipelineEvent) { _events.tryEmit(e) }

    /** Runs one spoken utterance: consumes [frames] until the flow completes, then resolves and acts. */
    suspend fun runUtterance(frames: Flow<FloatArray>, source: String = "mic"): PipelineRun {
        val engine = asr ?: return fail(Stage.ASR, "no speech engine", now())
        val t0 = now()
        emit(PipelineEvent.Started(t0, source))
        _state.value = PipelineState.LISTENING
        var samples = 0L
        val transcript: String
        try {
            val session = engine.open()
            withContext(asrDispatcher) {
                frames.collect { frame ->
                    samples += frame.size
                    emit(PipelineEvent.Level(now(), rms(frame)))
                    session.feed(frame)?.let { emit(PipelineEvent.Partial(now(), it)) }
                }
            }
            _state.value = PipelineState.TRANSCRIBING
            transcript = withContext(asrDispatcher) { session.finish() }.trim()
        } catch (e: Exception) {
            return fail(Stage.ASR, e.message ?: e::class.simpleName ?: "asr failed", t0)
        }
        val asrMs = now() - t0
        val audioMs = samples * 1000 / engine.sampleRateHz
        emit(PipelineEvent.Transcribed(now(), transcript, audioMs, asrMs))
        if (transcript.isEmpty()) return fail(Stage.ASR, "nothing recognized", t0, asrMs = asrMs)
        return resolveAndAct(transcript, t0, asrMs)
    }

    /** Runs the NLU and action stages on typed text (no ASR). */
    suspend fun runText(text: String): PipelineRun {
        val t0 = now()
        emit(PipelineEvent.Started(t0, "text"))
        emit(PipelineEvent.Transcribed(now(), text, 0, 0))
        return resolveAndAct(text.trim(), t0, 0)
    }

    private suspend fun resolveAndAct(transcript: String, t0: Long, asrMs: Long): PipelineRun {
        _state.value = PipelineState.RESOLVING
        emit(PipelineEvent.Resolving(now(), transcript))
        val tNlu = now()
        val outcome: NluOutcome = try {
            withTimeout(nluBudgetMs + 5_000) { withContext(nluDispatcher) { nlu.resolve(transcript, nluBudgetMs) } }
        } catch (e: TimeoutCancellationException) {
            NluOutcome.Failed("nlu exceeded ${nluBudgetMs + 5_000} ms")
        } catch (e: Exception) {
            NluOutcome.Failed(e.message ?: e::class.simpleName ?: "nlu failed", e)
        }
        val nluMs = now() - tNlu
        emit(PipelineEvent.Resolved(now(), outcome, nluMs))

        var action: ActionResult? = null
        var escalated: EscalationResult? = null
        var correction: Correction? = null
        when (outcome) {
            is NluOutcome.Call -> {
                correction = HostRules.apply(transcript, outcome)?.also { emit(PipelineEvent.Corrected(now(), it)) }
                val call = correction?.to ?: outcome
                _state.value = PipelineState.ACTING
                action = router.dispatch(Intent(call.name, call.args))
                emit(PipelineEvent.Acted(now(), action))
                if (!action.ok) escalated = escalate(transcript, outcome, "action rejected: ${action.message}")
            }
            is NluOutcome.NoCall -> escalated = escalate(transcript, outcome, "no function call")
            is NluOutcome.Failed -> escalated = escalate(transcript, outcome, outcome.reason)
        }
        val total = now() - t0
        emit(PipelineEvent.Finished(now(), total))
        _state.value = PipelineState.IDLE
        return PipelineRun(transcript, outcome, correction, action, escalated, asrMs, nluMs, total,
            failure = (outcome as? NluOutcome.Failed)?.reason)
    }

    private suspend fun escalate(transcript: String, outcome: NluOutcome, reason: String): EscalationResult {
        val result = try { escalation.escalate(transcript, outcome) } catch (e: Exception) { EscalationResult.NotHandled(e.message ?: "escalation failed") }
        emit(PipelineEvent.Escalated(now(), reason, result))
        return result
    }

    private fun fail(stage: Stage, reason: String, t0: Long, asrMs: Long = 0): PipelineRun {
        emit(PipelineEvent.Failed(now(), stage, reason))
        val total = now() - t0
        emit(PipelineEvent.Finished(now(), total))
        _state.value = PipelineState.IDLE
        return PipelineRun("", null, null, null, null, asrMs, 0, total, failure = reason)
    }

    companion object {
        fun rms(frame: FloatArray): Float {
            if (frame.isEmpty()) return 0f
            var acc = 0.0
            for (s in frame) acc += s * s
            return sqrt(acc / frame.size).toFloat()
        }
    }
}
