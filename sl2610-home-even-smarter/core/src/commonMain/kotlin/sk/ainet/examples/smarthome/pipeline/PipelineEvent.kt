package sk.ainet.examples.smarthome.pipeline

import sk.ainet.examples.smarthome.actions.ActionResult
import sk.ainet.examples.smarthome.engine.NluOutcome
import sk.ainet.examples.smarthome.hybrid.EscalationResult
import sk.ainet.examples.smarthome.tools.Correction

/** The stages of one utterance as the pipeline panel shows them. */
enum class Stage { LISTENING, ASR, NLU, ACTION, CLOUD }

/** Everything the UI needs to visualise a run. [atMs] is the pipeline clock (monotonic, ms). */
sealed interface PipelineEvent {
    val atMs: Long

    data class Started(override val atMs: Long, val source: String) : PipelineEvent
    data class Level(override val atMs: Long, val rms: Float) : PipelineEvent
    data class Partial(override val atMs: Long, val text: String) : PipelineEvent
    data class Transcribed(override val atMs: Long, val text: String, val audioMs: Long, val asrMs: Long) : PipelineEvent
    data class Resolving(override val atMs: Long, val text: String) : PipelineEvent
    data class Resolved(override val atMs: Long, val outcome: NluOutcome, val nluMs: Long) : PipelineEvent
    /** A host rule rewrote the call before it was acted on. */
    data class Corrected(override val atMs: Long, val correction: Correction) : PipelineEvent
    data class Escalated(override val atMs: Long, val reason: String, val result: EscalationResult) : PipelineEvent
    data class Acted(override val atMs: Long, val result: ActionResult) : PipelineEvent
    data class Failed(override val atMs: Long, val stage: Stage, val reason: String) : PipelineEvent
    data class Finished(override val atMs: Long, val totalMs: Long) : PipelineEvent
}

/** Coarse state for enabling and disabling the talk button. */
enum class PipelineState { IDLE, LISTENING, TRANSCRIBING, RESOLVING, ACTING }

/** The summary of one run, for the event log and the golden-set runner. */
data class PipelineRun(
    val transcript: String,
    val outcome: NluOutcome?,
    val correction: Correction? = null,
    val action: ActionResult?,
    val escalation: EscalationResult?,
    val asrMs: Long,
    val nluMs: Long,
    val totalMs: Long,
    val failure: String? = null,
) {
    /** The function that was acted on: the corrected one when a host rule applied. */
    val calledTool: String? get() = (correction?.to ?: outcome as? NluOutcome.Call)?.name
}
