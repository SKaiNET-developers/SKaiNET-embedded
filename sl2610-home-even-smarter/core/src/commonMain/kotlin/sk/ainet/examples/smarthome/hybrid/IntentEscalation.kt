package sk.ainet.examples.smarthome.hybrid

import sk.ainet.examples.smarthome.engine.NluOutcome

/**
 * The seam for the hybrid step that follows this demo: when the on-device NLU abstains ([NluOutcome.NoCall])
 * or cannot answer, the pipeline hands the transcript here before giving up. Today the only implementation is
 * [None]; a cloud-backed resolver (weather, open questions, choices) plugs in without touching the pipeline.
 */
interface IntentEscalation {
    suspend fun escalate(transcript: String, outcome: NluOutcome): EscalationResult

    object None : IntentEscalation {
        override suspend fun escalate(transcript: String, outcome: NluOutcome): EscalationResult =
            EscalationResult.NotHandled("no escalation configured")
    }
}

sealed interface EscalationResult {
    data class Handled(val message: String, val source: String) : EscalationResult
    data class NotHandled(val reason: String) : EscalationResult
}
