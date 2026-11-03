package sk.ainet.examples.smarthome.pipeline

import sk.ainet.examples.smarthome.tools.HomeTools

/** One rehearsed utterance and the function it must resolve to (`null` = must not call a home function). */
data class GoldenCase(val utterance: String, val expectedTool: String?)

/**
 * The utterances spoken on stage, also the typed regression set: run them through the NLU alone to separate
 * recognition errors from understanding errors. Ten in-catalog, two deliberately outside it (they must land on the
 * escalation seam, not on a wrong function).
 */
object GoldenSet {
    val cases: List<GoldenCase> = listOf(
        GoldenCase("Turn on the kitchen light", HomeTools.SET_LIGHT),
        GoldenCase("Switch off the bedroom light", HomeTools.SET_LIGHT),
        GoldenCase("Dim the living room light to twenty percent", HomeTools.SET_LIGHT),
        GoldenCase("Set the bedroom to twenty one degrees", HomeTools.SET_THERMOSTAT),
        GoldenCase("Make it nineteen degrees in the whole house", HomeTools.SET_THERMOSTAT),
        GoldenCase("Close the living room blinds", HomeTools.SET_BLINDS),
        GoldenCase("Open the kitchen blinds halfway", HomeTools.SET_BLINDS),
        GoldenCase("Lock the front door", HomeTools.SET_LOCK),
        GoldenCase("Movie time", HomeTools.SET_SCENE),
        GoldenCase("What is the status of the hallway", HomeTools.GET_STATUS),
        GoldenCase("What is the weather like tomorrow", null),
        GoldenCase("Order a pizza for dinner", null),
    )
}
