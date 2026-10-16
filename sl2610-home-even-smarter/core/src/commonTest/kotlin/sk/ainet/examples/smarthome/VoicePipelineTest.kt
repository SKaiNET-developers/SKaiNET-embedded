package sk.ainet.examples.smarthome

import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runTest
import sk.ainet.examples.smarthome.actions.HomeActions
import sk.ainet.examples.smarthome.actions.HomeStore
import sk.ainet.examples.smarthome.engine.NluEngine
import sk.ainet.examples.smarthome.engine.NluOutcome
import sk.ainet.examples.smarthome.fakes.FakeAsrEngine
import sk.ainet.examples.smarthome.fakes.KeywordNluEngine
import sk.ainet.examples.smarthome.fakes.SyntheticAudioSource
import sk.ainet.examples.smarthome.home.Room
import sk.ainet.examples.smarthome.hybrid.EscalationResult
import sk.ainet.examples.smarthome.hybrid.IntentEscalation
import sk.ainet.examples.smarthome.pipeline.PipelineEvent
import sk.ainet.examples.smarthome.pipeline.Stage
import sk.ainet.examples.smarthome.pipeline.VoicePipeline
import sk.ainet.examples.smarthome.tools.HomeTools
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

@OptIn(ExperimentalCoroutinesApi::class)
class VoicePipelineTest {

    private fun pipeline(nlu: NluEngine = KeywordNluEngine(), escalation: IntentEscalation = IntentEscalation.None, store: HomeStore = HomeStore(), script: String = "turn on the kitchen light") =
        VoicePipeline(
            asr = FakeAsrEngine({ script }), nlu = nlu, router = HomeActions(store).router(), escalation = escalation,
            asrDispatcher = UnconfinedTestDispatcher(), nluDispatcher = UnconfinedTestDispatcher(),
        )

    @Test
    fun `spoken utterance flows through every stage in order`() = runTest {
        val store = HomeStore()
        val p = pipeline(store = store)
        val events = mutableListOf<PipelineEvent>()
        val collector = launch(UnconfinedTestDispatcher(testScheduler)) { p.events.collect { events += it } }

        val run = p.runUtterance(SyntheticAudioSource(frames = 6, frameDelayMs = 0).frames())

        assertEquals("turn on the kitchen light", run.transcript)
        assertEquals(HomeTools.SET_LIGHT, run.calledTool)
        assertTrue(run.action!!.ok)
        assertTrue(store.current.room(Room.KITCHEN).light.on)
        assertNull(run.escalation)

        val kinds = events.map { it::class.simpleName }
        assertEquals("Started", kinds.first())
        assertTrue(kinds.indexOf("Partial") < kinds.indexOf("Transcribed"))
        assertTrue(kinds.indexOf("Transcribed") < kinds.indexOf("Resolving"))
        assertTrue(kinds.indexOf("Resolving") < kinds.indexOf("Resolved"))
        assertTrue(kinds.indexOf("Resolved") < kinds.indexOf("Acted"))
        assertEquals("Finished", kinds.last())
        val partials = events.filterIsInstance<PipelineEvent.Partial>().map { it.text }
        assertEquals(listOf("turn", "turn on", "turn on the", "turn on the kitchen", "turn on the kitchen light"), partials)
        assertEquals(480L, events.filterIsInstance<PipelineEvent.Transcribed>().single().audioMs)
        collector.cancel()
    }

    @Test
    fun `typed text skips the speech stage`() = runTest {
        val run = pipeline().runText("close the bedroom blinds")
        assertEquals(HomeTools.SET_BLINDS, run.calledTool)
        assertEquals(0, run.asrMs)
    }

    @Test
    fun `no call escalates through the seam`() = runTest {
        val seen = mutableListOf<String>()
        val escalation = object : IntentEscalation {
            override suspend fun escalate(transcript: String, outcome: NluOutcome): EscalationResult {
                seen += transcript; return EscalationResult.Handled("It is sunny.", "test-cloud")
            }
        }
        val p = pipeline(escalation = escalation)
        val events = mutableListOf<PipelineEvent>()
        val collector = launch(UnconfinedTestDispatcher(testScheduler)) { p.events.collect { events += it } }
        val run = p.runText("what is the weather like")
        assertIs<NluOutcome.NoCall>(run.outcome)
        assertEquals(listOf("what is the weather like"), seen)
        assertIs<EscalationResult.Handled>(run.escalation)
        assertNotNull(events.filterIsInstance<PipelineEvent.Escalated>().singleOrNull())
        collector.cancel()
    }

    @Test
    fun `an unmappable call is acted on softly and escalated`() = runTest {
        val nlu = object : NluEngine {
            override val id = "stub"; override val toolNames = HomeTools.names
            override fun warmUp() {}; override fun close() {}
            override fun resolve(transcript: String, budgetMs: Long) = NluOutcome.Call(HomeTools.SET_LIGHT, mapOf("room" to "garage", "state" to "on"), "", NluOutcome.Timing.NONE)
        }
        val run = pipeline(nlu = nlu).runText("garage light on")
        assertEquals(false, run.action!!.ok)
        assertIs<EscalationResult.NotHandled>(run.escalation)
    }

    @Test
    fun `a host rule repairs a temperature mistaken for brightness`() = runTest {
        val nlu = object : NluEngine {
            override val id = "stub"; override val toolNames = HomeTools.names
            override fun warmUp() {}; override fun close() {}
            override fun resolve(transcript: String, budgetMs: Long) = NluOutcome.Call(HomeTools.SET_LIGHT, mapOf("room" to "bedroom", "state" to "off", "brightness" to "21"), "", NluOutcome.Timing.NONE)
        }
        val store = HomeStore()
        val p = pipeline(nlu = nlu, store = store)
        val events = mutableListOf<PipelineEvent>()
        val collector = launch(UnconfinedTestDispatcher(testScheduler)) { p.events.collect { events += it } }
        val run = p.runText("Set the bedroom to twenty one degrees")
        assertEquals(HomeTools.SET_THERMOSTAT, run.calledTool)
        assertEquals(21.0, store.current.room(Room.BEDROOM).thermostat.targetCelsius)
        assertEquals(false, store.current.room(Room.BEDROOM).light.on)
        assertNotNull(events.filterIsInstance<PipelineEvent.Corrected>().singleOrNull())
        collector.cancel()
    }

    @Test
    fun `an engine exception is a failed outcome and not a crash`() = runTest {
        val nlu = object : NluEngine {
            override val id = "stub"; override val toolNames = HomeTools.names
            override fun warmUp() {}; override fun close() {}
            override fun resolve(transcript: String, budgetMs: Long): NluOutcome = throw IllegalStateException("runtime gone")
        }
        val run = pipeline(nlu = nlu).runText("kitchen light on")
        assertEquals("runtime gone", run.failure)
    }

    @Test
    fun `empty transcript fails at the speech stage`() = runTest {
        val p = pipeline(script = "")
        val events = mutableListOf<PipelineEvent>()
        val collector = launch(UnconfinedTestDispatcher(testScheduler)) { p.events.collect { events += it } }
        val run = p.runUtterance(SyntheticAudioSource(frames = 2, frameDelayMs = 0).frames())
        assertEquals("nothing recognized", run.failure)
        assertEquals(Stage.ASR, events.filterIsInstance<PipelineEvent.Failed>().single().stage)
        collector.cancel()
    }
}
