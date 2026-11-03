package sk.ainet.examples.smarthome

import kotlinx.coroutines.flow.asFlow
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import sk.ainet.examples.smarthome.pipeline.Endpointer
import kotlin.test.Test
import kotlin.test.assertEquals

class EndpointerTest {
    private fun frame(amplitude: Float) = FloatArray(1600) { amplitude } // 100 ms at 16 kHz

    @Test
    fun `completes after silence that follows speech`() = runTest {
        val frames = List(4) { frame(0.2f) } + List(20) { frame(0f) }
        val out = Endpointer(silenceMs = 700, minSpeechMs = 300).bound(frames.asFlow()).toList()
        assertEquals(4 + 7, out.size)
    }

    @Test
    fun `silence before any speech does not end the utterance`() = runTest {
        val frames = List(15) { frame(0f) } + List(3) { frame(0.2f) } + List(7) { frame(0f) } + List(5) { frame(0.2f) }
        val out = Endpointer(silenceMs = 700, minSpeechMs = 300).bound(frames.asFlow()).toList()
        assertEquals(15 + 3 + 7, out.size)
    }

    @Test
    fun `upstream completion ends the utterance`() = runTest {
        val out = Endpointer().bound(List(3) { frame(0.2f) }.asFlow()).toList()
        assertEquals(3, out.size)
    }

    @Test
    fun `maximum utterance length`() = runTest {
        val out = Endpointer(maxUtteranceMs = 1000).bound(List(50) { frame(0.2f) }.asFlow()).toList()
        assertEquals(10, out.size)
    }
}
