package voicecc

import voicecc.actions.Intent
import voicecc.actions.defaultRouter
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ActionRouterTest {
    @Test
    fun setLightsFormatsArgs() {
        val r = defaultRouter().dispatch(Intent("set_lights", mapOf("color" to "red", "state" to "on")))
        assertTrue(r.ok)
        assertEquals("color=red state=on", r.message)
    }

    @Test
    fun unknownToolFails() {
        val r = defaultRouter().dispatch(Intent("teleport", mapOf("x" to 1)))
        assertTrue(!r.ok)
    }

    @Test
    fun allSixToolsRegistered() {
        assertEquals(
            listOf("cancel_alarm", "get_system_status", "play_buzzer", "respond", "set_alarm", "set_lights"),
            defaultRouter().tools,
        )
    }
}
