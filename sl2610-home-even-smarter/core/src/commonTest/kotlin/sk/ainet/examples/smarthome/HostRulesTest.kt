package sk.ainet.examples.smarthome

import sk.ainet.examples.smarthome.engine.NluOutcome
import sk.ainet.examples.smarthome.tools.HomeTools
import sk.ainet.examples.smarthome.tools.HostRules
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

class HostRulesTest {
    private fun light(vararg args: Pair<String, String>) = NluOutcome.Call(HomeTools.SET_LIGHT, args.toMap(), "", NluOutcome.Timing.NONE)

    @Test
    fun `degrees in a light call become a thermostat call for the room`() {
        val c = assertNotNull(HostRules.apply("Set the bedroom to twenty one degrees", light("brightness" to "21", "room" to "bedroom", "state" to "off")))
        assertEquals(HomeTools.SET_THERMOSTAT, c.to.name)
        assertEquals(mapOf("temperature" to "21", "room" to "bedroom"), c.to.args)
        assertEquals(HomeTools.SET_LIGHT, c.from.name)
    }

    @Test
    fun `whole house drops the room`() {
        val c = assertNotNull(HostRules.apply("Make it nineteen degrees in the whole house", light("brightness" to "19", "room" to "living_room", "state" to "on")))
        assertEquals(mapOf("temperature" to "19"), c.to.args)
    }

    @Test
    fun `number taken from the transcript when brightness is not numeric`() {
        val c = assertNotNull(HostRules.apply("warm the kitchen up to twenty two degrees", light("brightness" to "on", "room" to "kitchen", "state" to "on")))
        assertEquals("22", c.to.args["temperature"])
    }

    @Test
    fun `plain light commands are untouched`() {
        assertNull(HostRules.apply("Dim the living room light to twenty percent", light("brightness" to "20", "room" to "living_room", "state" to "on")))
        assertNull(HostRules.apply("Turn on the kitchen light", light("room" to "kitchen", "state" to "on")))
        assertNull(HostRules.apply("Lock the front door", NluOutcome.Call(HomeTools.SET_LOCK, mapOf("door" to "front_door", "state" to "locked"), "", NluOutcome.Timing.NONE)))
    }
}
