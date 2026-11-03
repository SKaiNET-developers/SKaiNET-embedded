package sk.ainet.examples.smarthome

import sk.ainet.examples.smarthome.home.BlindPosition
import sk.ainet.examples.smarthome.home.DeviceRef
import sk.ainet.examples.smarthome.home.Door
import sk.ainet.examples.smarthome.home.HomeCommand
import sk.ainet.examples.smarthome.home.HomeReducer
import sk.ainet.examples.smarthome.home.HomeState
import sk.ainet.examples.smarthome.home.Room
import sk.ainet.examples.smarthome.home.Scene
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class HomeReducerTest {
    private val initial = HomeState.initial()

    @Test
    fun `light on with brightness changes one room only`() {
        val r = HomeReducer.apply(initial, HomeCommand.SetLight(Room.KITCHEN, on = true, brightness = 40))
        assertTrue(r.state.room(Room.KITCHEN).light.on)
        assertEquals(40, r.state.room(Room.KITCHEN).light.brightness)
        assertFalse(r.state.room(Room.BEDROOM).light.on)
        assertEquals(setOf(DeviceRef.LightOf(Room.KITCHEN)), r.changed)
        assertEquals("Kitchen light on at 40%", r.message)
    }

    @Test
    fun `light off keeps the last brightness and brightness is clamped`() {
        val on = HomeReducer.apply(initial, HomeCommand.SetLight(Room.KITCHEN, true, 250)).state
        assertEquals(100, on.room(Room.KITCHEN).light.brightness)
        val off = HomeReducer.apply(on, HomeCommand.SetLight(Room.KITCHEN, false)).state
        assertFalse(off.room(Room.KITCHEN).light.on)
        assertEquals(100, off.room(Room.KITCHEN).light.brightness)
    }

    @Test
    fun `thermostat without room sets every room`() {
        val r = HomeReducer.apply(initial, HomeCommand.SetThermostat(null, 22.0))
        assertTrue(Room.entries.all { r.state.room(it).thermostat.targetCelsius == 22.0 })
        assertEquals(Room.entries.size, r.changed.size)
        assertEquals("Whole home set to 22 °C", r.message)
        val one = HomeReducer.apply(initial, HomeCommand.SetThermostat(Room.BEDROOM, 18.5))
        assertEquals("Bedroom set to 18.5 °C", one.message)
        assertEquals(20.0, one.state.room(Room.KITCHEN).thermostat.targetCelsius)
    }

    @Test
    fun `blinds and locks`() {
        val b = HomeReducer.apply(initial, HomeCommand.SetBlinds(Room.LIVING_ROOM, BlindPosition.HALF))
        assertEquals(BlindPosition.HALF, b.state.room(Room.LIVING_ROOM).blinds)
        val l = HomeReducer.apply(b.state, HomeCommand.SetLock(Door.BACK_DOOR, true))
        assertEquals(true, l.state.locks[Door.BACK_DOOR])
        assertEquals(false, l.state.locks[Door.FRONT_DOOR])
        assertEquals("Back door locked", l.message)
    }

    @Test
    fun `night scene turns everything off then closes blinds and locks doors`() {
        val lit = HomeReducer.apply(initial, HomeCommand.SetLight(Room.HALLWAY, true)).state
        val r = HomeReducer.apply(lit, HomeCommand.SetScene(Scene.NIGHT))
        assertTrue(Room.entries.none { r.state.room(it).light.on })
        assertTrue(Room.entries.all { r.state.room(it).blinds == BlindPosition.CLOSED })
        assertTrue(Door.entries.all { r.state.locks[it] == true })
        assertEquals(Scene.NIGHT, r.state.activeScene)
        // a later single-device command leaves the scene
        val after = HomeReducer.apply(r.state, HomeCommand.SetLight(Room.HALLWAY, true)).state
        assertEquals(null, after.activeScene)
    }

    @Test
    fun `movie scene lights the living room dimly only`() {
        val r = HomeReducer.apply(initial, HomeCommand.SetScene(Scene.MOVIE)).state
        assertEquals(20, r.room(Room.LIVING_ROOM).light.brightness)
        assertTrue(r.room(Room.LIVING_ROOM).light.on)
        assertFalse(r.room(Room.KITCHEN).light.on)
    }

    @Test
    fun `status changes nothing and describes the room`() {
        val r = HomeReducer.apply(initial, HomeCommand.GetStatus(Room.KITCHEN))
        assertEquals(initial, r.state)
        assertTrue(r.changed.isEmpty())
        assertEquals("Kitchen: light off, 20 °C, blinds open", r.message)
        assertTrue(HomeReducer.apply(initial, HomeCommand.GetStatus(null)).message.contains("front door unlocked"))
    }
}
