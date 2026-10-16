package sk.ainet.examples.smarthome

import sk.ainet.examples.smarthome.home.BlindPosition
import sk.ainet.examples.smarthome.home.Door
import sk.ainet.examples.smarthome.home.HomeCommand
import sk.ainet.examples.smarthome.home.Room
import sk.ainet.examples.smarthome.home.Scene
import sk.ainet.examples.smarthome.tools.HomeTools
import sk.ainet.examples.smarthome.tools.IntentMapper
import sk.ainet.examples.smarthome.tools.Mapping
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertNull

class IntentMapperTest {
    private fun command(name: String, vararg args: Pair<String, String>): HomeCommand =
        assertIs<Mapping.Command>(IntentMapper.map(name, args.toMap())).command

    @Test
    fun `room synonyms`() {
        assertEquals(Room.LIVING_ROOM, IntentMapper.room("living_room"))
        assertEquals(Room.LIVING_ROOM, IntentMapper.room("Living Room"))
        assertEquals(Room.LIVING_ROOM, IntentMapper.room("the lounge"))
        assertEquals(Room.BEDROOM, IntentMapper.room("bed room"))
        assertEquals(Room.BATHROOM, IntentMapper.room("bath"))
        assertEquals(Room.HALLWAY, IntentMapper.room("hall"))
        assertEquals(Room.HALLWAY, IntentMapper.room("corridor"))
        assertNull(IntentMapper.room("garage"))
        assertNull(IntentMapper.room(""))
    }

    @Test
    fun `numbers in digits and words`() {
        assertEquals(21.0, IntentMapper.number("21"))
        assertEquals(21.5, IntentMapper.number("21.5 degrees"))
        assertEquals(21.0, IntentMapper.number("twenty one"))
        assertEquals(21.0, IntentMapper.number("twenty-one"))
        assertEquals(19.5, IntentMapper.number("nineteen and a half"))
        assertEquals(100.0, IntentMapper.number("a hundred percent"))
        assertEquals(50.0, IntentMapper.number("fifty"))
        assertNull(IntentMapper.number("warm"))
    }

    @Test
    fun `set_light maps state and optional brightness`() {
        assertEquals(HomeCommand.SetLight(Room.KITCHEN, true, null), command(HomeTools.SET_LIGHT, "room" to "kitchen", "state" to "on"))
        assertEquals(HomeCommand.SetLight(Room.KITCHEN, true, 100), command(HomeTools.SET_LIGHT, "room" to "kitchen", "state" to "ON", "brightness" to "150"))
        assertEquals(HomeCommand.SetLight(Room.BEDROOM, false, null), command(HomeTools.SET_LIGHT, "room" to "bedroom", "state" to "off"))
        assertIs<Mapping.Unmapped>(IntentMapper.map(HomeTools.SET_LIGHT, mapOf("state" to "on")))
        assertIs<Mapping.Unmapped>(IntentMapper.map(HomeTools.SET_LIGHT, mapOf("room" to "kitchen", "state" to "dim")))
    }

    @Test
    fun `set_thermostat with and without room`() {
        assertEquals(HomeCommand.SetThermostat(Room.BEDROOM, 21.0), command(HomeTools.SET_THERMOSTAT, "room" to "bedroom", "temperature" to "twenty one"))
        assertEquals(HomeCommand.SetThermostat(null, 19.0), command(HomeTools.SET_THERMOSTAT, "temperature" to "19"))
        assertIs<Mapping.Unmapped>(IntentMapper.map(HomeTools.SET_THERMOSTAT, mapOf("room" to "bedroom")))
    }

    @Test
    fun `blinds and locks and scenes`() {
        assertEquals(HomeCommand.SetBlinds(Room.LIVING_ROOM, BlindPosition.CLOSED), command(HomeTools.SET_BLINDS, "room" to "living room", "position" to "close"))
        assertEquals(HomeCommand.SetBlinds(Room.KITCHEN, BlindPosition.HALF), command(HomeTools.SET_BLINDS, "room" to "kitchen", "position" to "halfway"))
        assertEquals(HomeCommand.SetLock(Door.FRONT_DOOR, true), command(HomeTools.SET_LOCK, "door" to "front_door", "state" to "locked"))
        assertEquals(HomeCommand.SetLock(Door.BACK_DOOR, false), command(HomeTools.SET_LOCK, "door" to "back door", "state" to "unlock"))
        assertEquals(HomeCommand.SetLock(Door.FRONT_DOOR, true), command(HomeTools.SET_LOCK, "state" to "locked"))
        assertEquals(HomeCommand.SetScene(Scene.MOVIE), command(HomeTools.SET_SCENE, "scene" to "movie"))
        assertEquals(HomeCommand.SetScene(Scene.NIGHT), command(HomeTools.SET_SCENE, "scene" to "good night"))
        assertEquals(HomeCommand.GetStatus(null), command(HomeTools.GET_STATUS))
        assertEquals(HomeCommand.GetStatus(Room.HALLWAY), command(HomeTools.GET_STATUS, "room" to "hallway"))
    }

    @Test
    fun `unknown function is unmapped and not an exception`() {
        val m = IntentMapper.map("order_pizza", mapOf("size" to "large"))
        assertIs<Mapping.Unmapped>(m)
    }
}
