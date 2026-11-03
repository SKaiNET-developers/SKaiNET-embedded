package sk.ainet.examples.smarthome.home

import kotlinx.serialization.Serializable

/** The rooms of the demo home. [id] is the wire value used in the tool catalog. */
enum class Room(val id: String, val label: String) {
    LIVING_ROOM("living_room", "Living room"),
    KITCHEN("kitchen", "Kitchen"),
    BEDROOM("bedroom", "Bedroom"),
    BATHROOM("bathroom", "Bathroom"),
    HALLWAY("hallway", "Hallway");

    companion object {
        fun fromId(id: String): Room? = entries.firstOrNull { it.id == id }
    }
}

enum class Door(val id: String, val label: String) {
    FRONT_DOOR("front_door", "Front door"),
    BACK_DOOR("back_door", "Back door");

    companion object {
        fun fromId(id: String): Door? = entries.firstOrNull { it.id == id }
    }
}

enum class BlindPosition(val id: String, val label: String) {
    OPEN("open", "Open"), CLOSED("closed", "Closed"), HALF("half", "Half open");

    companion object {
        fun fromId(id: String): BlindPosition? = entries.firstOrNull { it.id == id }
    }
}

enum class Scene(val id: String, val label: String) {
    MORNING("morning", "Morning"), MOVIE("movie", "Movie"), NIGHT("night", "Night"), AWAY("away", "Away");

    companion object {
        fun fromId(id: String): Scene? = entries.firstOrNull { it.id == id }
    }
}

@Serializable
data class Light(val on: Boolean = false, val brightness: Int = 100)

@Serializable
data class Thermostat(val targetCelsius: Double = 20.0)

@Serializable
data class RoomState(
    val room: Room,
    val light: Light = Light(),
    val thermostat: Thermostat = Thermostat(),
    val blinds: BlindPosition = BlindPosition.OPEN,
)

/** A device the UI can highlight when a command changed it. */
sealed interface DeviceRef {
    data class LightOf(val room: Room) : DeviceRef
    data class ThermostatOf(val room: Room) : DeviceRef
    data class BlindsOf(val room: Room) : DeviceRef
    data class LockOf(val door: Door) : DeviceRef
}

@Serializable
data class HomeState(
    val rooms: Map<Room, RoomState>,
    /** `true` = locked. */
    val locks: Map<Door, Boolean>,
    val activeScene: Scene? = null,
) {
    fun room(room: Room): RoomState = rooms.getValue(room)

    companion object {
        fun initial(): HomeState = HomeState(
            rooms = Room.entries.associateWith { RoomState(it) },
            locks = Door.entries.associateWith { false },
        )
    }
}
