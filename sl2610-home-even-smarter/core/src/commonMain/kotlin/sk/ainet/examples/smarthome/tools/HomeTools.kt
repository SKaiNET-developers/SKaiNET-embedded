package sk.ainet.examples.smarthome.tools

/**
 * The function names and argument keys of the home tool catalog (`cartridges/catalogs/home-tools.v1.json`).
 * The JSON is the single source of truth — it is what the NLU cartridge is materialized with; a test asserts this
 * mirror matches it.
 */
object HomeTools {
    const val SET_LIGHT = "set_light"
    const val SET_THERMOSTAT = "set_thermostat"
    const val SET_BLINDS = "set_blinds"
    const val SET_LOCK = "set_lock"
    const val SET_SCENE = "set_scene"
    const val GET_STATUS = "get_status"

    val names: Set<String> = setOf(SET_LIGHT, SET_THERMOSTAT, SET_BLINDS, SET_LOCK, SET_SCENE, GET_STATUS)

    object Args {
        const val ROOM = "room"
        const val STATE = "state"
        const val BRIGHTNESS = "brightness"
        const val TEMPERATURE = "temperature"
        const val POSITION = "position"
        const val DOOR = "door"
        const val SCENE = "scene"
    }

    /** Argument keys per function, as the catalog declares them. */
    val arguments: Map<String, Set<String>> = mapOf(
        SET_LIGHT to setOf(Args.ROOM, Args.STATE, Args.BRIGHTNESS),
        SET_THERMOSTAT to setOf(Args.ROOM, Args.TEMPERATURE),
        SET_BLINDS to setOf(Args.ROOM, Args.POSITION),
        SET_LOCK to setOf(Args.DOOR, Args.STATE),
        SET_SCENE to setOf(Args.SCENE),
        GET_STATUS to setOf(Args.ROOM),
    )
}
