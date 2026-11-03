package sk.ainet.examples.smarthome.home

/** What the home can be asked to do. Produced by the intent mapper, consumed by the reducer. */
sealed interface HomeCommand {
    data class SetLight(val room: Room, val on: Boolean, val brightness: Int? = null) : HomeCommand
    /** `room == null` means every room. */
    data class SetThermostat(val room: Room?, val targetCelsius: Double) : HomeCommand
    data class SetBlinds(val room: Room, val position: BlindPosition) : HomeCommand
    data class SetLock(val door: Door, val locked: Boolean) : HomeCommand
    data class SetScene(val scene: Scene) : HomeCommand
    /** `room == null` means the whole home. Changes nothing; the reducer answers with a message. */
    data class GetStatus(val room: Room?) : HomeCommand
}
