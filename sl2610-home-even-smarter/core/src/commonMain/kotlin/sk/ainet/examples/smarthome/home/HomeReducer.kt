package sk.ainet.examples.smarthome.home

/** The result of applying one command: the new state, what changed (for the UI to flash) and a human line. */
data class Reduction(val state: HomeState, val changed: Set<DeviceRef>, val message: String)

/** Pure state transitions of the demo home. No I/O, no threads: trivially testable and the same on every platform. */
object HomeReducer {

    fun apply(state: HomeState, command: HomeCommand): Reduction = when (command) {
        is HomeCommand.SetLight -> setLight(state, command.room, command.on, command.brightness)
        is HomeCommand.SetThermostat -> setThermostat(state, command.room, command.targetCelsius)
        is HomeCommand.SetBlinds -> setBlinds(state, command.room, command.position)
        is HomeCommand.SetLock -> setLock(state, command.door, command.locked)
        is HomeCommand.SetScene -> setScene(state, command.scene)
        is HomeCommand.GetStatus -> Reduction(state, emptySet(), describe(state, command.room))
    }

    private fun setLight(state: HomeState, room: Room, on: Boolean, brightness: Int?): Reduction {
        val current = state.room(room)
        val light = Light(on = on, brightness = (brightness ?: current.light.brightness).coerceIn(0, 100))
        val message = buildString {
            append("${room.label} light ${if (on) "on" else "off"}")
            if (on && brightness != null) append(" at ${light.brightness}%")
        }
        return Reduction(state.update(room) { it.copy(light = light) }.copy(activeScene = null), setOf(DeviceRef.LightOf(room)), message)
    }

    private fun setThermostat(state: HomeState, room: Room?, targetCelsius: Double): Reduction {
        val target = targetCelsius.coerceIn(5.0, 30.0)
        val rooms = room?.let { listOf(it) } ?: Room.entries
        var next = state
        for (r in rooms) next = next.update(r) { it.copy(thermostat = Thermostat(target)) }
        val where = room?.label ?: "Whole home"
        return Reduction(next.copy(activeScene = null), rooms.map { DeviceRef.ThermostatOf(it) }.toSet(), "$where set to ${formatTemperature(target)}")
    }

    private fun setBlinds(state: HomeState, room: Room, position: BlindPosition): Reduction =
        Reduction(state.update(room) { it.copy(blinds = position) }.copy(activeScene = null), setOf(DeviceRef.BlindsOf(room)), "${room.label} blinds ${position.label.lowercase()}")

    private fun setLock(state: HomeState, door: Door, locked: Boolean): Reduction =
        Reduction(state.copy(locks = state.locks + (door to locked), activeScene = null), setOf(DeviceRef.LockOf(door)), "${door.label} ${if (locked) "locked" else "unlocked"}")

    private fun setScene(state: HomeState, scene: Scene): Reduction {
        var next = state
        val changed = mutableSetOf<DeviceRef>()
        fun everyRoom(block: (RoomState) -> RoomState) {
            for (r in Room.entries) { next = next.update(r, block); changed += setOf(DeviceRef.LightOf(r), DeviceRef.ThermostatOf(r), DeviceRef.BlindsOf(r)) }
        }
        fun lockAll() { next = next.copy(locks = Door.entries.associateWith { true }); changed += Door.entries.map { DeviceRef.LockOf(it) } }
        when (scene) {
            Scene.MORNING -> everyRoom { it.copy(light = Light(true, 80), blinds = BlindPosition.OPEN, thermostat = Thermostat(21.0)) }
            Scene.MOVIE -> {
                everyRoom { it.copy(light = Light(false, it.light.brightness), blinds = BlindPosition.CLOSED) }
                next = next.update(Room.LIVING_ROOM) { it.copy(light = Light(true, 20)) }
            }
            Scene.NIGHT -> { everyRoom { it.copy(light = Light(false, it.light.brightness), blinds = BlindPosition.CLOSED, thermostat = Thermostat(18.0)) }; lockAll() }
            Scene.AWAY -> { everyRoom { it.copy(light = Light(false, it.light.brightness), thermostat = Thermostat(17.0)) }; lockAll() }
        }
        return Reduction(next.copy(activeScene = scene), changed, "${scene.label} scene on")
    }

    fun describe(state: HomeState, room: Room?): String {
        fun line(r: RoomState): String = buildString {
            append(r.room.label); append(": light ")
            append(if (r.light.on) "on ${r.light.brightness}%" else "off")
            append(", ${formatTemperature(r.thermostat.targetCelsius)}, blinds ${r.blinds.label.lowercase()}")
        }
        if (room != null) return line(state.room(room))
        val locks = Door.entries.joinToString(", ") { "${it.label.lowercase()} ${if (state.locks[it] == true) "locked" else "unlocked"}" }
        return Room.entries.joinToString("; ") { line(state.room(it)) } + "; $locks"
    }

    private fun HomeState.update(room: Room, block: (RoomState) -> RoomState): HomeState =
        copy(rooms = rooms + (room to block(room(room))))

    fun formatTemperature(celsius: Double): String {
        val rounded = (celsius * 2).toInt() / 2.0
        return if (rounded == rounded.toInt().toDouble()) "${rounded.toInt()} °C" else "$rounded °C"
    }
}
