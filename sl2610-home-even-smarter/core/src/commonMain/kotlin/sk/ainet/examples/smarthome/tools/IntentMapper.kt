package sk.ainet.examples.smarthome.tools

import sk.ainet.examples.smarthome.home.BlindPosition
import sk.ainet.examples.smarthome.home.Door
import sk.ainet.examples.smarthome.home.HomeCommand
import sk.ainet.examples.smarthome.home.Room
import sk.ainet.examples.smarthome.home.Scene

/** A tool call turned into a home command, or the reason it could not be. */
sealed interface Mapping {
    data class Command(val command: HomeCommand) : Mapping
    data class Unmapped(val reason: String) : Mapping
}

/**
 * Host-side canonicalisation of what the NLU emits: the model returns a function name and *string* arguments;
 * the host owns the entities ("lounge" is the living room, "twenty one" is 21). Anything it cannot place becomes
 * [Mapping.Unmapped] — the pipeline then escalates instead of guessing.
 */
object IntentMapper {

    fun map(name: String, args: Map<String, String>): Mapping {
        val a = args.mapKeys { it.key.trim().lowercase() }.mapValues { it.value.trim() }
        return when (name) {
            HomeTools.SET_LIGHT -> {
                val room = room(a[HomeTools.Args.ROOM]) ?: return unmapped("no room in $a")
                val on = onOff(a[HomeTools.Args.STATE]) ?: return unmapped("no on/off state in $a")
                val brightness = a[HomeTools.Args.BRIGHTNESS]?.let { number(it)?.toInt()?.coerceIn(0, 100) }
                Mapping.Command(HomeCommand.SetLight(room, on, brightness))
            }
            HomeTools.SET_THERMOSTAT -> {
                val temperature = a[HomeTools.Args.TEMPERATURE]?.let { number(it) } ?: return unmapped("no temperature in $a")
                Mapping.Command(HomeCommand.SetThermostat(room(a[HomeTools.Args.ROOM]), temperature))
            }
            HomeTools.SET_BLINDS -> {
                val room = room(a[HomeTools.Args.ROOM]) ?: return unmapped("no room in $a")
                val position = position(a[HomeTools.Args.POSITION]) ?: return unmapped("no blind position in $a")
                Mapping.Command(HomeCommand.SetBlinds(room, position))
            }
            HomeTools.SET_LOCK -> {
                val door = door(a[HomeTools.Args.DOOR]) ?: Door.FRONT_DOOR
                val locked = locked(a[HomeTools.Args.STATE]) ?: return unmapped("no lock state in $a")
                Mapping.Command(HomeCommand.SetLock(door, locked))
            }
            HomeTools.SET_SCENE -> {
                val scene = scene(a[HomeTools.Args.SCENE]) ?: return unmapped("no scene in $a")
                Mapping.Command(HomeCommand.SetScene(scene))
            }
            HomeTools.GET_STATUS -> Mapping.Command(HomeCommand.GetStatus(room(a[HomeTools.Args.ROOM])))
            else -> unmapped("unknown function '$name'")
        }
    }

    private fun unmapped(reason: String) = Mapping.Unmapped(reason)

    private fun norm(s: String?): String? = s?.lowercase()?.replace(Regex("[^a-z0-9.]+"), " ")?.trim()?.takeIf { it.isNotEmpty() }

    fun room(raw: String?): Room? {
        val s = norm(raw) ?: return null
        Room.fromId(s.replace(' ', '_'))?.let { return it }
        return when {
            "living" in s || "lounge" in s || "sitting" in s || "family room" in s -> Room.LIVING_ROOM
            "kitchen" in s -> Room.KITCHEN
            "bed" in s -> Room.BEDROOM
            "bath" in s || "toilet" in s -> Room.BATHROOM
            "hall" in s || "corridor" in s || "entrance" in s -> Room.HALLWAY
            else -> null
        }
    }

    fun door(raw: String?): Door? {
        val s = norm(raw) ?: return null
        Door.fromId(s.replace(' ', '_'))?.let { return it }
        return when {
            "back" in s || "rear" in s || "garden" in s || "patio" in s -> Door.BACK_DOOR
            "front" in s || "main" in s || "entrance" in s || s == "door" -> Door.FRONT_DOOR
            else -> null
        }
    }

    fun onOff(raw: String?): Boolean? = when (norm(raw)) {
        "on", "true", "1", "enable", "enabled", "yes" -> true
        "off", "false", "0", "disable", "disabled", "no" -> false
        else -> null
    }

    fun locked(raw: String?): Boolean? = when (norm(raw)) {
        "locked", "lock", "close", "closed", "secure", "secured", "true" -> true
        "unlocked", "unlock", "open", "opened", "false" -> false
        else -> null
    }

    fun position(raw: String?): BlindPosition? {
        val s = norm(raw) ?: return null
        BlindPosition.fromId(s)?.let { return it }
        return when {
            "half" in s || "middle" in s || "50" in s -> BlindPosition.HALF
            "open" in s || "up" in s || "raise" in s -> BlindPosition.OPEN
            "clos" in s || "down" in s || "shut" in s || "lower" in s -> BlindPosition.CLOSED
            else -> null
        }
    }

    fun scene(raw: String?): Scene? {
        val s = norm(raw) ?: return null
        Scene.fromId(s)?.let { return it }
        return Scene.entries.firstOrNull { it.id in s }
    }

    private val units = mapOf(
        "zero" to 0, "one" to 1, "two" to 2, "three" to 3, "four" to 4, "five" to 5, "six" to 6, "seven" to 7, "eight" to 8,
        "nine" to 9, "ten" to 10, "eleven" to 11, "twelve" to 12, "thirteen" to 13, "fourteen" to 14, "fifteen" to 15,
        "sixteen" to 16, "seventeen" to 17, "eighteen" to 18, "nineteen" to 19,
    )
    private val tens = mapOf("twenty" to 20, "thirty" to 30, "forty" to 40, "fifty" to 50, "sixty" to 60, "seventy" to 70, "eighty" to 80, "ninety" to 90)

    /** "21", "21.5", "21 degrees", "twenty one", "twenty-one and a half", "a hundred" → a number, else null. */
    fun number(raw: String): Double? {
        val s = norm(raw) ?: return null
        Regex("""-?\d+(\.\d+)?""").find(s)?.let { return it.value.toDouble() }
        val words = s.split(' ').filter { it.isNotEmpty() && it != "and" && it != "a" && it != "degrees" && it != "percent" && it != "celsius" }
        var total = 0.0; var seen = false
        for (w in words) {
            when {
                w == "hundred" -> { total = if (total == 0.0) 100.0 else total * 100; seen = true }
                w == "half" -> { total += 0.5; seen = true }
                w in units -> { total += units.getValue(w); seen = true }
                w in tens -> { total += tens.getValue(w); seen = true }
                else -> return null
            }
        }
        return if (seen) total else null
    }
}
