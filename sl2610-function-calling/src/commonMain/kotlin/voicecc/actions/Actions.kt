package voicecc.actions

/**
 * Pluggable intent -> action layer (Kotlin port of the Python `voicecc.actions`).
 *
 * The default handlers simply **log** each action (the SL2610 `rdk` board has no
 * Coral HAT). Register your own handler per tool to drive real devices.
 */

/** A parsed intent: the tool the model chose + the args it emitted. */
data class Intent(
    val tool: String,
    val args: Map<String, Any?> = emptyMap(),
)

/** Outcome of handling one intent. */
data class ActionResult(
    val tool: String,
    val ok: Boolean,
    val message: String,
    val detail: Map<String, Any?>? = null,
) {
    override fun toString(): String {
        val tag = if (ok) "ok" else "ERR"
        return "[$tag] $tool: $message"
    }
}

/** A handler turns an Intent into an ActionResult. */
typealias Handler = (Intent) -> ActionResult

/** Maps intents to registered handlers. The integration point. */
class ActionRouter(
    private val default: Handler = { intent ->
        ActionResult(intent.tool, false, "no handler registered (args=${intent.args})")
    },
) {
    private val handlers = mutableMapOf<String, Handler>()

    fun register(tool: String, handler: Handler) {
        handlers[tool] = handler
    }

    fun registerAll(map: Map<String, Handler>) {
        handlers.putAll(map)
    }

    val tools: List<String> get() = handlers.keys.sorted()

    fun dispatch(intent: Intent): ActionResult =
        try {
            (handlers[intent.tool] ?: default)(intent)
        } catch (e: Throwable) {
            ActionResult(intent.tool, false, "handler error: ${e.message}")
        }

    fun dispatchAll(intents: List<Intent>): List<ActionResult> = intents.map { dispatch(it) }
}

/**
 * Log-only implementations of the demo's six FunctionGemma tools. No hardware:
 * each handler formats a human-readable line and returns an [ActionResult].
 * Subclass / override any handler to make a tool do something real.
 */
open class LoggingActions {
    fun handlers(): Map<String, Handler> = mapOf(
        "set_lights" to ::setLights,
        "play_buzzer" to ::playBuzzer,
        "set_alarm" to ::setAlarm,
        "cancel_alarm" to ::cancelAlarm,
        "get_system_status" to ::getSystemStatus,
        "respond" to ::respond,
    )

    protected open fun emit(tool: String, message: String, detail: Map<String, Any?>? = null): ActionResult {
        println("ACTION $tool :: $message")
        return ActionResult(tool, true, message, detail)
    }

    open fun setLights(i: Intent): ActionResult {
        val parts = listOf("color", "effect", "state")
            .mapNotNull { k -> i.args[k]?.let { "$k=$it" } }
        return emit("set_lights", parts.joinToString(" ").ifEmpty { "(no args)" }, i.args)
    }

    open fun playBuzzer(i: Intent): ActionResult =
        emit("play_buzzer", "pattern=${i.args["pattern"] ?: "beep"}")

    open fun setAlarm(i: Intent): ActionResult {
        val whenAt = i.args["duration"] ?: i.args["time"] ?: "?"
        val label = i.args["label"] ?: ""
        return emit("set_alarm", "when=$whenAt label=$label".trim())
    }

    open fun cancelAlarm(i: Intent): ActionResult =
        emit("cancel_alarm", "label=${i.args["label"] ?: "all"}")

    open fun getSystemStatus(i: Intent): ActionResult {
        val stats = readSystemStatus()
        val metric = i.args["metric"] as? String
        val shown = if (metric != null && metric != "all" && stats.containsKey(metric))
            mapOf(metric to stats[metric]) else stats
        return emit("get_system_status", shown.entries.joinToString(" ") { "${it.key}=${it.value}" }, shown)
    }

    open fun respond(i: Intent): ActionResult =
        emit("respond", (i.args["message"] as? String) ?: "")
}

/** Best-effort system status; platform-specific real values via expect/actual. */
expect fun readSystemStatus(): Map<String, Any?>

/** An [ActionRouter] wired to the log-only baseline handlers. */
fun defaultRouter(): ActionRouter = ActionRouter().apply { registerAll(LoggingActions().handlers()) }
