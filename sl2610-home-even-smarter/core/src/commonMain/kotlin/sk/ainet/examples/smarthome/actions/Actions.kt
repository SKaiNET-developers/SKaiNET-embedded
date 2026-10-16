package sk.ainet.examples.smarthome.actions

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import sk.ainet.examples.smarthome.home.DeviceRef
import sk.ainet.examples.smarthome.home.HomeReducer
import sk.ainet.examples.smarthome.home.HomeState
import sk.ainet.examples.smarthome.tools.IntentMapper
import sk.ainet.examples.smarthome.tools.Mapping

/** A resolved tool call: the function the model chose and its string arguments. */
data class Intent(val tool: String, val args: Map<String, String> = emptyMap())

/** What a handler did with an intent. [changed] lets the UI highlight the devices touched. */
data class ActionResult(
    val tool: String,
    val ok: Boolean,
    val message: String,
    val changed: Set<DeviceRef> = emptySet(),
)

typealias Handler = (Intent) -> ActionResult

/**
 * Maps tool names to handlers and dispatches intents. The same shape as the router of the embedded
 * function-calling sample: the model never touches devices, a handler does, and an unknown tool is a
 * failed result rather than an exception.
 */
class ActionRouter {
    private val handlers = LinkedHashMap<String, Handler>()

    fun register(tool: String, handler: Handler): ActionRouter = apply { handlers[tool] = handler }
    fun registerAll(all: Map<String, Handler>): ActionRouter = apply { handlers += all }
    val tools: List<String> get() = handlers.keys.sorted()

    fun dispatch(intent: Intent): ActionResult {
        val handler = handlers[intent.tool] ?: return ActionResult(intent.tool, false, "unknown tool '${intent.tool}'")
        return try {
            handler(intent)
        } catch (e: Exception) {
            ActionResult(intent.tool, false, "handler error: ${e.message ?: e::class.simpleName}")
        }
    }

    fun dispatchAll(intents: List<Intent>): List<ActionResult> = intents.map(::dispatch)
}

/** Holds the home. The single writer is [HomeActions]; the UI observes [state]. */
class HomeStore(initial: HomeState = HomeState.initial()) {
    private val _state = MutableStateFlow(initial)
    val state: StateFlow<HomeState> = _state.asStateFlow()
    val current: HomeState get() = _state.value

    fun reset() { _state.value = HomeState.initial() }
    internal fun set(next: HomeState) { _state.update { next } }
}

/** The six home handlers: intent → [IntentMapper] → [HomeReducer] → [HomeStore]. */
class HomeActions(private val store: HomeStore) {

    fun handle(intent: Intent): ActionResult = when (val mapping = IntentMapper.map(intent.tool, intent.args)) {
        is Mapping.Unmapped -> ActionResult(intent.tool, false, mapping.reason)
        is Mapping.Command -> {
            val reduction = HomeReducer.apply(store.current, mapping.command)
            store.set(reduction.state)
            ActionResult(intent.tool, true, reduction.message, reduction.changed)
        }
    }

    fun handlers(): Map<String, Handler> =
        sk.ainet.examples.smarthome.tools.HomeTools.names.associateWith { { intent: Intent -> handle(intent) } }

    fun router(): ActionRouter = ActionRouter().registerAll(handlers())
}
