# Wiring `open_door` (`<tool_6>`) into the app

Two edits, applied after the model is finetuned to emit `<tool_6>`.

## 1. Codec: map the token to a name

`CompactCodec.TOKEN_TO_NAME` lives in the upstream `gemma-iree` module. Add the row:

```kotlin
private val TOKEN_TO_NAME: Map<String, String?> = mapOf(
    "0" to "set_lights", "1" to "play_buzzer", "2" to "set_alarm",
    "3" to "cancel_alarm", "4" to "get_system_status", "5" to "respond",
    "6" to "open_door",          // <-- new
    "none" to null,
)
```

Until that map is injectable upstream (tracked in `BOARD-RUNBOOK.md`), either patch the dependency locally
or map the raw token in the demo. The demo already converts `ToolCall.tool` (a name) → `Intent`, so once the
codec yields `"open_door"` no other pipeline change is needed.

## 2. Router: register a handler

`src/commonMain/kotlin/voicecc/actions/Actions.kt` — extend `defaultRouter()` (leave `LoggingActions`
untouched so the base six keep working):

```kotlin
fun defaultRouter(): ActionRouter = ActionRouter().apply {
    registerAll(LoggingActions().handlers())
    register("open_door") { i ->
        val which = i.args["which"] ?: "front"
        // TODO drive the real actuator here; log-only by default:
        println("ACTION open_door :: which=$which")
        ActionResult("open_door", true, "opening $which door")
    }
}
```

## 3. Test

Add to `src/commonTest/kotlin/voicecc/ActionRouterTest.kt`:

```kotlin
@Test fun openDoorRoutes() {
    val r = defaultRouter().dispatch(Intent("open_door", mapOf("which" to "garage")))
    assertTrue(r.ok); assertEquals("open_door", r.tool)
}
```
