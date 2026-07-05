# Make it your own — using this demo as a template

This app is deliberately layered so you can keep the skeleton (KMP scaffold, the
speech → LLM → action pipeline, the host↔board dev loop) and replace just the parts
that are specific to *this* board and *this* toolset. There are three seams. Most
projects only touch the first.

The dependency layering the template assumes:

```
your app  (composition root: pick tools, models, board)
 ├── shared voice→LLM→action pipeline        [reused as-is]
 ├── SKaiNET core            (Maven Central)  [HW-agnostic]
 └── a vendor plugin  (e.g. synaptics-torq)   [only this names your NPU]
```

---

## Seam 1 — Your tools (the common case)

The "function calling" runtime is just a router from a tool name to a Kotlin lambda.
The model emits `Intent(tool, args)`; you decide what that does. Nothing here is
board- or model-specific, so this is pure `commonMain`.

```kotlin
// Register real handlers instead of the log-only defaults:
val router = ActionRouter().apply {
    register("set_lights") { intent ->
        val color = intent.args["color"] as? String ?: "white"
        gpio.setRgb(color)                       // <- your hardware
        ActionResult("set_lights", ok = true, message = "lights $color")
    }
    register("respond") { intent ->
        ActionResult("respond", ok = true, message = intent.args["message"].toString())
    }
    // ... your own tools
}
```

To change *which* tools exist, edit the map in `LoggingActions.handlers()` (or stop
subclassing it and build your own set). To make a tool do something real without
changing its name, override that one handler — the default `LoggingActions` is `open`
for exactly this. The tool names and arg schema must match what the LLM was trained /
prompted to emit; everything else is yours.

Platform-specific reads (sensors, board status) go through `expect/actual`:

```kotlin
expect fun readSystemStatus(): Map<String, Any?>   // commonMain
// jvmMain actual: fake/host values;  linuxArm64Main actual: real board sensors
```

---

## Seam 2 — Your models

Two models plug into the pipeline independently (`Pipeline.kt`):

- **ASR** (`MoonshineRunner().transcribe(wav)`) — swap for any speech model you can
  compile to the board runtime, or a different one entirely. Returns text.
- **Function-calling LLM** (`GemmaDecoder(...).generate(text)`) — returns parsed tool
  calls. Swap the checkpoint, or the decoder, as long as it yields `{tool, args}`.

```kotlin
val text    = MyAsr().transcribe(wav)            // seam 2a
val calls   = MyDecoder(...).generate(text).calls // seam 2b
val intents = calls.map { Intent(it.tool, it.args) }
router.dispatchAll(intents)                       // seam 1
```

The models are authored in the SKaiNET NN DSL and compiled DSL → StableHLO → IREE, so
"swap the model" means author/point-to a different DSL model and rebuild its `.vmfb` —
not rewrite the pipeline.

---

## Seam 3 — Your board / NPU

The pipeline is target-agnostic; only the *compile step* and one plugin know your NPU.
This is the pattern from `skainet-embedded-vendors`: the model emits portable
StableHLO, and a **vendor plugin** contributes the target's optimizations + compile
flags, registered once at your composition root.

```kotlin
import sk.ainet.vendors.torq.TorqPlugin

TorqPlugin.install()                              // registers the "torq" target passes
val tiled = dagPipelineFor(TorqPlugin.TARGET).optimize(graph).graph
val mlir  = toStableHlo(tiled, "encoder").content
// compile with TorqPlugin.compileFlags for the SL2610 NPU
```

For a different accelerator, write (or depend on) a plugin for *its* target string and
swap `TorqPlugin` for it. The shared model and app code do not change — they never name
a backend. Add a `linuxX64()` target the same way you'd add any KMP target if you want
a host-native build alongside the board one.

---

## Keeping the host dev loop

Whatever you swap, keep the `jvm()` target as your reference harness: it runs the same
`commonMain` logic on your laptop with no hardware, so you can iterate on tools and
intent handling in seconds and only deploy to the board (`scripts/deploy.sh`) once the
logic is right. That host↔board parity is the point of the KMP structure — don't lose
it when you fork.
