# A voice-controlled function caller on an NPU dev board — in Kotlin, end to end

You say *"turn the lights red and set an alarm for ten minutes."* A small board on
your desk — no cloud, no network — hears you, transcribes the words, decides which
functions to call with which arguments, and calls them. Speech recognition and a
small language model, both running on-device, on a low-power NPU.

That part is now ordinary. Here's the part that isn't: **every layer is Kotlin.**

## Why that's an odd sentence

Edge AI is a C++ and Python world. Models are authored in PyTorch, converted through
a chain of tools, and driven by a runtime you talk to from C. Kotlin shows up, if at
all, in the app on top.

This project pushes Kotlin all the way down:

- The whole thing compiles with **Kotlin/Native** to a *single* aarch64 binary that
  runs on the board. No interpreter shipped, no venv to reproduce.
- It's **Kotlin Multiplatform**: the exact same shared code has a `jvm()` target you
  run and debug on your laptop as a reference, and a `linuxArm64` target that deploys
  to the board. Same logic, two worlds — one for the fast dev loop, one for the metal.
- The speech model is authored in a **Kotlin DSL**, then compiled — Kotlin objects to
  StableHLO to an IREE module — and run on the NPU. The model definition reads like
  the network diagram, and it's the same language as the app calling it.
- Talking to the on-device runtime is plain **cinterop**: Kotlin calling the board's
  inference C API directly, no bridge process in between.

The pipeline on the board is four honest steps:

```
mic → speech-to-text (on the NPU) → a function-calling LLM → an action router → your handler
```

The language model doesn't chat. It reads *"turn the lights red"* and emits
`set_lights(color=red, state=on)` — a structured tool call. A tiny router maps that to
a Kotlin function. The six default tools just log; you register your own to move real
pins, LEDs, buzzers, whatever your board has.

## The parts a Kotlin developer will recognize

Nothing exotic in the app itself — it's the KMP you already know, pointed at hardware:

- `expect fun readSystemStatus()` with a JVM actual and a board actual — the same
  pattern you'd use for anything platform-specific, here reading real board sensors.
- An `ActionRouter` that's a `Map<String, (Intent) -> ActionResult>`. That's the whole
  "function calling" runtime. It fits on a napkin, and it's the only thing you have to
  touch to make the demo do something on *your* device.
- A `linuxArm64` `main` that's a few `cinterop` calls and a decode loop.

You can clone it, run the `jvm` target on your laptop, and watch the intents dispatch
before any hardware is involved.

## What's still rough (because pretending otherwise helps no one)

- Mic capture + voice-activity detection is currently a small **Python helper** shelled
  out from the board — the one bit of Python left in the loop, and it's on the way out
  (replaced by Kotlin capture).
- The board is memory-tight, so the heavy models run one at a time, sequentially.
- Getting a transformer to compile cleanly for this NPU took real fights with the
  compiler (attention layouts, low-precision LayerNorm, rotary embeddings). Those are
  written up honestly in the sibling docs if you like that kind of thing.

## Have a look

It's **MIT licensed** and self-contained (its own Gradle wrapper; `./gradlew jvmRun`
works with nothing attached). Start in `src/commonMain/kotlin/voicecc/actions/` — the
router is ~40 lines — then follow `Pipeline.kt` on the board target to see the four
steps wired together.

If you've ever wanted your Kotlin to end up on something smaller than a phone, doing
something more interesting than blinking an LED, this is a working example of exactly
that. The neural-net stack underneath is [SKaiNET][skainet], but the story here isn't
the library — it's that the whole edge AI loop fit in one language, and that language
was Kotlin.

[skainet]: https://github.com/SKaiNET-developers/SKaiNET
