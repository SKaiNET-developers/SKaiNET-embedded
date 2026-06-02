# sl2610-function-calling

Native **Kotlin Multiplatform** re-implementation of the Synaptics **FunctionGemma
function-calling demo**, built on **SKaiNET** + **SKaiNET-transformers**, targeting the
Synaptics **Astra Machina SL2610** (`rdk`, aarch64, Torq NPU).

Goal: the same on-device voice/text → tool-call pipeline — mic → VAD →
**Moonshine ASR (Torq NPU)** → **FunctionGemma LLM (CPU)** → action — as a single
**cross-compiled aarch64 binary**, at ≥ the Python app's speed with Kotlin/KMP comfort.
Heavy models go **DSL → StableHLO → IREE** (Torq for NPU, IREE `llvm-cpu` +NEON for CPU).
See the full plan in `../../.claude/plans/`.

## Based on the original Synaptics example

This is a Kotlin port of the upstream Python demo:

> **[synaptics-astra-demos/sl2610-examples → `Function_calling`](https://github.com/synaptics-astra-demos/sl2610-examples/tree/main/Function_calling)**

The original demonstrates fully on-device, cloud-free voice/text control of hardware on a
Synaptics Coralboard. It uses **FunctionGemma 270M** (fine-tuned for tool routing) for the
LLM and **Moonshine** for ASR on the Torq NPU, with a **Microphone → Silero VAD → Moonshine
ASR → FunctionGemma → dispatcher → hardware** pipeline. Its key trick is **functional
tokens**: each tool is a single special token, so no JSON schema is injected into the prompt,
which keeps inference fast (~1.3–2.0 s/turn). It exposes six tools — `set_lights`,
`play_buzzer`, `set_alarm`, `cancel_alarm`, `get_system_status`, `respond` — and ships as
Python (PyQt6 UI + CLI REPL) driving RGB LEDs, a piezo buzzer, and an optional Neopixel ring.

**What this port keeps:** the same six tools, the same compact tool-call codec
(`<tool_N>(args)<end>` Octopus-v2 named-arg style), and the same model lineup.
**What changes:** the whole pipeline is rewritten in Kotlin/Native and runs the heavy models
through the SKaiNET **DSL → StableHLO → IREE** path instead of the vendor Python runtime —
shipping as one cross-compiled aarch64 binary rather than a Python venv. The default action
handlers are log-only (no Coral HAT assumed); register your own to drive real hardware.

## Status
- **Phase 0 ✅** — KMP scaffold (`jvm` + `linuxArm64`), ActionRouter ported, dev loop proven:
  host `jvmRun` and a cross-compiled aarch64 binary that **deploys + runs on the board**.
- Phase 1 (next) — LLM swap: FunctionGemma → SKaiNET-transformers Kotlin Gemma.
- Phases 2–4 — StableHLO→IREE bridge, Moonshine-on-NPU from Kotlin, full pipeline + parity/latency gate.

## Targets
- `jvm()` — fast host dev + A/B reference harness.
- `linuxArm64()` — the shipped board binary (cross-compiled from an **x64** host; Kotlin/Native
  cannot cross-compile from ARM).

## Dev loop
```bash
./gradlew :jvmRun                          # run on host
./gradlew :jvmTest                         # common + jvm tests
./gradlew :linkReleaseExecutableLinuxArm64 # cross-compile aarch64 binary
BOARD=root@<board-ip> ./gradlew deployBoard   # build + push + run on the board
# or directly:
BOARD=root@<board-ip> sh scripts/deploy.sh --run
```
The board IP changes per boot — set `BOARD` accordingly. Deploy streams the binary over ssh
(`cat`; the BusyBox board has no rsync/sftp) and adds a `libcrypt.so.1` compat symlink
(Kotlin/Native links glibc `libcrypt.so.1`; the board ships libxcrypt `libcrypt.so.2`),
running with `LD_LIBRARY_PATH` so nothing in the system tree is touched.

## IntelliJ IDEA / Android Studio
Open the Gradle project (no Android target — `jvm` + `linuxArm64` only). The KMP plugin
surfaces `jvmRun` and `link…LinuxArm64`. Run configs live in `.run/`.

## Layout
```
src/commonMain/voicecc/   actions/ (ActionRouter, the 6 tools) + llm/ (CompactCodec) + App.kt
src/jvmMain/              host main + readSystemStatus actual + export/ (DAG→StableHLO bridge)
src/linuxArm64Main/       board main + Pipeline (ASR→LLM→codec→action) + runtime/ (IREE)
scripts/deploy.sh         host→board deploy (+ libcrypt compat)
```

## License
Apache-2.0 (consistent with SKaiNET).
