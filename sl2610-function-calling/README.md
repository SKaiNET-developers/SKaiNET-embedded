# sl2610-function-calling

Native **Kotlin Multiplatform** re-implementation of the Synaptics **FunctionGemma
function-calling demo**, built on **SKaiNET** + **SKaiNET-transformers**, targeting the
Synaptics **Astra Machina SL2610** (`rdk`, aarch64, Torq NPU).

Goal: the same on-device voice/text → tool-call pipeline — mic → VAD →
**Moonshine ASR (Torq NPU)** → **FunctionGemma LLM (CPU)** → action — as a single
**cross-compiled aarch64 binary**, at ≥ the Python app's speed with Kotlin/KMP comfort.
Heavy models go **DSL → StableHLO → IREE** (Torq for NPU, IREE `llvm-cpu` +NEON for CPU).
Finish plan (clone → run → finetune): **[`../../FINISH-PLAN.md`](../../FINISH-PLAN.md)**.

## Quick start (clone → run)

```bash
./bootstrap.sh                 # creates demo.env, locates the Torq toolchain, checks models
$EDITOR demo.env               # set BOARD=root@<ip> and GEMMA_GGUF=<your .gguf>
./gradlew :jvmRun              # host smoke test — no board needed

# board:
GEMMA_GGUF=... scripts/compile-gemma.sh board   # self-compile FunctionGemma (Python-free)
BOARD=root@<ip> ./gradlew deployBoard           # cross-compile aarch64, push, run on the SL2610
```

All machine/board-specific values live in **`demo.env`** (git-ignored; copy of `demo.env.example`).
Every `scripts/*.sh` sources it, and inline env still wins (`BOARD=root@10.0.0.5 ./gradlew deployBoard`).
No paths are hardcoded — the repo is meant to be cloned and run. **Teach it your own commands:**
[`docs/FINETUNING.md`](docs/FINETUNING.md). The one gated dependency is the private Torq compiler wheel —
`bootstrap.sh` tells you where to place it (see [`docs/TOOLCHAIN-PIN.md`](docs/TOOLCHAIN-PIN.md)).

## Based on the original Synaptics example

This is a Kotlin port of the upstream Python demo:

> **[synaptics-astra-demos/sl2610-examples → `Function_calling`](https://github.com/synaptics-astra-demos/sl2610-examples/tree/main/Function_calling)**

The original demonstrates fully on-device, cloud-free voice/text control of hardware on a
Synaptics Coralboard. It uses **FunctionGemma 270M** (fine-tuned for tool routing) for the
LLM and **Moonshine** for ASR on the Torq NPU, with a **Microphone → Silero VAD → Moonshine
ASR → FunctionGemma → dispatcher → hardware** pipeline. Moonshine is a **Whisper-family**
speech-to-text model — a Whisper-style **encoder-decoder** (a bidirectional audio encoder +
an autoregressive text decoder with cross-attention), re-optimised for fast on-device ASR. Its key trick is **functional
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
- **Phase 0 ✅** — KMP scaffold (`jvm` + `linuxArm64`), ActionRouter, dev loop (host `jvmRun` +
  cross-compiled aarch64 binary that deploys + runs on the board).
- **Phase 1 ✅** — LLM swap: consumes upstream `runtime-gemma-iree` (`GemmaDecoder`); demo-local
  runtime/codec duplication deleted; builds against **published** `sk.ainet.transformers:*` (0.34.x).
- **Weights: bf16 ✅** — a board A/B proved bf16 is a bit-exact drop-in for the f16 vmfb; the export
  bakes bf16.
- **Moonshine ASR — encoder DONE ✅** — the Moonshine **encoder** is authored in the SKaiNET NN DSL
  and **published** as `sk.ainet.transformers:skainet-transformers-inference-moonshine` (0.34.0;
  0.34.1 layer-qualifies its weights). It matches the reference (cos **1.0** f32 CPU, **~0.9998** bf16
  CPU), **compiles + runs on the SL2610 Torq NPU**, and transcribes correctly end-to-end
  (*"One, two, three."*). The former hardest blocker — the Torq `getWeightMemoryFormat` crash on
  attention — is solved.
- **Self-compile toolchain ✅** — the demo builds the encoder from the DSL itself, no vendor binary:
  host export (`./gradlew moonshineEncoderMlir`, real weights baked via `CHECKPOINT=`) →
  **dockerized stable v2.0.0 Torq compiler** pulled from `github.com/synaptics-torq/torq-compiler`
  (`scripts/iree-compile-torq-docker.sh`) + `gen_config` executor-map discovery
  (`scripts/gen-config-discover.sh`). Select our vmfb on the board with `MOONSHINE_ENCODER_VMFB`.
- **Interim ASR path** — `voicecc/asr/` still runs Moonshine **Python-free** on the **vendor prebuilt
  vmfbs** (`encoder/decoder/decoder_with_past.vmfb`) via `torq-run-module`, so the pipeline works today
  while the self-compiled path lands.
- **Next** — the Moonshine **decoder** in the DSL (still external: vendor vmfb / HF reference), then
  Kotlin VAD/mic (the last runtime Python), host-native `linuxX64`, and the parity/latency gate.
  Traceable plan: `../../PLAN.md`.

## Targets
- `jvm()` — fast host dev + A/B reference harness.
- `linuxArm64()` — the shipped board binary (cross-compiled from an **x64** host; Kotlin/Native
  cannot cross-compile from ARM).

## Dev loop
```bash
./bootstrap.sh                             # one-time: demo.env + toolchain + model checks
./gradlew :jvmRun                          # run on host
./gradlew :jvmTest                         # common + jvm tests
./gradlew :linkReleaseExecutableLinuxArm64 # cross-compile aarch64 binary
BOARD=root@<board-ip> ./gradlew deployBoard   # build + push + run on the board
# or directly:
BOARD=root@<board-ip> sh scripts/deploy.sh --run
```
Set `BOARD` in `demo.env` (or inline — it changes per boot). Deploy streams the binary over ssh
(`cat`; the BusyBox board has no rsync/sftp) and adds a `libcrypt.so.1` compat symlink
(Kotlin/Native links glibc `libcrypt.so.1`; the board ships libxcrypt `libcrypt.so.2`),
running with `LD_LIBRARY_PATH` so nothing in the system tree is touched.

## IntelliJ IDEA / Android Studio
Open the Gradle project (no Android target — `jvm` + `linuxArm64` only). The KMP plugin
surfaces `jvmRun` and `link…LinuxArm64`. Run configs live in `.run/`.

## Layout
```
bootstrap.sh              one-time setup (demo.env, Torq toolchain, model checks)
demo.env.example          the single config surface (BOARD, TORQ_PKG, GEMMA_GGUF, board dirs)
src/commonMain/voicecc/   actions/ (ActionRouter, the 6 tools) + App.kt
src/jvmMain/              host main + readSystemStatus actual + export/ (DSL→StableHLO:
                          HloBridge, MoonshineEncoderExport, MoonshineWeights)
src/linuxArm64Main/       board main + Pipeline (ASR→LLM→codec→action) + asr/ (Moonshine on the NPU)
scripts/                  bootstrap-sourced: deploy.sh · compile-gemma.sh · iree-compile-torq*.sh ·
                          moonshine_compile_preprocessor.sh · gen-config-discover.sh · .docker/
docs/                     FINETUNING.md (teach new commands) · TOOLCHAIN-PIN.md (toolchain currency) ·
                          BOARD-RUNBOOK.md (self-compiled-default + KV-cache board steps) ·
                          PERF-LOGBOOK.md · GEMMA-KV-BOARD-LOOP.md
```
See **[`../../FINISH-PLAN.md`](../../FINISH-PLAN.md)** for what remains to reach a fully self-compiled,
zero-Python, KV-fast, clone-to-run port.

## License
MIT — see [LICENSE](LICENSE) (consistent with SKaiNET). This demo is a complete
clean-room rewrite of Google's function-calling sample; no Apache-2.0-licensed
code is retained.
