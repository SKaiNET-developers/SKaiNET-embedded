# sl2610-voice-cc-kt

Native **Kotlin Multiplatform** re-implementation of the [`sl2610-voice-cc`](../sl2610-voice-cc)
Python voice command-and-control app, built on **SKaiNET** + **SKaiNET-transformers**,
targeting the Synaptics **Astra Machina SL2610** (`rdk`, aarch64, Torq NPU).

Goal: the same pipeline — mic → VAD → **Moonshine ASR (Torq NPU)** → **Gemma LLM (CPU)** →
action — as a single **cross-compiled aarch64 binary**, at ≥ the Python app's speed with
Kotlin/KMP comfort. Heavy models go **DSL → StableHLO → IREE** (Torq for NPU, IREE `llvm-cpu`
+NEON for CPU). See the full plan in `../../.claude/plans/`.

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
src/commonMain/voicecc/   actions/ (ActionRouter, ported) + App.kt
src/jvmMain/              host main + readSystemStatus actual
src/linuxArm64Main/       board main + readSystemStatus actual
scripts/deploy.sh         host→board deploy (+ libcrypt compat)
```

## License
Apache-2.0 (consistent with SKaiNET).
