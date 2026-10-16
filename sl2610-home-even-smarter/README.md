# sl2610-home-even-smarter

HomeEvenSmarter on the Synaptics SL26xx edge board: a **hybrid-AI smart-home voice app** that runs as a native
terminal UI on the board.

```
USB mic → Moonshine v2 streaming ASR (Torq NPU) → FunctionGemma-270M (CPU / GPU) → home action
                                                      └─ out of catalog → escalation to the middleware on another machine
```

## What it shows

1. **Hybrid AI.** The board resolves the common home commands locally, fast and offline. Anything outside the tool
   catalog is escalated, never mapped to a wrong action. The TUI shows every stage with its timing:
   listening → ASR → NLU → action → escalation.
2. **NPU cartridges from blueprints.** Moonshine for the Torq NPU is built from a public blueprint with the pinned
   Torq toolchain into a signed cartridge. The board downloads it from `:server`, verifies it and runs it, the same
   flow as the GPU/CPU cartridges of the phone app.
3. **One Kotlin codebase.** The same pipeline and domain code as the Android HomeEvenSmarter app, compiled with
   Kotlin/Native. No Python and no JVM on the board.
4. **One native runtime.** A single in-process IREE runtime (Torq fork, built from source) drives the CPU and the
   NPU. See [`native/iree-torq-runtime`](native/iree-torq-runtime).
5. **Measured.** A 12-utterance golden set (WAV replay and live mic) with per-stage latency and memory.

## Status

| Part | State |
|---|---|
| TUI app (`:tui`) on fake engines, host + board | works |
| Native IREE runtime, CPU + NPU in one process (C smoke test) | works; Kotlin/Native binding next |
| Real engines (Moonshine, FunctionGemma) | next: cinterop binding + Linux engine factory |
| Torq target for the Moonshine blueprint | blocked on NPU multi-dispatch zeros; see the board update |
| Middleware client | pending protocol |
| Board / Torq update 2.0 → 2.2 | procedure ready: [docs/BOARD-UPDATE.md](docs/BOARD-UPDATE.md) |

## Modules

| Module | Targets | Content |
|---|---|---|
| `:core` | jvm, linuxX64, linuxArm64 | home model, tool catalog mapping, `VoicePipeline`, `Endpointer`, host rules, escalation seam, engine contracts, fakes |
| `:cartridges` | jvm, linuxX64, linuxArm64 | cartridge store, resumable downloader with sha256 checks, engine factory |
| `:presentation` | jvm, linuxX64, linuxArm64 | `AppViewModel`: engines, pipeline, golden set, cartridge flows, log |
| `:tui` | linuxX64, linuxArm64 | the app: Mosaic terminal UI, `arecord` microphone, WAV replay |
| `:server` | jvm | cartridge server the board downloads from |

## Build and run

```
./gradlew :core:allTests :cartridges:allTests :server:test           # tests (JVM + linuxX64)
./gradlew :tui:linkDebugExecutableLinuxX64                          # run on the host:
tui/build/bin/linuxX64/debugExecutable/hes-tui.kexe --wav some-16k-mono.wav
BOARD=root@<board-ip> scripts/deploy-tui.sh                          # cross-build + install on the board
ssh -t root@<board-ip> /home/root/hes-tui/hes-tui --mic plughw:1,0   # USB mic on card 1
```

Keys: `space` talk (press again or pause to stop; terminals report no key release), `t` type a command, `g` golden
set, `r` reset the home, `f` fakes on/off, `tab` cartridges (`i` fetch index, `d` download, `v` verify,
`l` load and warm up), `q` quit.

Cartridges: `scripts/serve-cartridges.sh <dir>` on the host, then `hes-tui --server http://<host-ip>:8080`.

## Board

- [`board/board.lock`](board/board.lock): the board image ↔ Torq toolchain ↔ runtime set the app is verified against.
- `scripts/board-audit.sh`: read-only drift check against the lock.
- `scripts/board-canaries.sh`: NPU canary gate for a (Torq compiler, board runtime) pair.
- `scripts/board-backup.sh`: save what a reflash would destroy.
- `scripts/board-update-download.sh`: images, Torq release files and the flashing tool, sha256-verified.
- `board/docker/Dockerfile.torq-compiler`: the Torq compiler image for a given version.
- [docs/BOARD-UPDATE.md](docs/BOARD-UPDATE.md): step-by-step board image / Torq tools / NPU update.

The board userland is BusyBox: there is no rsync or sftp, so files are streamed with `ssh cat`. Kotlin/Native
binaries link `libcrypt.so.1` while the board ships `.so.2`. `deploy-tui.sh` handles both.
