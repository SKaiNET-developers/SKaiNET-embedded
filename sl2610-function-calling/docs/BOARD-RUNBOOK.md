# Board runbook — the steps that need the SL2610 (and the final packaging)

Everything in `PLAN.md` (Part A) that can't run off-board, plus the last packaging move. Each step is
independently runnable and has an explicit "done when." Ordered by the plan's sequencing.

Prereqs: `./bootstrap.sh` done, `demo.env` set (BOARD, TORQ_PKG, GEMMA_GGUF), board reachable
(`ssh $BOARD true`). Toolchain pin rationale + canary gate: `TOOLCHAIN-PIN.md`.

---

## P0 — freeze the working baseline (no board)

Commit the in-flight work so the pre-restructure state is reproducible. Left to you (I don't commit
without your ask). The dirty trees at 2026-07-11:
```bash
# SKaiNET-transformers (feature branch): GemmaKvDecoder.kt, MoonshineKvDecoder.kt, FunctionGemma KV/int8 tests
# SKaiNET-embedded (functiongemma-selfcompile): compile-gemma.sh, Pipeline.kt, MoonshineRunner.kt,
#   PERF-LOGBOOK.md, GEMMA-KV-BOARD-LOOP.md, MoonshineKvDecoder.kt, weights/
git -C <repo> add -A && git -C <repo> commit -m "wip: KV-cache decoders + perf logbook (pre-finish baseline)"
```
**Done when:** both repos have clean trees on their feature branches.

---

## P3 — make the board path fully self-compiled + zero Python

### P3.1 Self-compiled encoder + decoder as the default
Today `MoonshineRunner`/`MoonshineDecoder` default to the vendor prebuilt vmfbs (`modelDir`), with OUR
vmfbs opt-in via `MOONSHINE_ENCODER_VMFB` / `MOONSHINE_DECODER_VMFB`. Flip the default so ours ship and the
vendor path is the fallback.
```bash
# build ours (see TOOLCHAIN-PIN.md re: the encoder fusion/40-dispatch caveat — this is the open NPU item):
./gradlew moonshineEncoderMlir && scripts/iree-compile-torq.sh build/mlir/moonshine-encoder.mlir build/mlir/enc.vmfb
scripts/compile-moonshine-decoder.sh          # our re-decode decoder (CPU) — see the script
```
Then in `src/linuxArm64Main/kotlin/voicecc/asr/MoonshineDecoder.kt`, make the `modelDir` default point at
`$MOON_DIR` with OUR `enc.vmfb`/`decoder_redecode_cpu.vmfb`, and demote the vendor files to an explicit
`MOONSHINE_VENDOR=1` opt-in.
**Done when:** `voicecc asr test.wav` → "One, two, three." with no vendor Moonshine vmfb on the board
(`ls $MOON_DIR` shows only our artifacts), parity with the 3.0 transcripts.

> Open NPU item (tracked): our self-compiled encoder fragments into ~40 dispatches and returns zeros
> on-device, while the vendor `encoder.vmfb` is one fused dispatch (`docs/synaptics-support/README.md`).
> Until the vendor fusing compiler / tiling recipe lands, the *self-compiled* encoder default runs on
> `--device=local-task` (CPU); NPU stays on the vendor vmfb via `MOONSHINE_VENDOR=1`. This is the one
> place "fully self-compiled" and "on the NPU" still trade off — call it out honestly in the demo README
> once resolved.

### P3.2 Kill the last runtime Python (VAD in Kotlin)
`runListen` (`Pipeline.kt`) still `popen`s `scripts/vad_capture.py` (Silero VAD) via a venv. Replace it:
- **Option A (simplest, ship first):** an energy/zero-crossing gate in Kotlin/Native (ALSA capture via
  cinterop) — no new model. Good enough for push-to-talk / quiet rooms.
- **Option B (parity):** author Silero VAD in the DSL (needs an LSTM layer upstream — GRU is the template,
  guardrail discussion per PLAN Phase 4.1) or run Silero ONNX as a small IREE vmfb via the existing
  `torq-run-module` binding.
Then delete the `venvPython`/`helper` popen in `runListen` and `scripts/vad_capture.py`.
**Done when:** `voicecc listen mic` segments utterances with **no Python process** (verify on-board:
`ps | grep -i python` empty during a `listen` session), parity vs the Python helper on recorded wavs.

### P3.3 Retire bring-up stubs (no board)
Delete `src/jvmMain/kotlin/voicecc/export/{HloBridge,SdpaTrace}.kt` (PLAN 2.4) once nothing references them
(`grep -rn HloBridge\\\|SdpaTrace src/`). Retire build-only Python helpers already superseded by the DSL
(`add_argmax_perpos.py`, `make_f16.py`); keep one-time weight-extraction helpers but document them as
offline steps, not runtime.
**Done when:** `grep -rn 'popen\|\.py' src/` shows no *runtime* Python; README's "interim vendor" note is gone.

---

## P4 — KV-cache performance (board-verify the already-authored graphs)

The graphs + native loop exist; only the first-board confirmation of trace-derived unknowns remains.

### P4.1 FunctionGemma KV — follow `GEMMA-KV-BOARD-LOOP.md` exactly
```bash
GEMMA_KV=1 scripts/compile-gemma.sh board          # builds gemma-prefill.vmfb + gemma-with-past.vmfb
# deploy all three vmfbs + gemma-gen.irpa to $GEMMA_DIR, then:
GEMMA_KV=1 BOARD=root@<ip> ./gradlew deployBoard
GEMMA_KV=1 VOICECC_PROFILE=1 voicecc gen "turn the light on"
```
Confirm on the first run (both are silent-corruption traps, both caught by the oracle):
1. **dynamic `?` in `gemma_with_past` compiles** (first real test of the sentinel-relax; if iree rejects
   the dynamic concat, fall back to fixed-pad+mask — noted in the loop doc).
2. **per-block K-vs-V output order** — the converter's terminal ordering may emit V,K not K,V. Verify via the
   prefill-then-1-step logit match (`GEMMA-KV-BOARD-LOOP.md` §CAVEAT); swap if garbage.
**Done when:** the loop reproduces the oracle `[262146,236769,3255,718,498,1373,262152,106]` and
`PERF-LOGBOOK.md` gets the ms/token row (expect the O(seq²)→O(seq) collapse vs the ~6 s/token baseline).
Then make `GEMMA_KV=1` the default in `Pipeline.kt`.

Follow-up worth doing: have the exporter emit a tiny `.manifest` mapping each output slot → role (K/V/token),
so the K-vs-V order stops being a guess. Small change in `kgemma/FunctionGemmaExport.kt`.

### P4.2 Moonshine KV — same drill
`MoonshineKvDecoder` is a `⚠️ BOARD-UNVERIFIED DRAFT` with the same two unknowns (per-block K-vs-V,
vmfb entry-fn names). Verify `MOONSHINE_KV=1 voicecc asr test.wav` transcribes at parity, then default it on.
**Done when:** KV ASR transcribes correctly; row added to `PERF-LOGBOOK.md`.

---

## Final packaging — extract the standalone `skainet-fc-demo` repo

The demo dir is already clone-ready (bootstrap + demo.env + externalized scripts). Two things remain to make
it a *standalone* clone that doesn't need its 4 sibling repos:

1. **The `kgemma` FunctionGemma exporter is a Gradle task in `SKaiNET-transformers`**, not a published CLI —
   `compile-gemma.sh` does `cd ../../SKaiNET-transformers && ./gradlew :llm-runtime:kgemma:exportFunctionGemma`.
   For a standalone repo, publish that exporter as a runnable artifact (a thin `llm-apps` CLI or a fat jar on
   local Maven) and have `compile-gemma.sh` invoke the published tool instead of the sibling checkout. Same
   for `moonshineEncoderMlir`/decoder export tasks.
2. **`CompactCodec.TOKEN_TO_NAME` is a private hardcoded map upstream** — adding a tool means patching the
   dependency (see `FINETUNING.md`/`examples/custom-command`). Make the map injectable (constructor param or
   a `ToolMap` the demo passes in) so custom tools need zero upstream edits. Small, high-leverage upstream tweak.

Then: `git init skainet-fc-demo`, move this dir's contents in, set the Maven deps to published `0.35.0`
(already the version in `gradle/libs.versions.toml`), drop the composite `includeBuild`s, and re-run the
Quick start from a fresh clone as the acceptance test.
**Done when:** a clone into a fresh temp dir → `./bootstrap.sh` → `./gradlew :jvmRun` works with no sibling
repos present, and `deployBoard` runs on the board.
