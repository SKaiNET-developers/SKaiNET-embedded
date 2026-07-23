# PLAN — SL2610 FunctionGemma demo on the SKaiNET stack

> **THE single plan** (version-controlled). **Part A** is the forward-looking finish plan
> ("clone → run → finetune"); **Part B** is the detailed historical tracker (traceable checkbox log).
> This file supersedes and folds in the former loose `coraldevboard/PLAN.md` and `docs/FINISH-PLAN.md`.
> Legend: **[D]** demo (`SKaiNET-embedded/sl2610-function-calling`) · **[U-core]** SKaiNET ·
> **[U-tx]** SKaiNET-transformers · **[U-conf]** skainet-iree-conformance · **[coral]** board/Torq toolchain.

---

# Part A — Finish plan (forward)

## Goal (north star)

```
git clone <the port>  →  ./bootstrap.sh  →  BOARD=root@<ip> ./gradlew deployBoard  →  speak a command
                                                                                      →  then finetune FunctionGemma on MY commands
```

Everything on the board runs on our own self-compiled vmfbs (no vendor Moonshine binaries), zero runtime
Python, at KV-cache-usable latency — and a documented recipe lets you retrain FunctionGemma to understand
new commands and re-bake into the demo.

## Status (updated 2026-07-23)

**Landed & merged — PR #9 → `feature/google-io-demo` (2026-07-22):** clone-to-run packaging (single
`demo.env` config surface; every `scripts/*.sh` sources it, inline env wins, no hardcoded paths/IP;
`bootstrap.sh`; README Quick-start), the toolchain-currency decision (`TOOLCHAIN-PIN.md`), the finetuning
recipe (`FINETUNING.md` + `examples/custom-command/`), this plan, and the in-flight KV-cache decode drafts
frozen as a reproducible baseline (opt-in behind `GEMMA_KV=1` / `MOONSHINE_KV=1`; default paths unchanged).

- **Done:** P0 (docs reconciled, KV WIP frozen), **P1 packaging**, **P2 doc**, **P5 docs**. Also verified
  2026-07-23: the Moonshine **decoder + preprocessor + embeddings already default to OUR self-compiled
  artifacts** on the board (`MoonshineDecoder.kt`); the DSL encoder + decoder are authored and CPU-proven
  (`MoonshineDecoderE2ECpuTest`).
- **Remaining (needs board + the private Torq wheel):** **P3** — only the **encoder** default is still a
  vendor binary (NPU); making OUR encoder the default is blocked by the ~40-dispatch fusion gap, and VAD is
  still Python. **P4** — board-verify the Gemma + Moonshine KV loops (~6 s/token → KV win); **note the Gemma
  KV export is uncommitted upstream** (see P4). **P2 canary** re-validation of a newer Torq release. And the
  **standalone-repo extraction** (publish the `kgemma` exporter as a CLI + make `CompactCodec.TOKEN_TO_NAME`
  injectable). Executable steps: `BOARD-RUNBOOK.md`.

## Where we are (maturity)

The **hard ML/compiler risk is retired.** FunctionGemma self-compiles from the DSL (`compile-gemma.sh`,
Python-free) and is board-verified on CPU; the Moonshine **encoder** is DSL-authored and self-compiled; the
Moonshine **decoder** is DSL-authored and CPU-proven (`MoonshineDecoderE2ECpuTest` → "One, two, three.").
Engine + models are at a matched **0.35.0**; IREE conformance last green on 0.33.0. What's left is
**productization, board performance verification, and extensibility** — not research. The granular status of
every model/op/toolchain item lives in Part B.

## Definition of done (all four — locked with the user 2026-07-11)

1. **Reproducible clone-to-run** — fresh clone + bootstrap + `deployBoard`, no manual path/IP surgery.
2. **Finetuning recipe** — scripted path to retrain FunctionGemma on custom commands and re-bake.
3. **Fully self-compiled, zero Python** — our encoder+decoder as the board default; VAD in Kotlin.
4. **Usable performance** — Gemma (and Moonshine) KV-cache decode board-verified.

## Phases

> Paths below are relative to the demo root `SKaiNET-embedded/sl2610-function-calling/`; the docs named here
> (`TOOLCHAIN-PIN.md`, `FINETUNING.md`, `BOARD-RUNBOOK.md`, `PERF-LOGBOOK.md`) are this file's siblings in `docs/`.

### P0 — Freeze & reconcile ✅
Froze the in-flight KV WIP as a reproducible baseline and consolidated the plan docs (this file). Merged in PR #9.

### P1 — Reproducible clone-to-run → *done-when #1* — **packaging ✅, standalone extraction remaining**
Packaging = one consolidated demo (engine + transformers as published **0.35.0** Maven artifacts).
- ✅ `demo.env(.example)` single config surface; `bootstrap.sh`; README Quick-start; config-externalized
  `scripts/{deploy,compile-gemma,iree-compile-torq*,moonshine_compile_preprocessor}.sh` (no hardcoded paths/IP).
- ◻ **Standalone extraction:** the `kgemma` FunctionGemma exporter is a Gradle task in `SKaiNET-transformers`,
  not a published CLI — a standalone clone must fetch that repo or the exporter must ship as a runnable
  artifact. Also make `CompactCodec.TOKEN_TO_NAME` injectable. Steps in `BOARD-RUNBOOK.md`.

### P2 — Toolchain currency: re-validate & decide ✅ (doc) / ◻ (canary) → *honors "latest libs/tools"*
Policy: re-validate the newest Torq release; adopt if it passes the canary gate, else keep g165e12a and
document the pin. Board OS SDK already current (`scarthgap_6.12_v2.4.0`).
- ✅ Decision + procedure in `TOOLCHAIN-PIN.md` (supersedes the raw audit at the workspace-root
  `docs/torq-debug-notes/torq-toolchain-update.md`).
- ◻ **Canary gate:** compile encoder + Gemma with the candidate, deploy, check the silent-zeros probe + the
  token oracle `[262146,236769,3255,718,498,1373,262152,106]`. Needs board + the private wheel.

### P3 — Fully self-compiled, zero Python → *done-when #3* — **partially done: decoder ✅, encoder + VAD remain**
- ✅ **Decoder, preprocessor, and token embeddings already default to OUR self-compiled artifacts**
  (`MoonshineDecoder.kt`: `decoder_redecode_cpu.vmfb` on `local-task`, `preprocessor_cpu.vmfb`,
  `our_embed_tokens.npy`). The vendor **decoder** is no longer in the default path (3DEC-e / 3D-d).
- ◻ The **encoder** default is still the vendor NPU vmfb (`$modelDir/encoder.vmfb` on `torq`); OUR encoder is
  opt-in (`MOONSHINE_ENCODER_VMFB` + `MOONSHINE_ENCODER_DEVICE=local-task`) and runs on **CPU** because it
  fragments to ~40 dispatches → zeros on the NPU (dispatch-fusion gap, `docs/synaptics-support/README.md`).
  Making our encoder the board default (ideally on the NPU) is the open self-compiled item.
- ◻ Reimplement Silero VAD in Kotlin (DSL or ported gate logic) to kill `scripts/vad_capture.py` — the last
  runtime Python. Retire bring-up stubs `HloBridge.kt` / `SdpaTrace.kt` (item 2.4). Steps in `BOARD-RUNBOOK.md`.

### P4 — Usable performance via KV-cache → *done-when #4* — **◻ needs board**
- Board-verify the Gemma KV 2-graph loop (`GEMMA_KV=1` → `gemma-prefill.vmfb` + `gemma-with-past.vmfb`,
  driver `GemmaKvDecoder`); confirm the flagged unknowns (arg order, K-vs-V output order, entry-fn names).
- Board-verify the Moonshine KV decode (`MOONSHINE_KV=1`, `MoonshineKvDecoder`).
- ⚠️ **Provenance (verified 2026-07-23):** the Gemma KV export (`exportPrefill`/`exportWithPast` + int8 in
  `kgemma/FunctionGemmaExport.kt`) and its tests are **uncommitted** in SKaiNET-transformers (dirty
  `release/0.35.0` tree); the committed/released 0.35.0 ships only the re-decode bf16 export. Commit + release
  that upstream before the `GEMMA_KV=1` board loop is reproducible from a clean clone.
- Record before/after in `PERF-LOGBOOK.md`. Full steps + the K-vs-V trap in `GEMMA-KV-BOARD-LOOP.md`.

### P5 — Finetuning recipe (custom commands) ✅ (docs) → *done-when #2, the downstream goal*
- ✅ `FINETUNING.md`: define new tools → dataset in the Octopus-v2 prompt format → LoRA train (offline
  Python OK) → merge → GGUF → `GEMMA_GGUF=… compile-gemma.sh board` → `deployBoard` → verify. The finetuning
  story is literally "swap the GGUF."
- ✅ `examples/custom-command/`: a minimal worked example (one new tool + a tiny dataset template).
- ◻ Board-verify the loop end-to-end on real hardware (part of P3/P4 board time).

## Sequencing
`P0 → P1 → P3 → P4 → P5` (P5 docs done; P5 board-verify rides P3/P4); `P2 canary` slots in when the board is
free. Biggest residual risk: the KV-cache K-vs-V / entry-fn unknowns (P4) — the only items needing live board
debugging. Everything else is packaging/wiring/docs over already-proven components.

## Verification (end-to-end, per phase)
- **P1:** clone into a fresh temp dir → `./bootstrap.sh` → `./gradlew :jvmRun` with no edits to any path.
- **P2:** candidate toolchain passes the canary + token oracle; decision written to `TOOLCHAIN-PIN.md`.
- **P3:** board run with vendor `modelDir` unset and no Python process (verify `ps` on-board during `listen`).
- **P4:** `VOICECC_PROFILE=1 voicecc gen "turn the light on"` shows KV latency well under the re-decode
  baseline; token stream matches the oracle.
- **P5:** run `examples/custom-command/` end-to-end → new spoken command routes to the new tool.

---

# Part B — Detailed item log (history + traceable checklist)

> The granular tracker for `spec.md`. Each item names its owning repo and a concrete "done means".
> Paths here are as originally written (relative to the demo root, or to the workspace root for
> `docs/torq-debug-notes/…` and `docs/torq-repro/…`); from this file's `docs/` location prepend `../` (demo
> root) or `../../../` (workspace root) as needed.

## Baseline (updated 2026-07-11)

Original 2026-07-02 baseline, with what has since landed:
- Demo Phase 0 done: KMP scaffold, ActionRouter, board deploy works (`deployBoard`).
- **Phase 1 ✅** — demo consumes upstream `runtime-gemma-iree` (`GemmaDecoder`, `IreeRuntime`,
  `CompactCodec`); the demo-local runtime/codec near-copies were deleted (1.3).
- Demo now pins transformers **0.35.0**; conformance last fully green on **0.33.0**
  (12/12 models, 33/33 ops, stock IREE 3.11.0) — conformance re-run for 0.34/0.35 is a tracked item (C.3).
- **ASR is Python-free** (3.0); the last runtime Python is **VAD** (`scripts/vad_capture.py`, Phase 4.1).
- No `linuxX64` demo target yet (host-native requirement unmet — track H).
- **Moonshine encoder authored in the DSL + self-compiled + on the NPU** (see the 2026-07-05 STATUS
  banner below); the **decoder** is authored + CPU-proven, board path still on vendor vmfbs (3DEC).
- NEON C kernels exist (`skainet_simd.h`) but are BOARD-VERIFY-PENDING; `native-cpu` has no
  `linuxArm64` target (track S).
- The former Torq NPU blocker (`getWeightMemoryFormat` segfault on fp32 weights) is **solved** —
  bf16-native weights compile (see 3N-a/b). Residual tiling/LRAM limits tracked in `TOOLCHAIN-PIN.md`.

## Phase 1 — Adopt upstream Gemma runtime (dedup + versions)

- [x] **1.1** [D] Opt-in composite builds (`includeBuild` SKaiNET + SKaiNET-transformers,
      default ON, `useLocalStack=false` opts out) in demo `settings.gradle.kts`.
      Transformers coordinates need explicit `dependencySubstitution` (project names ≠
      published artifact ids — same pattern as skainet-iree-conformance). Demo Gradle
      wrapper bumped 9.3.1 → 9.6.1 (SKaiNET's AGP requires ≥ 9.4.1 under composite).
      *Done 2026-07-02:* both targets compile with `sk.ainet.*` resolved from source.
- [x] **1.2** [D] Bump `skainetTransformers` 0.25.0 → 0.33.0 in `gradle/libs.versions.toml`;
      add `skainet-transformers-runtime-gemma-iree` dependency.
      *Done 2026-07-02:* jvm + linuxArm64 builds green, no API drift hit.
- [x] **1.3** [D] Delete demo copies of `IreeRuntime.kt`, `llm/CompactCodec.kt` + inline decode
      loop in `Pipeline.kt`; consume `sk.ainet.transformers.gemma.iree.{IreeRuntime,GemmaDecoder,
      CompactCodec}`. Codec grammar parity verified by diff (identical except `Intent`→`ToolCall`
      return type; demo maps `ToolCall`→`Intent`). `GemmaProbe.kt` retired (superseded).
      *Done 2026-07-02:* jvmTest green, release aarch64 binary deployed via **adb**
      (`adb connect 192.168.3.26 && adb push …/voicecc-kt/voicecc`) and verified on-board:
      `cmd.wav` → ASR "Turn the light on" → `<tool_0>(state="on")<end>` → `set_lights state=on`
      dispatched. Observed ~95 s/turn wall (ASR via Python venv + per-token vmfb re-invocation)
      — the 4.4 latency gate (target ~1.3–2.0 s/turn) is the relevant baseline to beat.
- [ ] **1.4** [U-tx] Board-specific decoder tweaks land in `gemma-iree` via feature-branch PR,
      not in the demo. *Done:* PR merged; demo consumes it.

## Phase 2 — Productize DSL→StableHLO→IREE export

> Planning-phase design input: **`PROPOSAL-f16-storage.md`** — retire `make_f16.py` by moving
> weight conversion into the export/bake (2.1/2.2). **A/B done 2026-07-02: bf16 weights are a
> bit-exact drop-in for f16 on FunctionGemma (board-verified) → bake bf16 (reuses core's existing
> `Bf16DenseTensorData` + SafeTensors `KEEP_NATIVE`), do NOT add f16 to core.** So 2.1/2.2 should
> emit bf16 weight globals + `convert bf16→f32`; no new dtype needed.

- [x] **2.1** [U-tx] Promote FunctionGemma export into a runnable exporter: GGUF → `gemma-gen.mlir` + `.irpa`,
      absorbing the demo's `add_argmax_perpos.py` transform.
      *Done 2026-07-09:* `kgemma` `exportFunctionGemma` + `scripts/compile-gemma.sh` — one Python-free
      command; the argmax tail is the DSL `argMax` op and weights are emitted as bf16 externals.
- [ ] **2.2** [U-core] Fix `IrpaWriter` IREE-v0 header compat in `skainet-io-iree-params`
      (drop `iree-convert-parameters` workaround). *Done:* baked `.irpa` loads directly; conformance ⚠️ → ✅.
- [ ] **2.3** [D] Dockerized `iree-compile` (stock 3.11.0 host / Torq fork g165e12a board)
      replacing `scripts/iree-compile-*.sh` + `make_f16.py`. *Done:* scripts reproducibly emit host + board vmfbs.
- [ ] **2.4** [D] Retire `HloBridge.kt`/`SdpaTrace.kt` bring-up stubs. *Done:* deleted; README updated.

## Phase 3 — Moonshine ASR on the SKaiNET stack (re-planned 2026-07-02)

Spec NPU-parity acceptance: Moonshine runs on the Torq NPU on the board (board-only, via Torq-fork
IREE tooling); on the Intel host it falls back to IREE llvm-cpu.

**Re-plan:** 3.0 shipped Python-free ASR but on **vendor prebuilt vmfbs + vendor runtime** — NOT the
SKaiNET stack. The real work is Moonshine **authored in the NN DSL, compiled by us**, leading with the
**hardest problem** (Torq NPU crash). Decisions: author in DSL; NPU-bisect + CPU-compile in parallel.

### 3N — HARDEST FIRST: Torq NPU compile crash · [D]+[coral] · ROOT-CAUSE SOLVED 2026-07-02
- [x] **3N-a** Minimal reproducer: a **single `dot_general` with an fp32 weight** crashes
      `getWeightMemoryFormat` (`Kernel.cpp:2602`) — NOT SDPA-specific. `docs/torq-repro/t1_matmul.mlir`.
- [x] **3N-b** **Workaround proven: bf16-native weights compile** (`t2_bf16.mlir`). Root cause = compiler
      has no fp32→bf16 weight conversion; weight must be bf16 AT THE MATMUL. `docs/torq-npu-weight-crash.md`.
- [~] **3N-c** Full bf16 attention: the `MatMulPattern.cpp:57` K-transpose issue is **fixed** and the full
      6-layer encoder now **compiles** on the Torq NPU. Residual: it fragments to ~40 dispatches and returns
      **zeros on-device** (the runtime executes only a few dispatches / no fusion), while the vendor
      `encoder.vmfb` is one fused dispatch. This is the standing blocker for a self-compiled NPU encoder —
      escalation + analysis in `docs/synaptics-support/README.md` and `TOOLCHAIN-PIN.md`.

### 3D — Moonshine in the NN DSL (our-stack build; parallel to 3N) · [U-tx]+[D]
- [x] **3D-a** Author Moonshine-tiny in the NN DSL (`llm-inference:moonshine`): conv preprocessor,
      RoPE encoder, decoder + cross-attention; **bf16-native weights**.
      *✓ 2026-07-23:* `MoonshineEncoder.kt` + `MoonshineDecoder.kt` + `MoonshinePreprocessor.kt` in the DSL.
- [x] **3D-b** Weight loader (ONNX/safetensors via `skainet-io`) + tokenizer (`GGUFTokenizer.fromTokenizerJson`).
      *✓ 2026-07-23:* `MoonshineDecoderWeights.kt` layer-qualified loaders + `convert_moonshine_weights.py`
      (currently in **jvmTest**, not prod `commonMain` — a hardening follow-up).
- [x] **3D-c** Export DSL→StableHLO → compile aarch64 **CPU** vmfbs (proven-viable). Depends 3D-b.
      *✓ 2026-07-23:* encoder/decoder/preprocessor CPU vmfbs build and run.
- [x] **3D-d** Rewire `voicecc/asr/MoonshineDecoder.kt` to run OUR vmfbs on `--device=local-task`.
      *✓ 2026-07-23:* the demo's **decoder + preprocessor default to OUR CPU vmfbs**; only the encoder default
      is still the vendor NPU vmfb (see P3).
- [ ] **3D-e** Once 3N-c unblocks: compile the DSL attention to **Torq NPU**; ASR on the NPU. *(encoder still
      CPU-only — the ~40-dispatch fusion gap.)*
- [ ] **3C** [U-conf] Conformance models `moonshine-encoder` + decoder step (+ new ops). Depends 3D-a/b.
- [ ] **3R** [U-tx] Extract a reusable IREE runtime (shared `iree-runtime` out of `gemma-iree` +
      `TorqRunModule`) so ASR + LLM share one driver. Depends 3D-d.

### ★ STATUS 2026-07-05 — ENCODER DONE, DECODER IS THE NEXT MILESTONE
The Moonshine **encoder** is complete and shipped: authored in the NN DSL, **published** as
`sk.ainet.transformers:skainet-transformers-inference-moonshine` (0.34.0; **0.34.1** layer-qualifies its
param names), matches the reference (cos **1.0** f32 CPU, **~0.9998** bf16 CPU), **compiles + runs on the
SL2610 NPU**, e2e-proven `test.wav` → encoder (Torq) → decoder → "One, two, three.". So 3N-c ✅, 3D-a
(encoder half) ✅, 3D-c ✅, 3D-e ✅. The **decoder** in that run is *external* (HF harness / vendor vmfbs) —
never authored in the DSL. That is the 3DEC gap.
> **Correction (2026-07-23):** the "3D-e ✅ / 3N-c ✅" NPU claim was over-optimistic — the self-compiled
> encoder *compiles* on the NPU but returns **zeros at runtime** (~40 dispatches, no fusion). See the
> reconciled 3N-c `[~]` and 3D-e `[ ]` above; the shipped encoder default remains the vendor NPU vmfb.

### 3SC — Finish self-compiled ENCODER integration (near-term) · [D]+[U-tx]
- [x] **3SC-a** [D] Name-based Moonshine weight mapping (layer-qualified). *✓ 2026-07-23.*
- [x] **3SC-b** [D] `safetensors → per-tensor .bin` converter (`convert_moonshine_weights.py`). *✓ 2026-07-23.*
- [x] **3SC-c** [D]+[coral] Dockerized Torq iree image with the real `torq_compiler` wheel; `moonshineEncoderMlir`
      → `iree-compile-torq-docker.sh` → `encoder.vmfb`. *✓ 2026-07-23.*
- [~] **3SC-d** [D]+[coral] Deploy + run OUR encoder.vmfb on the board via `MOONSHINE_ENCODER_VMFB`.
      *Partial:* runs on **CPU** (`local-task`) and transcribes; on the **NPU** it zeros (3N-c dispatch-fusion),
      so the shipped encoder default stays the vendor NPU vmfb.

### 3DEC — Moonshine DECODER in the NN DSL (the milestone) · [U-tx]+[D]+[U-conf]
> Goal: a **fully self-compiled** Moonshine ASR — no vendor Moonshine binaries. Two graphs mirroring the
> vendor: `decoder` (prefill → logits + self/cross KV) and `decoder_with_past` (1 token + cached KV → logits).
- [x] **3DEC-a** [U-tx] Author `moonshineDecoder()` (token embed, pre-norm causal self-attn + cross-attn +
      **gated-SiLU** MLP, final norm + LM head tied to `embed_tokens`).
      *✓ 2026-07-23:* `MoonshineDecoder.kt` (impl complete; its top KDoc still says "GELU stub" — stale).
- [x] **3DEC-b** [U-tx] Emit the two decode graphs (`forwardPrefill` + `forwardWithPast`) with KV-cache I/O.
      *✓ 2026-07-23:* `MoonshineDecoderPrefillMlirDumpTest` + `MoonshineDecoderWithPastMlirDumpTest` export/compile.
- [x] **3DEC-c** [U-tx]+[D] Decoder weight mapping + validation.
      *✓ 2026-07-23:* `MoonshineDecoderE2ECpuTest` — full DSL encoder→decoder transcribes "One, two, three."
      on **CPU** (env-gated; skips without a checkpoint) + a KV two-graph loop test.
- [ ] **3DEC-d** [D]+[coral] **Torq-compile** the decoder graphs. *Done:* decoder vmfbs run on the SL2610 NPU.
      *(open — decoder currently runs on CPU.)*
- [x] **3DEC-e** [D] Rewire `MoonshineDecoder.kt` to drive **our** decoder vmfb.
      *✓ 2026-07-23:* the demo's decoder default is `decoder_redecode_cpu.vmfb` (ours) on `local-task`; the
      vendor decoder is out of the default path. **Caveat:** CPU, not NPU (3DEC-d); the encoder default is
      still vendor. So ASR is *decoder-self-compiled* today, not yet *fully* vendor-free.
- [ ] **3DEC-f** [U-conf] Conformance: `moonshine-decoder` prefill + step models (+ new ops). *(extends 3C)*
- [x] **3DEC-g** [U-tx] Release transformers **0.35.0** shipping the decoder; demo consumes it.
      *Done 2026-07-09:* 0.35.0 ships `moonshineDecoder()` (self/cross KV graphs) + FunctionGemma
      self-compile; demo pins 0.35.0; `MoonshineDecoderE2ECpuTest` transcribes on CPU.

> **Progress since 2026-07-05 (updated 2026-07-11).** The Moonshine decoder is now **authored in the DSL and
> CPU-proven** (3DEC-a/b/c largely landed), and transformers **0.35.0** is released (3DEC-g ✅). What remains
> for a *fully self-compiled, no-vendor* ASR: Torq-compile the decoder graphs (3DEC-d) and rewire + delete the
> vendor decoder (3DEC-e), both needing board time (tracked as Part A → P3).

- [x] **3.0** [D] **STOPGAP (vendor binaries, not SKaiNET stack)** — NPU-now, Python-out: drive the prebuilt
      Synaptics Moonshine vmfbs from Kotlin via the board's native `torq-run-module`, replacing the
      `moonshine_npu.py` + onnxruntime + torq.runtime subprocess. New Kotlin/Native code in
      `src/linuxArm64Main/kotlin/voicecc/asr/`: `MoonshineDecoder`, `TorqRunModule`, `Wav`,
      `Bin`/`Bf16EmbeddingTable`. Preprocessor compiled to aarch64 llvm-cpu via the Torq-fork iree-compile.
      *Done 2026-07-02:* `voicecc asr cmd.wav` → "Turn the light on", `test.wav` → "One, two, three.",
      **zero Python processes**, ASR on the NPU ~3.5 s. `moonshine_npu.py` deleted.

> **3.1–3.5 SUPERSEDED (2026-07-02 re-plan).** The original Moonshine items were replaced by the
> hardest-first re-plan above: **3N** (Torq NPU crash), **3D** (Moonshine in the DSL), **3SC** (self-compiled
> encoder integration), **3DEC** (DSL decoder). The Python stopgap they anticipated shipped as **3.0**.

## Phase 4 — VAD, mic, full pipeline, parity gate

- [ ] **4.1** [U-core]+[D] Kotlin VAD replacing `vad_capture.py`. **Decision point:** (a) Silero port —
      needs new LSTM layer upstream (GRU is the template), (b) Silero ONNX via IREE, (c) energy/GRU VAD.
      *Done:* segmentation parity vs the Python helper on recorded wavs.
- [ ] **4.2** [D] Native mic capture (ALSA via cinterop) as a small KMP module.
- [ ] **4.3** [D] All-Kotlin `listen` pipeline; delete runtime `scripts/*.py`.
- [ ] **4.4** [D] Parity/latency gate: scripted A/B vs the original Python demo (same wavs → same tool calls,
      per-stage timings, target ≥ ~1.3–2.0 s/turn). *Done:* committed benchmark report; gate fails on regression.

## Host-native track (Intel/Linux, both models on CPU)
- [ ] **H.1** [D] Add `linuxX64()` target; move `Pipeline.kt`/`Main.kt` into a shared `nativeMain` source set.
- [ ] **H.2** [D] Host vmfbs (stock IREE, llvm-cpu AVX) for Gemma + Moonshine via 2.3.

## ARM SIMD kernel track
- [ ] **S.1** [U-core] Add `linuxArm64` to `skainet-backend-native-cpu`; execute + bit-check the NEON paths
      in `skainet_simd.h` via QEMU + on-board.
- [ ] **S.2** [U-core]+[D] Wire `NativeKnKernelProvider` into demo eager paths; microbench vs IREE llvm-cpu.
- [ ] **S.3** [U-core] ADR: Kotlin/Native has no SIMD intrinsics → Kotlin orchestration + C-NEON cinterop stays.

## Conformance track
- [ ] **C.1** [U-conf] Add `functiongemma-270m` model (argmax-tail variant).
- [ ] **C.2** [U-conf] Moonshine models + new ops (tracks 3C/3DEC-f).
- [ ] **C.3** [U-conf] Re-run full suite on every version bump the demo consumes; keep `conformance-history.md` current.

## Documentation track (Diátaxis via dockerized Antora)
- [ ] **D.1** [D] Antora docs component for the demo (tutorial/how-to/reference/explanation).
- [ ] **D.2** [U-tx] How-to pages for `gemma-iree` + new `moonshine` modules.

## Top risks
1. Torq NPU compiler segfault/fusion on Moonshine — external; mitigation is the vendor-vmfb path (3.0), which
   still satisfies NPU-parity acceptance. See `TOOLCHAIN-PIN.md` + `docs/synaptics-support/README.md`.
2. KV-cache board loops unverified — fixed-prefill re-decode proven for Gemma seq=24; KV drafts board-pending (P4).
3. LSTM missing for faithful Silero VAD — upstream work behind guardrail discussion, or pick an alternative (4.1).
4. 1.9 GB board memory ceiling — ASR + LLM need the sequential load/free discipline `Pipeline.kt` already encodes.

## Acceptance verification (maps to spec)
- Board: `BOARD=root@<ip> ./gradlew deployBoard` → spoken/wav command → correct tool call (six actions),
  no Python on the board, latency ≥ original demo.
- NPU parity (spec): Moonshine ASR on the Torq NPU on the board; host uses IREE llvm-cpu for the same model.
- Host: `./gradlew linkReleaseExecutableLinuxX64` → same commands on the Intel host, both models on CPU.
- Conformance: full suite green on the pinned pair incl. `functiongemma-270m` + `moonshine-*`.
- Kernels: linuxArm64 NEON parity tests green on the board.
- Dedup: grep gate — no `IreeRuntime`/decoder/codec copies left in the demo.
