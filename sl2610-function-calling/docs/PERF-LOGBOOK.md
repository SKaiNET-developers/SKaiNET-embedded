# FunctionGemma / SL2610 — Performance Logbook

Traceable record of the perf-optimization program (plan: *Make FunctionGemma function-calling usable on the
SL2610*). Every phase that lands an on-board change **must append a measured row** here — this is both the
status tracker and the acceptance gate for each phase.

**Board:** SL2610, 2× ARM Cortex-A55 (in-order), 1.9 GB RAM. **Runtime:** g165 Torq-fork `iree-run-module`
(subprocess). **Oracle:** `"turn the light on"` → `[262146,236769,3255,718,498,1373,262152,106]` =
`<tool_0>(state="on")<end>` (token-for-token vs llama.cpp).

## How to measure

```sh
# on the board, per-step timing + child (iree-run-module) peak RSS to stderr:
VOICECC_PROFILE=1 voicecc gen "turn the light on"
# [perf] gemma step N: <ms> ms (rss <kB>)
# [perf] gemma total: <ms> ms, <n> tokens, <ms> ms/token
```
Knobs: `GEMMA_TASK_GROUPS=N` (local-task worker groups = cores; default 2, `0` disables the flag).
`ms/token` is the headline number. `rss` is the `iree-run-module` child peak (RUSAGE_CHILDREN), not the driver.

## Measurements

| date | phase / change | seq | tokens | total ms | ms/token | child RSS | notes |
|------|----------------|-----|--------|----------|----------|-----------|-------|
| 2026-07-05 | **baseline** (fixed seq=24 re-decode, per-step subprocess, `drop_caches`, bf16→f32 load) | 24 | 8 | ~48000 | ~6000 | ~930 MB | starting point that motivated this program |
| _pending_ | Phase 0 quick wins (remove `drop_caches`, `--task_topology_group_count=2`) | 24 | 8 | | | | warm-mmap + both cores; measure on board |

<!-- Append one row per phase after its on-board run. Keep the baseline row first. -->

## Phase ledger (status)

- **Phase 0** — logbook + `VOICECC_PROFILE` harness + quick wins (remove `drop_caches`, thread count). _Code landed; awaiting board measurement._
- **Phase 1** — KV-cache `with_past` Gemma decoder (DSL) + CPU token parity. **✅ DONE (CPU).** `GemmaModel.forwardPrefill`/`forwardWithPast`/`buildRopeCosSin` + `RoPE.buildSplitHalfCosSin`/split-half `forwardWithCosSin`. `FunctionGemmaWithPastCpuTest` drives the 2-graph KV loop eagerly and matches the oracle token-for-token `[262146,236769,3255,718,498,1373,262152,106]`. On-device speedup lands in Phase 2 (needs the compiled graphs + board loop).
- **Phase 2** — export both graphs + board 2-graph runtime loop (the on-device KV win). _Export DONE (host-verified); board parts remain._
  `FunctionGemmaExport.exportWithPast` → `func @gemma_with_past` and `exportPrefill` → `func @gemma_prefill`
  (bf16 externals, shared "model" irpa scope, argMax tails). **Riskiest unknown RESOLVED — with a correction:**
  a naive `-1` dynamic placeholder mis-infers `concat(?,1) = -1+1 = 0` → broken `1x1x0x256` OUTPUT caches
  (inputs looked fine, which fooled the first probe). Fix = trace at a **sentinel prime (7919)** so concat
  infers `7919→7920`, then regex-relax both to `?`. `FunctionGemmaWithPastMlirDumpTest` now confirms input
  AND return caches are `1x1x?x256`, no `x0x256`, no sentinel leak; static probe cache-in `x7x256`→`x8x256`;
  prefill emits initial K/V `1x1x16x256`. iree-compile must confirm the dynamic concat on the board.
  Wiring DONE + verified: `GEMMA_GRAPH=redecode|prefill|with_past|all` gradle switch (ran on GGUF, emits the
  MLIRs), `compile-gemma.sh GEMMA_KV=1` builds all three vmfbs sharing one irpa (re-decode stays default).
  Exact I/O order captured (with_past args: token@0, sliding cos/sin@1-2, layers0-4 K/V@3-12, global
  cos/sin@13-14, layers5-17 K/V@15-40; returns 36 K/V then token). Stacked-I/O attempt reverted (converter
  unpacks it). **Native board loop DRAFTED + compiles for linuxArm64**: `IreeRuntime.invokeFiles` (raw-bin
  I/O), `GemmaKvDecoder` (prefill→with_past loop + host `splitHalfCosSin` + `Bin` f32/i32 I/O), wired into
  `Pipeline` behind `GEMMA_KV=1`; spec + board-confirm points in `docs/GEMMA-KV-BOARD-LOOP.md`. REMAINING
  (board only): `GEMMA_KV=1 compile-gemma.sh board` (first real test of the dynamic-`?` vmfb), deploy, run,
  confirm the two flags (`kFirstInOutput`, raw-vs-npy output — both caught by the oracle check), append the
  measured ms/token row.
- **Phase 3** — bf16 through the matmul (drop convert-on-load). **⏸ INVESTIGATED → DEFERRED (low value here).**
  Three findings: (1) the A55 has **no bf16 FMA** (`skainet_simd.h`: asimddp yes, bf16 no) → a bf16 matmul is
  emulated (widen-to-f32 to multiply), so **no FLOP win**; (2) weights are already stored bf16 in the irpa
  (2 B/elem) and iree-compile almost certainly **fuses** the elementwise `convert bf16→f32` into the matmul
  (widen in-register, no f32 materialization) → **no bandwidth win** either; (3) the clean implementation
  (`DtypeForwardPropagationPass("BF16")` to make the graph bf16-native) **OOM-kills** on 270M — the pass over
  the embedded-constant graph + bf16 re-conversion pushes peak memory past the host. Net: bf16-through-matmul
  is speculative-to-zero on this CPU and infeasible as a graph pass here. The real dtype/memory win is
  **quantization (Phase 5)** — int8/Q4 with the A55's `asimddp` udot. Phase 3 reverted; revisit only for a
  target with native bf16 FMA. _No logbook row (nothing shippable)._
- **Phase 4** — persistent in-process runtime (kill subprocess/mmap floor). _Not started._
- **Phase 5** — quantized compiled path (int8/Q4). **✅ int8 weight-only DONE + host-verified (board bring-up remains).**
  `FunctionGemmaExport.export(quantizeInt8=true)` (opt-in `GEMMA_QUANT=int8`) quantizes the 200 2-D matmul
  weight globals to **per-row (per-output-channel) symmetric int8** — `tensor<rows x cols x i8>` + a
  `tensor<rows x f32>` scale, dequant'd in graph (`convert i8->f32` + `broadcast_in_dim` scale + `multiply`);
  1-D norm globals stay bf16. Done in the safetensors writer + a text rewrite (NOT a graph pass → no Phase-3
  OOM). `FunctionGemmaInt8QuantTest` confirms: **weightMiB 831→418** (the RAM win on the 1.9 GB board),
  200 i8 globals each with a scale, in-graph dequant present, norms bf16. `compile-gemma.sh` forwards
  `GEMMA_QUANT`. This is the memory + weight-bandwidth win (weight-only; f32 activations, f32 accumulate).
  A further **int8-udot COMPUTE** path (quantize activations too → int8×int8→i32 on the A55's `asimddp`) is a
  bigger net-new step (activation calibration/requant), deferred. REMAINING (board only): `GEMMA_QUANT=int8
  compile-gemma.sh board` — confirm iree accepts the int8 dequant graph + per-row-int8 numeric quality (oracle
  token check) + measure. The eager Q5_K/Q4_K NEON path (already board-verified) is the ship-anyway fallback.
- **Phase 6** — board-wire Moonshine KV decode (shared infra). **✅ DRAFTED + compiles for linuxArm64 (board bring-up remains).**
  `MoonshineKvDecoder` (voicecc.asr) — the seq2seq analogue of `GemmaKvDecoder`: prefill-once (embeds(START)
  + encoder memory → logits + per-layer self-K/V + **cross-K/V**) then a with_past loop (token-embed + INTERLEAVED
  cos/sin + self-K/V + fixed cross-K/V → logits + extended self-K/V), driven via the existing `TorqRunModule`
  raw-bin I/O + `Bin`/`Bf16EmbeddingTable`. Cross-K/V computed once, re-fed every step; host `interleavedCosSin`
  port (rotaryDim 32, freqDenom=rotaryDim); K/V + cos/sin bf16, logits f32. Wired into `MoonshineRunner` behind
  `MOONSHINE_KV=1` (re-decode stays default). The `with_past`/prefill EXPORTS already exist
  (`MoonshineDecoderExport`, `DEC_GRAPH=with_past|prefill`). Same board-confirm flags as Gemma (`kFirstInOutput`
  K-vs-V, vmfb entry fn names). REMAINING (board only): compile the prefill + with_past decoder vmfbs
  (`compile-moonshine-decoder.sh`), deploy, `MOONSHINE_KV=1 voicecc pipeline`, confirm the flags against a
  reference-clip transcription, append the ASR ms/token row.
