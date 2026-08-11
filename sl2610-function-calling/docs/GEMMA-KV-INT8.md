# FunctionGemma KV-cache + int8 — landed status & open issues

The KV-cache 2-graph decode + int8 quant is the fix for FunctionGemma's ~6000 ms/token re-decode
(`PERF-LOGBOOK.md` baseline) — plan item **P4**. This tracks what's **landed** and every **board / release /
philosophy** step still required.

## GitHub issues (filed 2026-07-24)
| Open item | Issue |
|---|---|
| Board-verify the KV loop | SKaiNET-embedded #12 |
| int8 on-board | SKaiNET-embedded #13 |
| MLIR-text-rewrite → graph transforms (philosophy debt) | SKaiNET-transformers #248 |
| Release KV/int8 + demo bump | SKaiNET-transformers #249 |
| Test-heap default OOM | SKaiNET-transformers #250 |
| Conformance rows | skainet-iree-conformance #24 |

## Landed (2026-07-23)
- **SKaiNET-transformers PR #245** (`feat/functiongemma-kv-int8` → `develop` 0.36.1): rescued the work from
  uncommitted WIP, forward-ported onto develop, **CPU-verified**. All three tests pass with the Q5_K_M GGUF:
  `FunctionGemmaWithPastCpuTest` (token-for-token `[262146,236769,3255,718,498,1373,262152,106]`),
  `FunctionGemmaWithPastMlirDumpTest` (dynamic `x?x256` relax, no `x0x256`/sentinel leak, prefill `x16x256`),
  `FunctionGemmaInt8QuantTest` (`i8Globals==scaleGlobals`, ~half archive).
- The decode graphs are **DSL/DAG-authored** (`GemmaModel.forwardPrefill`/`forwardWithPast` compose module
  forwards; `RoPE.buildSplitHalfCosSin` gives runtime-position cos/sin).

## Board verification — ✅ DONE 2026-08-11 (SL2610, g165 Torq-fork)
Runbook: `docs/GEMMA-KV-BOARD-LOOP.md` (now carries the full resolution table). All six confirmed:
1. **Per-block K-vs-V output order** — **K then V**: `GemmaKvDecoder.kFirstInOutput=true`. The draft's
   return-SSA "(V,K)" analysis was WRONG (it read SSA id order, not the defining ops); oracle parity confirms.
2. **`--output=@file` format** — extension-driven; `@file.bin` is RAW little-endian bytes (as
   `IreeRuntime.invokeFiles` assumes). `.npy` would add a header.
3. **`gemma_with_past` input arg order** — exactly as trace-derived; confirmed against the compiled vmfb.
4. **Dynamic-concat** — the true-dynamic (`1x1x?x256`, T2.2/#248) MLIR compiles on the g165 Torq-fork and one
   vmfb served every position. No `GEMMA_SENTINEL_PAST=1` rollback, no fixed-pad+mask fallback needed.
5. **`--task_topology_group_count`** — accepted by the board `iree-run-module`.
6. Torq-fork `iree-compile` (g165e12a) used for all three vmfbs (standing requirement holds).
- **NEW finding:** the "3 graphs share one `gemma-gen.irpa`" assumption was invalid — each trace numbers its
  `model` externals independently (`t0`, `t10`, …), so the prefill/with_past graphs need archives written
  from THEIR traces (loud `NOT_FOUND … key 'tN'` otherwise). Fixed: `exportPrefill`/`exportWithPast` write
  `gemma-prefill.safetensors`/`gemma-with-past.safetensors`, `compile-gemma.sh` converts per-graph irpas,
  `GemmaKvDecoder` binds them.
- **Success gate MET:** oracle reproduced token-for-token; `PERF-LOGBOOK.md` rows: 2139 ms/token (steady
  ~1740) vs 4419 same-day re-decode. `GEMMA_KV` flipped to default-on in `Pipeline.kt` (#249).

## Open — int8 on-board
- Numeric quality of per-row int8 from Q5_K (oracle check), decode speed, and the RAM claim (831→~415 MiB on
  the 1.9 GB board) are all on-device.

## Open — release + demo bump
- Merge PR #245; cut a SKaiNET-transformers release (0.36.2 / 0.37) shipping KV/int8.
- Bump the demo's `skainetTransformers` to it; the demo's `compile-gemma.sh` `GEMMA_KV=1` / `GEMMA_QUANT=int8`
  path already consumes it. Make `GEMMA_KV=1` default after board parity.
- **Test harness:** heavy full-model-trace tests OOM at the 4 GB default — run with `-PkgemmaTestMaxHeap=12g`
  (or bump the default upstream).

## Open — conformance
- Add `functiongemma-270m` + the KV-graph (`gemma_prefill` / `gemma_with_past`) rows to
  `skainet-iree-conformance` (plan item **C.1**).

## Open — DSL-philosophy debt (make it the SKaiNET way)
The graph *authoring* is DSL/DAG-native, but the graph *post-processing* is currently **MLIR-text regex
rewrites**. Convert each to a proper StableHLO/DAG transform:
1. **bf16 weight emission** (`rewriteGlobalsToBf16`, regex on the MLIR string) → a DAG/StableHLO dtype transform
   (leverage the existing `DtypeForwardPropagationPass`).
2. **int8 quant** (`rewriteGlobalsToInt8` injecting dequant as text) → a real graph quantization pass; keep the
   host per-row quant writer, but emit the `i8→f32 × scale` dequant as graph nodes.
3. **Dynamic KV dims** (sentinel-prime `7919` → `x?x` string relax — the most fragile; a magic prime that must
   never collide with a real dim/SSA id) → proper dynamic-shape tracing/inference.
4. **`forwardWithPast` `attnWithPast`** (hand-wired single-token attention, real nodes but not module-native) →
   an MHA-with-past module forward.
5. **`refsFor`** (positional sub-module resolution, brittle vs `HybridTransformerBlock`'s SwiGLU-typed fields) →
   typed field access.

## Not blocking
- PLE models are unsupported by the KV path (`require(ple == null)`) — fine for FunctionGemma-270M (no PLE).
