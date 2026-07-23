# FunctionGemma KV-cache + int8 — landed status & open issues

The KV-cache 2-graph decode + int8 quant is the fix for FunctionGemma's ~6000 ms/token re-decode
(`PERF-LOGBOOK.md` baseline) — plan item **P4**. This tracks what's **landed** and every **board / release /
philosophy** step still required.

## Landed (2026-07-23)
- **SKaiNET-transformers PR #245** (`feat/functiongemma-kv-int8` → `develop` 0.36.1): rescued the work from
  uncommitted WIP, forward-ported onto develop, **CPU-verified**. All three tests pass with the Q5_K_M GGUF:
  `FunctionGemmaWithPastCpuTest` (token-for-token `[262146,236769,3255,718,498,1373,262152,106]`),
  `FunctionGemmaWithPastMlirDumpTest` (dynamic `x?x256` relax, no `x0x256`/sentinel leak, prefill `x16x256`),
  `FunctionGemmaInt8QuantTest` (`i8Globals==scaleGlobals`, ~half archive).
- The decode graphs are **DSL/DAG-authored** (`GemmaModel.forwardPrefill`/`forwardWithPast` compose module
  forwards; `RoPE.buildSplitHalfCosSin` gives runtime-position cos/sin).

## Open — board verification (needs the SL2610 + Torq-fork toolchain)
Runbook: `docs/GEMMA-KV-BOARD-LOOP.md`. On the **first** board run, confirm (each is a silent-corruption trap):
1. **Per-block K-vs-V output order** — `GemmaKvDecoder.kFirstInOutput=false` ((V,K) per return-SSA analysis);
   flip to `true` if the first run produces garbage tokens.
2. **`--output=@file` format** — `IreeRuntime.invokeFiles` assumes RAW bytes; if the board writes NumPy, strip
   the `.npy` header on read.
3. **`gemma_with_past` input arg order** — trace-derived; confirm against the compiled vmfb.
4. **Dynamic-concat** — the sentinel-prime `7919`→`x?x` relax leaves real dynamic-shape inference to
   `iree-compile`; confirm the `with_past` vmfb accepts `x?x256` (else fall back to fixed-pad+mask).
5. **`--task_topology_group_count`** — confirm the board `iree-run-module` accepts the flag (gated to revert).
6. Standing: vmfbs MUST be built with the **Torq-fork `iree-compile` (g165e12a)** (stock IREE 3.x "Ch" bytecode
   is rejected).
- **Success gate:** the KV loop reproduces the oracle and `PERF-LOGBOOK.md` shows the O(seq²)→O(seq) collapse.
  Then make `GEMMA_KV=1` the default in `Pipeline.kt`.

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
