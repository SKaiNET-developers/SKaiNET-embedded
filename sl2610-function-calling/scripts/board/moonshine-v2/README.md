# Moonshine v2 — SL2610 board bring-up & verification (Phase C)

The fully-DSL, zero-vendor-neural-binary Moonshine v2 streaming ASR runs **end-to-end on the SL2610**,
decoding token-for-token identical to the ONNX reference. This directory is the board-verification
harness that proved it.

## The five self-compiled DSL vmfbs

| stage | entry symbol | board latency¹ |
|---|---|---|
| frontend | `main` | ~0.05 s |
| encoder | `main` | ~0.30 s |
| adapter | `main` | ~0.02 s |
| masked prefill (MAX_MEM=96) | `moonshine_v2_decoder_prefill` | ~0.32 s |
| dynamic-cache with_past (`?` self-cache) | `moonshine_v2_decoder_with_past` | — |

¹ single-thread `--device=local-task`, `torq2-stable` runtime. Per-window heavy compute ≈ 0.7 s, vs
the v1 re-decode's O(N²) growth that dominated the old ~95 s/turn.

## Toolchain (see `../../torq-toolchain.lock`)

- **Compile** every vmfb with the pinned `sl2610-iree:v2.0.0` via
  `scripts/compile-cpu-arm64-docker.sh <in.mlir> <out.vmfb>` (llvm-cpu, aarch64, `+neon`, ld→iree-lld
  shim). A stock `iree-cpu-toolchain:3.11.0` vmfb is **rejected** by the board runtime ("required
  module features [Ch] are not available").
- **Run** on the canonical `torq2-stable` runtime:
  ```
  RM=/home/root/torq2-stable/torq/_runtime_libs/torq-run-module
  LD_LIBRARY_PATH=/home/root/torq2-stable/iree/_runtime_libs:/home/root/torq2-stable/torq/_runtime_libs \
    $RM --device=local-task --module=X.vmfb --function=main --input=... --output=@out.bin
  ```

## Producing the decoder MLIRs

The two decoder graphs are exported by the SKaiNET-transformers bake test (needs the local dynamic-shape
core — `-PuseLocalSkainet`, branches `feat/dynamic-shapes-local` + `feat/decode-true-dynamic-flags`):

```
cd $SKAINET_TRANSFORMERS
MOONSHINE_V2_DEC_CHECKPOINT=<baked .bin dir> \
  MOONSHINE_V2_TRUE_DYNAMIC=1 MOONSHINE_V2_MAX_MEM=96 DEC_SEQ=1 \
  MOONSHINE_V2_DEC_PREFILL_OUT=pre_masked.mlir \
  MOONSHINE_V2_DEC_WITHPAST_OUT=wp_masked.mlir \
  ./gradlew :llm-inference:moonshine:jvmTest --tests '*MoonshineV2DecoderBakeTest*' --rerun-tasks -PuseLocalSkainet=true
```

`DEC_SEQ=1` → prefill takes a single BOS embed `[1,1,320]`. `MOONSHINE_V2_TRUE_DYNAMIC=1` → the
with_past self-cache is a real dynamic extent (`tensor<1x8x?x40>`). `MOONSHINE_V2_MAX_MEM=96` →
fixed-max-pad the cross memory to 96 with an additive cross-attention mask input.

## Verification scripts (host, `MOON_SCRATCH` = the scratch layout)

1. `golden_dump.py` — host x86 golden for frontend/encoder/adapter I/O.
2. `dec_golden.py` — host x86 golden for one prefill + one with_past step (all outputs).
3. `run_dec.sh` — run both decoder graphs once on the board (all outputs) for a stage-level bit-exact check.
4. `board_decode.py` — the E2E: greedy decode with prefill + growing with_past executing on the board,
   argmax/embed/RoPE on host. Prints the board token ids.

Phase-C result: every stage cos ≥ 0.9999999 vs host golden; `board_decode.py` → `[18274,1898,29973,2]`
== ONNX → **" Ever tried?"**.

## Remaining (productization, not correctness)
- Build the Kotlin/Native `MoonshineV2StreamingRunner` binary for a true wall-clock latency/turn
  (blocked by a pre-existing `GemmaKvDecoder` unresolved ref in `Pipeline.kt` → needs `-PuseLocalStack`).
- Rename decoder entry symbols → `@main` if the runner wiring expects it (board runs used `--function=`).
