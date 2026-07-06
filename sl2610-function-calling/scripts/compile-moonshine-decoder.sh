#!/usr/bin/env bash
# Build both Moonshine decode-graph vmfbs (prefill + decoder_with_past) from the NN DSL.
#
#   DECODER_CHECKPOINT=weights scripts/compile-moonshine-decoder.sh cpu    # host/aarch64 llvm-cpu
#   DECODER_CHECKPOINT=weights scripts/compile-moonshine-decoder.sh torq   # SL2610 NPU (dockerized)
#
# `weights/` is the per-tensor HF .bin dir from scripts/convert_moonshine_weights.py (decoder tensors).
# Steps: gradle moonshineDecoderMlir (DSL -> StableHLO, baked constants) -> rename the entry to @main
# (torq-run-module / the demo runtime invoke function "main") -> iree-compile per backend.
#
# NOTE (open item, see .claude/plans/moonshine-demo-rewrite-3DEC-e.md): decoder_with_past is traced at
# a FIXED past length (DEC_PAST=1). One board vmfb over a GROWING self-cache needs a dynamic seq dim
# (`1x8x?x36`, add --iree flags here) or the vendor-style fixed-pad rework. Validate on CPU first.
set -euo pipefail
BACKEND="${1:?usage: compile-moonshine-decoder.sh <cpu|torq>}"
: "${DECODER_CHECKPOINT:?set DECODER_CHECKPOINT to the per-tensor HF .bin dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MLIR="$ROOT/build/mlir"

echo ">> [1/2] DSL -> StableHLO (both graphs, baked weight constants)"
( cd "$ROOT" && DECODER_CHECKPOINT="$DECODER_CHECKPOINT" ./gradlew moonshineDecoderMlir )

compile_one() { # <fn-suffix> <src.mlir>
  local suffix="$1" src="$2"
  local main="$MLIR/moonshine_decoder_${suffix}.main.mlir"
  # Entry rename: our export names the function moonshine_decoder_<suffix>; the runner wants @main.
  sed "s/@moonshine_decoder_${suffix}/@main/g" "$src" > "$main"
  echo ">> [2/2] iree-compile ($BACKEND) $(basename "$main")"
  case "$BACKEND" in
    cpu)  "$ROOT/scripts/iree-compile-cpu.sh" "$main" ;;                                  # -> *_cpu.vmfb
    torq) "$ROOT/scripts/iree-compile-torq-docker.sh" "$main" "$MLIR/decoder_${suffix}.vmfb" ;;
    *) echo "backend must be cpu|torq" >&2; exit 2 ;;
  esac
}

compile_one prefill   "$MLIR/moonshine-decoder.mlir"
compile_one with_past "$MLIR/moonshine-decoder-with-past.mlir"
echo ">> done. vmfbs in $MLIR/ (prefill -> decoder, with_past -> decoder_with_past)."
