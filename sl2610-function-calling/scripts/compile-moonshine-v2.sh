#!/usr/bin/env bash
# Build the Moonshine **v2** streaming encoder + adapter vmfbs from the SKaiNET NN DSL.
#
#   MOONSHINE_V2_CHECKPOINT=baked SKAINET_TRANSFORMERS=../../SKaiNET-transformers \
#       scripts/compile-moonshine-v2.sh cpu            # host/aarch64 llvm-cpu
#
# `baked/` is the per-tensor f32 .bin dir from scripts/convert_moonshine_v2_weights.py (encoder + adapter).
# Steps: gradle MoonshineV2EncoderBakeTest (DSL -> StableHLO with the baked weights folded to constants) ->
# rename the entry to @main (torq-run-module / the demo runtime invoke function "main") -> iree-compile per
# backend. The encoder graph is a fixed CHUNK (ENC_FRAMES, default 64); the adapter takes (positions, memory).
#
# CPU is the shippable floor (proven: both compile + run on iree-cpu-toolchain:3.11.0). Torq/NPU tiling of the
# bounded window is a follow-up (docs/MOONSHINE-V2-STREAMING.md). Numeric parity vs the ONNX reference (onnxrt)
# is tracked separately.
set -euo pipefail
BACKEND="${1:?usage: compile-moonshine-v2.sh <cpu>}"
: "${MOONSHINE_V2_CHECKPOINT:?set MOONSHINE_V2_CHECKPOINT to the per-tensor .bin dir (convert_moonshine_v2_weights.py)}"
: "${SKAINET_TRANSFORMERS:?set SKAINET_TRANSFORMERS to the SKaiNET-transformers repo (has the v2 DSL + bake test)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FRAMES="${ENC_FRAMES:-64}"
IREE_IMAGE="${IREE_CPU_IMAGE:-iree-cpu-toolchain:3.11.0}"
MLIR="$ROOT/build/mlir/moonshine-v2"
mkdir -p "$MLIR"

echo ">> [1/3] DSL -> StableHLO (encoder + adapter, baked weight constants) via SKaiNET-transformers"
( cd "$SKAINET_TRANSFORMERS" \
  && MOONSHINE_V2_CHECKPOINT="$MOONSHINE_V2_CHECKPOINT" MOONSHINE_V2_MLIR_OUT="$MLIR" ENC_FRAMES="$FRAMES" \
     ./gradlew :llm-inference:moonshine:jvmTest --tests '*MoonshineV2EncoderBakeTest*' --rerun-tasks )

echo ">> [2/3] rename entry -> @main"
sed 's/@moonshine_v2_encoder/@main/g' "$MLIR/moonshine-v2-encoder.mlir" > "$MLIR/encoder.main.mlir"
sed 's/@moonshine_v2_adapter/@main/g' "$MLIR/moonshine-v2-adapter.mlir" > "$MLIR/adapter.main.mlir"

echo ">> [3/3] iree-compile ($BACKEND) via $IREE_IMAGE"
case "$BACKEND" in
  cpu)
    EXTRA=""
    [ "${TARGET:-}" = "aarch64" ] && EXTRA="--iree-llvmcpu-target-triple=aarch64-unknown-linux-gnu --iree-llvmcpu-target-cpu-features=+neon"
    docker run --rm -v "$MLIR":/work "$IREE_IMAGE" bash -lc "
      set -e
      for m in encoder adapter; do
        echo '>> iree-compile '\$m
        iree-compile /work/\$m.main.mlir --iree-hal-target-device=local \
          --iree-hal-local-target-device-backends=llvm-cpu $EXTRA -o /work/moonshine-v2-\$m-cpu.vmfb
        echo '>> wrote moonshine-v2-'\$m'-cpu.vmfb'
      done
    "
    ;;
  *) echo "backend must be cpu (torq/NPU is a follow-up)" >&2; exit 2 ;;
esac
echo ">> done. vmfbs in $MLIR/ (moonshine-v2-encoder-cpu.vmfb, moonshine-v2-adapter-cpu.vmfb)"
