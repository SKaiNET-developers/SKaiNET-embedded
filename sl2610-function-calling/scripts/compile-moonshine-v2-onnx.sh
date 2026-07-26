#!/usr/bin/env bash
# Compile the Moonshine **v2** frontend + decode graphs to IREE CPU vmfbs FROM THE VENDOR ONNX.
#
#   MOONSHINE_V2_ONNX=<dir with frontend.onnx,cross_kv.onnx,decoder_kv.onnx> \
#       scripts/compile-moonshine-v2-onnx.sh
#
# PROVENANCE: unlike the encoder + adapter (authored in the SKaiNET NN DSL and self-compiled by
# compile-moonshine-v2.sh), these three graphs are compiled from the **vendor float ONNX**
# (download.moonshine.ai/model/tiny-streaming-en/float/). This is the fast path to a full runnable v2
# pipeline (frontend -> encoder -> adapter -> cross_kv -> decoder_kv) entirely on IREE — no onnxruntime, no
# vendor runtime — but it is NOT yet the "all DSL-authored" north star. DSL-authoring the frontend + decoder
# (following v1's DSL decoder) is the principled follow-up; these vmfbs unblock the streaming runtime now.
#
# Steps: iree-import-onnx (ONNX -> torch-onnx MLIR) -> iree-compile (llvm-cpu). decoder_kv additionally needs
# its opset-17 `onnx.LayerNormalization` decomposed first (IREE 3.11.0 can't legalize it) via
# onnx_decompose_layernorm.py. Tooling runs under `uv` (onnx + iree-base-compiler==3.11.0, matching the pinned
# board toolchain). CPU is the shippable floor; Torq/NPU is a follow-up.
set -euo pipefail
: "${MOONSHINE_V2_ONNX:?set MOONSHINE_V2_ONNX to a dir with frontend.onnx, cross_kv.onnx, decoder_kv.onnx}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$ROOT/scripts"
OUT="$ROOT/build/mlir/moonshine-v2"
mkdir -p "$OUT"

# uv env with onnx + the IREE importer/compiler (3.11.0 == the pinned board toolchain).
UV="uv run"
command -v uv >/dev/null || { echo "need uv (https://docs.astral.sh/uv/)"; exit 1; }
$UV python -c "import onnx, iree.compiler" 2>/dev/null || uv add onnx iree-base-compiler==3.11.0

compile_one() { # <name> <src.onnx>
  local name="$1" src="$2"
  echo ">> [$name] iree-import-onnx"
  $UV iree-import-onnx "$src" -o "$OUT/$name.mlir"
  echo ">> [$name] iree-compile (llvm-cpu)"
  local extra=""
  [ "${TARGET:-}" = "aarch64" ] && extra="--iree-llvmcpu-target-triple=aarch64-unknown-linux-gnu --iree-llvmcpu-target-cpu-features=+neon"
  $UV iree-compile "$OUT/$name.mlir" --iree-input-type=onnx \
    --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu $extra \
    -o "$OUT/moonshine-v2-$name-cpu.vmfb"
  echo ">> [$name] wrote moonshine-v2-$name-cpu.vmfb"
}

compile_one frontend "$MOONSHINE_V2_ONNX/frontend.onnx"
compile_one cross_kv "$MOONSHINE_V2_ONNX/cross_kv.onnx"

# decoder_kv: decompose onnx.LayerNormalization first (IREE 3.11.0 legalization gap).
echo ">> [decoder_kv] decompose LayerNormalization"
$UV python "$SCRIPTS/onnx_decompose_layernorm.py" "$MOONSHINE_V2_ONNX/decoder_kv.onnx" "$OUT/decoder_kv_lnexp.onnx"
compile_one decoder_kv "$OUT/decoder_kv_lnexp.onnx"

echo ">> done. entry function is @main_graph. vmfbs in $OUT/:"
ls -la "$OUT"/moonshine-v2-{frontend,cross_kv,decoder_kv}-cpu.vmfb 2>/dev/null | awk '{print "   ", $5, $9}'
