#!/usr/bin/env bash
# Build the self-compiled DSL Moonshine **v2** decoder vmfbs (KV-cached prefill + with_past) — the replacement
# for the vendor ONNX cross_kv + decoder_kv.
#
#   MOONSHINE_V2_ONNX=<float-onnx dir> SKAINET_TRANSFORMERS=../../SKaiNET-transformers \
#       scripts/compile-moonshine-v2-decoder.sh
#
# Steps: bake_moonshine_v2_decoder.py (ONNX -> DSL .bin, incl. dec_embed.weight) -> gradle
# MoonshineV2DecoderBakeTest emits the two KV-cached graphs' StableHLO (weights folded to constants) ->
# rename entry @main -> iree-compile (llvm-cpu). The decoder is numerically validated (cos-sim 1.0 vs ONNX,
# validate_moonshine_v2_decoder.py); wired into MoonshineV2StreamingRunner's decode seam (per-layer K/V).
set -euo pipefail
: "${MOONSHINE_V2_ONNX:?set MOONSHINE_V2_ONNX to a dir with decoder_kv.onnx + cross_kv.onnx}"
: "${SKAINET_TRANSFORMERS:?set SKAINET_TRANSFORMERS to the SKaiNET-transformers repo (has the v2 decoder + bake test)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/build/mlir/moonshine-v2"
BAKED="$OUT/decoder-weights"
IREE_IMAGE="${IREE_CPU_IMAGE:-iree-cpu-toolchain:3.11.0}"
mkdir -p "$OUT"

echo ">> [1/3] bake decoder weights (incl. dec_embed.weight) from the float ONNX"
uv run python "$ROOT/scripts/bake_moonshine_v2_decoder.py" --onnx-dir "$MOONSHINE_V2_ONNX" --out "$BAKED"
cp "$BAKED/dec_embed.weight.bin" "$OUT/dec_embed.weight.bin"

echo ">> [2/3] DSL -> StableHLO (prefill + with_past, baked constants) via SKaiNET-transformers"
( cd "$SKAINET_TRANSFORMERS" && MOONSHINE_V2_DEC_CHECKPOINT="$BAKED" \
    MOONSHINE_V2_DEC_PREFILL_OUT="$OUT/v2-dec-prefill.mlir" \
    MOONSHINE_V2_DEC_WITHPAST_OUT="$OUT/v2-dec-withpast.mlir" \
    ./gradlew :llm-inference:moonshine:jvmTest --tests '*MoonshineV2DecoderBakeTest*' --rerun-tasks )

echo ">> [3/3] rename entry -> @main + iree-compile ($IREE_IMAGE)"
sed 's/@moonshine_v2_decoder_prefill/@main/g' "$OUT/v2-dec-prefill.mlir" > "$OUT/prefill.main.mlir"
sed 's/@moonshine_v2_decoder_with_past/@main/g' "$OUT/v2-dec-withpast.mlir" > "$OUT/withpast.main.mlir"
EXTRA=""
[ "${TARGET:-}" = "aarch64" ] && EXTRA="--iree-llvmcpu-target-triple=aarch64-unknown-linux-gnu --iree-llvmcpu-target-cpu-features=+neon"
docker run --rm -v "$OUT":/work "$IREE_IMAGE" bash -lc "
  set -e
  for m in prefill withpast; do
    iree-compile /work/\$m.main.mlir --iree-hal-target-device=local \
      --iree-hal-local-target-device-backends=llvm-cpu $EXTRA -o /work/moonshine-v2-dec-\$m-cpu.vmfb
    echo '>> wrote moonshine-v2-dec-'\$m'-cpu.vmfb'
  done
"
echo ">> done. vmfbs + dec_embed.weight.bin in $OUT/ (wire via MOONSHINE_V2_{PREFILL,WITHPAST}_VMFB + MOONSHINE_V2_EMBED)"
