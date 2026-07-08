#!/usr/bin/env bash
# Self-compile FunctionGemma from the SKaiNET DSL — ONE command, NO Python.
#
#   GEMMA_GGUF=/path/functiongemma-physical-ai-v10-Q5_K_M.gguf \
#     scripts/compile-gemma.sh [host|aarch64]
#
#   DSL gemmaNetwork() (argMax tail)  ->  gemma-gen.mlir + bf16 gemma.safetensors   (kgemma exporter)
#                                     ->  gemma-gen.irpa                            (iree-convert-parameters)
#                                     ->  gemma-gen.vmfb                            (iree-compile llvm-cpu)
#
# TARGET: host (default, x64 — validates the pipeline) or aarch64 (SL2610 board, +NEON).
# IREE_IMAGE: iree-cpu-toolchain:3.11.0 (default). For the board runtime use the Torq-fork
#   iree image (the board's iree-run-module rejects a bytecode feature stock IREE 3.x emits).
#
# Replaces the old test+Python chain (RealGemmaBakeIrpaTest + add_argmax_perpos.py + make_f16.py):
#   - the argmax tail is now the DSL `argMax` op (in the emitted StableHLO), and
#   - weights are emitted as bf16 externals directly by the exporter.
set -euo pipefail

TARGET="${1:-host}"
: "${GEMMA_GGUF:?set GEMMA_GGUF to the FunctionGemma Q5_K_M .gguf}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"                 # sl2610-function-calling
TF="$(cd "$ROOT/../../SKaiNET-transformers" && pwd)"     # SKaiNET-transformers (kgemma exporter)
MLIR="$ROOT/build/mlir"; mkdir -p "$MLIR"
IREE_IMAGE="${IREE_IMAGE:-iree-cpu-toolchain:3.11.0}"

echo ">> [1/3] DSL -> gemma-gen.mlir + bf16 gemma.safetensors  (kgemma FunctionGemmaExport)"
( cd "$TF" && GEMMA_GGUF="$GEMMA_GGUF" GEMMA_OUT_DIR="$MLIR" \
    ./gradlew -PuseLocalSkainet=true :llm-runtime:kgemma:exportFunctionGemma -q )

echo ">> [2/3] gemma.safetensors -> gemma-gen.irpa  (iree-convert-parameters)"
docker run --rm -v "$MLIR:/work" "$IREE_IMAGE" \
    iree-convert-parameters --parameters=model=/work/gemma.safetensors --output=/work/gemma-gen.irpa

echo ">> [3/3] iree-compile (llvm-cpu, $TARGET) -> gemma-gen.vmfb"
EXTRA=""
[ "$TARGET" = aarch64 ] && EXTRA="--iree-llvmcpu-target-triple=aarch64-unknown-linux-gnu --iree-llvmcpu-target-cpu-features=+neon"
docker run --rm -v "$MLIR:/work" "$IREE_IMAGE" \
    iree-compile /work/gemma-gen.mlir --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu $EXTRA -o /work/gemma-gen.vmfb

echo ">> done: $MLIR/gemma-gen.{vmfb,irpa}"
echo "   deploy both to the board (e.g. /home/root/ireetest/) — the demo's GemmaDecoder loads them."
