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
# TARGET:
#   host  (default) — x64 llvm-cpu via the stock IREE docker image; validates the pipeline.
#   board           — aarch64+NEON llvm-cpu via the g165 Torq-fork iree-compile (TORQ_PKG). REQUIRED
#                     for the SL2610: the board's iree-run-module rejects the "Ch" bytecode feature
#                     stock IREE 3.x emits (verified). Produces a board-runnable gemma-gen.vmfb.
# TORQ_PKG: g165 torq_compiler package dir (default /home/miso/projects/coral/build-mlir/torqpkg).
# IREE_IMAGE: stock iree docker image for the host target (default iree-cpu-toolchain:3.11.0).
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
docker run --rm --user "$(id -u):$(id -g)" -v "$MLIR:/work" "$IREE_IMAGE" \
    iree-convert-parameters --parameters=model=/work/gemma.safetensors --output=/work/gemma-gen.irpa

echo ">> [3/3] iree-compile (llvm-cpu, $TARGET) -> gemma-gen.vmfb"
if [ "$TARGET" = board ]; then
    # g165 Torq-fork compiler (local, not docker): aarch64+NEON, board-compatible bytecode.
    TORQ_PKG="${TORQ_PKG:-/home/miso/projects/coral/build-mlir/torqpkg}"
    MLIRLIBS="$TORQ_PKG/iree/compiler/_mlir_libs"
    [ -x "$MLIRLIBS/iree-compile" ] || { echo "error: g165 iree-compile not at $MLIRLIBS (set TORQ_PKG)" >&2; exit 1; }
    SHIM="$(mktemp -d)"; printf '#!/usr/bin/env bash\nexec "%s/iree-lld" -flavor gnu "$@"\n' "$MLIRLIBS" > "$SHIM/ld"; chmod +x "$SHIM/ld"
    PATH="$SHIM:$PATH" LD_LIBRARY_PATH="$MLIRLIBS" "$MLIRLIBS/iree-compile" \
        --iree-input-type=stablehlo \
        --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
        --iree-llvmcpu-target-triple=aarch64-unknown-linux-gnu --iree-llvmcpu-target-cpu-features=+neon \
        "$MLIR/gemma-gen.mlir" -o "$MLIR/gemma-gen.vmfb"
    rm -rf "$SHIM"
else
    docker run --rm --user "$(id -u):$(id -g)" -v "$MLIR:/work" "$IREE_IMAGE" \
        iree-compile /work/gemma-gen.mlir --iree-hal-target-device=local \
        --iree-hal-local-target-device-backends=llvm-cpu -o /work/gemma-gen.vmfb
fi

echo ">> done: $MLIR/gemma-gen.{vmfb,irpa}"
echo "   deploy both to /home/root/ireetest/ on the board — the demo's GemmaDecoder loads them."
echo "   verified: 'turn the light on' -> [262146,236769,3255,718,498,1373,262152,106] = <tool_0>(state=\"on\")<end>"
