#!/usr/bin/env bash
# Self-compile FunctionGemma from the SKaiNET DSL — ONE command, NO Python.
#
#   GEMMA_GGUF=/path/functiongemma-physical-ai-v10-Q5_K_M.gguf \
#     scripts/compile-gemma.sh [host|board]
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
#
# GEMMA_KV=1 — ALSO build the KV-cache 2-graph decode (perf program Phase 2): gemma-prefill.vmfb +
#   gemma-with-past.vmfb (the latter with a DYNAMIC `1x1x?x256` self-cache — one vmfb serves every
#   position). All three graphs share the one gemma-gen.irpa (same "model" external weights). The
#   board's GemmaKvDecoder drives prefill-once + with_past-loop. NOT board-verified yet — opt-in.
#
# GEMMA_QUANT=int8 — quantize the 2D matmul weights to per-row int8 in the compiled graph (Phase 5):
#   ~half the irpa (831->~418 MiB — a real RAM win on the 1.9 GB board) + half the weight-read traffic.
#   Norms stay bf16; dequant is in-graph (convert i8->f32 x broadcast(scale)). NOT board-verified —
#   opt-in; on-board must confirm iree accepts it + the per-row-int8 numeric quality (oracle check).
#
# Config comes from demo.env (see demo.env.example) or inline env:
#   TORQ_PKG   g165 torq_compiler package dir (default $ROOT/.toolchain/torqpkg; bootstrap.sh fetches it).
#   IREE_IMAGE stock iree docker image for the host target (default iree-cpu-toolchain:3.11.0).
#   GEMMA_GGUF path to the FunctionGemma Q5_K_M .gguf (your own, or your finetuned one).
#
# Replaces the old test+Python chain (RealGemmaBakeIrpaTest + add_argmax_perpos.py + make_f16.py):
#   - the argmax tail is now the DSL `argMax` op (in the emitted StableHLO), and
#   - weights are emitted as bf16 externals directly by the exporter.
set -euo pipefail

TARGET="${1:-host}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"                 # sl2610-function-calling
[ -f "$ROOT/demo.env" ] && . "$ROOT/demo.env"           # local config (GEMMA_GGUF, TORQ_PKG, …); inline env still wins
: "${GEMMA_GGUF:?set GEMMA_GGUF (in demo.env or inline) to the FunctionGemma Q5_K_M .gguf}"
TF="$(cd "$ROOT/../../SKaiNET-transformers" && pwd)"     # SKaiNET-transformers (kgemma exporter)
MLIR="$ROOT/build/mlir"; mkdir -p "$MLIR"
IREE_IMAGE="${IREE_IMAGE:-iree-cpu-toolchain:3.11.0}"
GEMMA_KV="${GEMMA_KV:-0}"

# iree-compile one StableHLO .mlir -> .vmfb for the selected TARGET (dynamic `?` dims are fine on llvm-cpu).
compile_one() {
    local mlir_in="$1" vmfb_out="$2"
    if [ "$TARGET" = board ]; then
        # g165 Torq-fork compiler (local, not docker): aarch64+NEON, board-compatible bytecode.
        local TORQ_PKG="${TORQ_PKG:-$ROOT/.toolchain/torqpkg}"
        local MLIRLIBS="$TORQ_PKG/iree/compiler/_mlir_libs"
        [ -x "$MLIRLIBS/iree-compile" ] || { echo "error: g165 iree-compile not at $MLIRLIBS (set TORQ_PKG)" >&2; exit 1; }
        local SHIM; SHIM="$(mktemp -d)"
        printf '#!/usr/bin/env bash\nexec "%s/iree-lld" -flavor gnu "$@"\n' "$MLIRLIBS" > "$SHIM/ld"; chmod +x "$SHIM/ld"
        PATH="$SHIM:$PATH" LD_LIBRARY_PATH="$MLIRLIBS" "$MLIRLIBS/iree-compile" \
            --iree-input-type=stablehlo \
            --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
            --iree-llvmcpu-target-triple=aarch64-unknown-linux-gnu --iree-llvmcpu-target-cpu-features=+neon \
            "$mlir_in" -o "$vmfb_out"
        rm -rf "$SHIM"
    else
        docker run --rm --user "$(id -u):$(id -g)" -v "$MLIR:/work" "$IREE_IMAGE" \
            iree-compile "/work/$(basename "$mlir_in")" --iree-input-type=stablehlo --iree-hal-target-device=local \
            --iree-hal-local-target-device-backends=llvm-cpu -o "/work/$(basename "$vmfb_out")"
    fi
}

# GEMMA_GRAPH=all also emits gemma-prefill.mlir + gemma-with-past.mlir (share the redecode safetensors).
GRAPHS="redecode"; [ "$GEMMA_KV" = 1 ] && GRAPHS="all"
echo ">> [1/3] DSL -> gemma-gen.mlir + bf16 gemma.safetensors${GEMMA_KV:+ (+ KV graphs)}  (kgemma FunctionGemmaExport)"
( cd "$TF" && GEMMA_GGUF="$GEMMA_GGUF" GEMMA_OUT_DIR="$MLIR" GEMMA_GRAPH="$GRAPHS" GEMMA_QUANT="${GEMMA_QUANT:-}" \
    ./gradlew -PuseLocalSkainet=true :llm-runtime:kgemma:exportFunctionGemma -q )

echo ">> [2/3] gemma.safetensors -> gemma-gen.irpa  (iree-convert-parameters; shared by all graphs)"
docker run --rm --user "$(id -u):$(id -g)" -v "$MLIR:/work" "$IREE_IMAGE" \
    iree-convert-parameters --parameters=model=/work/gemma.safetensors --output=/work/gemma-gen.irpa

echo ">> [3/3] iree-compile (llvm-cpu, $TARGET) -> vmfb(s)"
compile_one "$MLIR/gemma-gen.mlir" "$MLIR/gemma-gen.vmfb"
if [ "$GEMMA_KV" = 1 ]; then
    compile_one "$MLIR/gemma-prefill.mlir"   "$MLIR/gemma-prefill.vmfb"
    compile_one "$MLIR/gemma-with-past.mlir" "$MLIR/gemma-with-past.vmfb"
fi

echo ">> done:"
echo "   $MLIR/gemma-gen.{vmfb,irpa}   (re-decode — the shipping path)"
[ "$GEMMA_KV" = 1 ] && echo "   $MLIR/gemma-prefill.vmfb + gemma-with-past.vmfb  (KV-cache 2-graph decode, share gemma-gen.irpa)"
echo "   deploy to /home/root/ireetest/ on the board — the demo's GemmaDecoder / GemmaKvDecoder loads them."
echo "   verified: 'turn the light on' -> [262146,236769,3255,718,498,1373,262152,106] = <tool_0>(state=\"on\")<end>"
