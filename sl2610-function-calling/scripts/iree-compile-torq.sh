#!/usr/bin/env bash
# Compile a StableHLO .mlir for the SL2610 Torq NPU with the CURRENT toolchain
# (torq_compiler g165e12a wheel — matches the board's torq-runtime; the prebuilt
# Synaptics vmfbs are version-stale and no longer load).
#
#   scripts/iree-compile-torq.sh <in.mlir> <out.vmfb>
#
# Then on the board:  iree-run-module --device=torq --module=out.vmfb ...
#
# KEY FIXES baked in here:
#  1. Torq target = --iree-hal-target-device=torq --torq-hw=SL2610 (sets the
#     aarch64 host-code triple for the board automatically).
#  2. CROSS-LINKER: a Torq vmfb contains aarch64 "host code" linked via a SYSTEM
#     `ld` resolved on PATH. On an x86 host that's /usr/bin/ld (x86) -> fails with
#     "Relocations in generic ELF (EM: 183) ... wrong format". Fix: shim `ld` to
#     the bundled multi-target iree-lld (-flavor gnu) and put it FIRST on PATH.
#  3. NPU ops are bf16 conv/matmul — f32 elementwise does NOT run on the NPU
#     (returns zeros). Author NPU graphs in bf16.
set -euo pipefail

IN=${1:?usage: iree-compile-torq.sh <in.mlir> <out.vmfb>}
OUT=${2:?usage: iree-compile-torq.sh <in.mlir> <out.vmfb>}

# QUARANTINED — this is the g165e12a (dev, exec-format v1) compiler. It is NOT the pinned ASR
# toolchain: it emits a different executable format than our stable-v2 pin and it SIGSEGVs on
# 3+-matmul / softmax graphs (the encoder). It survives only for the separate FunctionGemma path
# whose board runtime matches v1. Do NOT compile ASR vmfbs with it — use
# scripts/iree-compile-torq-docker.sh (the pinned compiler in scripts/torq-toolchain.lock).
if [ "${TORQ_FUNCTIONGEMMA_OK:-0}" != "1" ]; then
  echo "error: iree-compile-torq.sh is quarantined (g165e12a — wrong exec-format for ASR). Use" >&2
  echo "       scripts/iree-compile-torq-docker.sh (pinned in scripts/torq-toolchain.lock)." >&2
  echo "       Override with TORQ_FUNCTIONGEMMA_OK=1 only for the FunctionGemma v1 path." >&2
  exit 1
fi

TORQ_PKG=${TORQ_PKG:-/home/miso/projects/coral/build-mlir/torqpkg}
PY=${PY:-/home/miso/.local/bin/python3.12}

MLIRLIBS="$TORQ_PKG/iree/compiler/_mlir_libs"
LLD="$MLIRLIBS/iree-lld"

# ld -> iree-lld shim
SHIM=$(mktemp -d)
cat > "$SHIM/ld" <<EOF
#!/usr/bin/env bash
exec "$LLD" -flavor gnu "\$@"
EOF
chmod +x "$SHIM/ld"

PATH="$SHIM:$PATH" LD_LIBRARY_PATH="$MLIRLIBS" \
  "$MLIRLIBS/iree-compile" \
    --iree-input-type=stablehlo \
    --iree-hal-target-device=torq --torq-hw=SL2610 \
    "$IN" -o "$OUT"

rm -rf "$SHIM"
echo "[torq] wrote $OUT ($(stat -c %s "$OUT") bytes) — run on board: iree-run-module --device=torq --module=$(basename "$OUT")"
