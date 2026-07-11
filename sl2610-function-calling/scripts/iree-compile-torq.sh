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
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$ROOT/demo.env" ] && . "$ROOT/demo.env"   # local config (TORQ_PKG); inline env still wins
TORQ_PKG=${TORQ_PKG:-$ROOT/.toolchain/torqpkg}

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
