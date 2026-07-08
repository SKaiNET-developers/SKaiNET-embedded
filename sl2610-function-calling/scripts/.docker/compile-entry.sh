#!/usr/bin/env bash
# In-container entrypoint: iree-compile a StableHLO .mlir for the SL2610 Torq NPU.
#   compile-entry.sh <in.mlir> <out.vmfb>
# Mounted by iree-compile-torq-docker.sh with /work as the shared dir.
set -euo pipefail

IN=${1:?usage: <in.mlir> <out.vmfb>}
OUT=${2:?usage: <in.mlir> <out.vmfb>}
LLD="$MLIRLIBS/iree-lld"

# ld -> iree-lld shim (Torq vmfb links aarch64 host code; system ld is x86 → EM:183).
SHIM=$(mktemp -d)
cat > "$SHIM/ld" <<EOF
#!/usr/bin/env bash
exec "$LLD" -flavor gnu "\$@"
EOF
chmod +x "$SHIM/ld"

# --torq-disable-slices is OPT-IN (TORQ_DISABLE_SLICES=1). It disables the linalg
# slicing that tiles the attention softmax to fit the CSS stack, so on the ENCODER it
# causes "CSS program allocations of 1328 bytes exceed maximum CSS stack size" (verified:
# enc6.mlir compiles WITHOUT it → 277627 bytes, FAILS with it). It was originally added
# for a decoder LayerNorm-NaN fix, so keep it available but default OFF.
EXTRA=()
if [ "${TORQ_DISABLE_SLICES:-0}" = "1" ]; then EXTRA+=(--torq-disable-slices); fi

PATH="$SHIM:$PATH" LD_LIBRARY_PATH="$MLIRLIBS:${LD_LIBRARY_PATH:-}" \
  "$MLIRLIBS/iree-compile" \
    --iree-input-type=stablehlo \
    --iree-hal-target-device=torq --torq-hw=SL2610 \
    "${EXTRA[@]}" \
    --torq-fallback-f32-to-host \
    "$IN" -o "$OUT"

rm -rf "$SHIM"

# Provenance sidecar: stamp which compiler + flags produced this vmfb, so a deploy can be traced
# to a toolchain and matched against the board runtime (see scripts/torq-toolchain.lock).
CVER="$(LD_LIBRARY_PATH="$MLIRLIBS" "$MLIRLIBS/iree-compile" --version 2>/dev/null | tr '\n' ' ' | sed 's/"/\\"/g')"
cat > "$OUT.toolchain.json" <<JSON
{
  "compiler_id": "${COMPILER_ID:-unknown}",
  "compiler_version": "$CVER",
  "flags": "--iree-hal-target-device=torq --torq-hw=SL2610 ${EXTRA[*]:-} --torq-fallback-f32-to-host",
  "torq_disable_slices": ${TORQ_DISABLE_SLICES:-0},
  "built": "$(date -u +%FT%TZ)"
}
JSON
echo "[torq-docker] wrote $OUT ($(stat -c %s "$OUT") bytes) + $OUT.toolchain.json"
