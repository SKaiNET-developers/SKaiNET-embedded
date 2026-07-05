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

PATH="$SHIM:$PATH" LD_LIBRARY_PATH="$MLIRLIBS" \
  "$MLIRLIBS/iree-compile" \
    --iree-input-type=stablehlo \
    --iree-hal-target-device=torq --torq-hw=SL2610 \
    --torq-disable-slices \
    "$IN" -o "$OUT"

rm -rf "$SHIM"
echo "[torq-docker] wrote $OUT ($(stat -c %s "$OUT") bytes)"
