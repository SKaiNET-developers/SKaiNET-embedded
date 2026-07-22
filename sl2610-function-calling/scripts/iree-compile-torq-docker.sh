#!/usr/bin/env bash
# Dockerized Torq compile: StableHLO .mlir -> SL2610 vmfb, no host toolchain setup.
# The compiler runs inside the sl2610-iree image (scripts/.docker/Dockerfile.iree), so
# the only host requirement is Docker. Mirrors the Antora docs' dockerized-tooling pattern.
#
#   scripts/iree-compile-torq-docker.sh <in.mlir> <out.vmfb>
#
# One-time image build (needs the private torq_compiler wheel — not on PyPI):
#   cp /path/to/torq_compiler-*.whl scripts/.docker/
#   docker build -f scripts/.docker/Dockerfile.iree -t sl2610-iree:local scripts/.docker
#
# Example end-to-end:
#   ./gradlew moonshineEncoderMlir                                   # DSL -> mlir
#   scripts/iree-compile-torq-docker.sh build/mlir/moonshine-encoder.mlir build/mlir/encoder.vmfb
#   scp build/mlir/encoder.vmfb root@BOARD:/home/root/moon/encoder-selfcompiled.vmfb
#   # then run the board binary with MOONSHINE_ENCODER_VMFB=/home/root/moon/encoder-selfcompiled.vmfb
set -euo pipefail

IN=${1:?usage: iree-compile-torq-docker.sh <in.mlir> <out.vmfb>}
OUT=${2:?usage: iree-compile-torq-docker.sh <in.mlir> <out.vmfb>}
IMAGE=${IREE_IMAGE:-sl2610-iree:v2.0.0}

# --- toolchain pin: refuse to compile with a compiler that isn't the canary-verified one ---
# The pinned compiler lives in scripts/torq-toolchain.lock (COMPILER_ID). A vmfb built with any
# other compiler may silently mismatch the board runtime's executable format. Override only with
# TORQ_ALLOW_UNPINNED=1 (prints a loud warning). Update the pin via docs/torq-toolchain-update.md.
LOCK="$(cd "$(dirname "$0")" && pwd)/torq-toolchain.lock"
if [ -f "$LOCK" ]; then
  PINNED=$(grep -oaE '^COMPILER_ID=.*' "$LOCK" | head -n1 | cut -d= -f2- | sed 's/[[:space:]]*#.*//;s/[[:space:]]*$//')
  if [ -n "$PINNED" ] && [ "$IMAGE" != "$PINNED" ]; then
    if [ "${TORQ_ALLOW_UNPINNED:-0}" = "1" ]; then
      echo "WARNING: IREE_IMAGE='$IMAGE' != pinned COMPILER_ID='$PINNED' (TORQ_ALLOW_UNPINNED=1) — vmfb may not run on the board." >&2
    else
      echo "error: IREE_IMAGE='$IMAGE' does not match the pinned COMPILER_ID='$PINNED' in $LOCK." >&2
      echo "       Compile with the pinned compiler, or set TORQ_ALLOW_UNPINNED=1 to override (unsafe)." >&2
      exit 1
    fi
  fi
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "error: image '$IMAGE' not found. Build it first:" >&2
  echo "  cp /path/to/torq_compiler-*.whl scripts/.docker/" >&2
  echo "  docker build -f scripts/.docker/Dockerfile.iree -t $IMAGE scripts/.docker" >&2
  exit 1
fi

IN_DIR=$(cd "$(dirname "$IN")" && pwd);  IN_BASE=$(basename "$IN")
OUT_DIR=$(cd "$(dirname "$OUT")" && pwd); OUT_BASE=$(basename "$OUT")

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e "TORQ_DISABLE_SLICES=${TORQ_DISABLE_SLICES:-0}" \
  -e "COMPILER_ID=$IMAGE" \
  -v "$IN_DIR:/in:ro" \
  -v "$OUT_DIR:/out" \
  "$IMAGE" \
  "/in/$IN_BASE" "/out/$OUT_BASE"

echo "[torq-docker] $OUT"
