#!/usr/bin/env bash
# Dockerized CPU compile for the SL2610 board: StableHLO .mlir -> aarch64 llvm-cpu vmfb.
#
# This is the sibling of iree-compile-torq-docker.sh (which targets the Torq NPU). Here we build a
# *CPU* executable for the board's ARM cores — the shippable CPU floor. The catch (proven in Phase C):
#
#   * The board's torq runtimes ONLY accept the pinned compiler's executable format. A vmfb from the
#     stock `iree-cpu-toolchain:3.11.0` is REJECTED at load ("required module features [Ch] are not
#     available"). So the CPU vmfb MUST be built with the SAME pinned compiler (sl2610-iree:v2.0.0)
#     as the NPU path — just with the llvm-cpu backend + aarch64 target instead of the torq backend.
#   * The fork's llvm-cpu link step shells out to `ld`; the image ships `iree-lld`, so we drop in a
#     one-line `ld -> iree-lld -flavor gnu` shim on PATH before compiling.
#
# Run the resulting vmfb on the board's CANONICAL runtime (torq-toolchain.lock: torq2-stable):
#   RM=/home/root/torq2-stable/torq/_runtime_libs/torq-run-module
#   LD_LIBRARY_PATH=/home/root/torq2-stable/iree/_runtime_libs:/home/root/torq2-stable/torq/_runtime_libs \
#     $RM --device=local-task --module=X.vmfb --function=main --input=... --output=@out.bin
#
#   scripts/compile-cpu-arm64-docker.sh <in.mlir> <out.vmfb>
#
# Verified in Phase C: all five self-compiled DSL Moonshine v2 graphs (frontend, encoder, adapter,
# masked prefill, dynamic with_past) compile this way and run bit-exact on the board (cos ~1.0 vs
# host x86), decoding beckett.wav's first window token-for-token identical to ONNX ("Ever tried?").
set -euo pipefail

IN=${1:?usage: compile-cpu-arm64-docker.sh <in.mlir> <out.vmfb>}
OUT=${2:?usage: compile-cpu-arm64-docker.sh <in.mlir> <out.vmfb>}
IMAGE=${IREE_IMAGE:-sl2610-iree:v2.0.0}

# --- toolchain pin: refuse to compile with a compiler that isn't the canary-verified one ---
# (Identical guard to iree-compile-torq-docker.sh — a CPU vmfb from an unpinned compiler will not
# load on the board runtime just the same.) Override only with TORQ_ALLOW_UNPINNED=1.
LOCK="$(cd "$(dirname "$0")" && pwd)/torq-toolchain.lock"
if [ -f "$LOCK" ]; then
  PINNED=$(grep -oaE '^COMPILER_ID=.*' "$LOCK" | head -n1 | cut -d= -f2- | sed 's/[[:space:]]*#.*//;s/[[:space:]]*$//')
  if [ -n "$PINNED" ] && [ "$IMAGE" != "$PINNED" ]; then
    if [ "${TORQ_ALLOW_UNPINNED:-0}" = "1" ]; then
      echo "WARNING: IREE_IMAGE='$IMAGE' != pinned COMPILER_ID='$PINNED' (TORQ_ALLOW_UNPINNED=1) — vmfb may not load on the board." >&2
    else
      echo "error: IREE_IMAGE='$IMAGE' does not match the pinned COMPILER_ID='$PINNED' in $LOCK." >&2
      echo "       Compile with the pinned compiler, or set TORQ_ALLOW_UNPINNED=1 to override (unsafe)." >&2
      exit 1
    fi
  fi
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "error: image '$IMAGE' not found (see scripts/.docker/Dockerfile.iree)." >&2
  exit 1
fi

IN_DIR=$(cd "$(dirname "$IN")" && pwd);  IN_BASE=$(basename "$IN")
OUT_DIR=$(cd "$(dirname "$OUT")" && pwd); OUT_BASE=$(basename "$OUT")
TRIPLE=${TARGET_TRIPLE:-aarch64-unknown-linux-gnu}
CPUFEAT=${TARGET_CPU_FEATURES:-+neon}

docker run --rm --entrypoint /bin/sh \
  -v "$IN_DIR:/in:ro" \
  -v "$OUT_DIR:/out" \
  "$IMAGE" -c '
    set -e
    LLD=$(command -v iree-lld || echo /usr/local/bin/iree-lld)
    printf "#!/bin/sh\nexec %s -flavor gnu \"\$@\"\n" "$LLD" > /usr/local/bin/ld
    chmod +x /usr/local/bin/ld
    IREEC=$(command -v iree-compile || echo /usr/local/bin/iree-compile)
    "$IREEC" "/in/'"$IN_BASE"'" -o "/out/'"$OUT_BASE"'" \
      --iree-input-type=stablehlo \
      --iree-hal-target-device=local \
      --iree-hal-local-target-device-backends=llvm-cpu \
      --iree-llvmcpu-target-triple='"$TRIPLE"' \
      --iree-llvmcpu-target-cpu-features='"$CPUFEAT"'
  '
echo "[cpu-arm64] $OUT"
