#!/usr/bin/env bash
# Compile a StableHLO .mlir to an IREE CPU vmfb and (optionally) run it, via the
# dockerized IREE toolchain (image coral-iree-toolchain, repo mounted at /work).
#
#   scripts/iree-compile-cpu.sh <file.mlir> [--run <iree-run-module args...>]
#   scripts/iree-compile-cpu.sh build/mlir/addrelu.mlir --run \
#       --input="1x4xf32=1 2 3 4" --input="1x4xf32=-2 -2 -2 1"
#
# Host x64 vmfb by default. For the aarch64 board, set TARGET=aarch64 to add
# --iree-llvmcpu-target-triple=aarch64-unknown-linux-gnu --iree-llvmcpu-target-cpu-features=+neon
set -euo pipefail

MLIR_HOST=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); shift
ROOT=$(cd "$(dirname "$0")/../.." && pwd)               # coral workspace root
REL=${MLIR_HOST#"$ROOT"/}                                # path relative to /work
VMFB_REL="${REL%.mlir}_cpu.vmfb"
COMPOSE_DIR="$ROOT/docker/iree-toolchain"

EXTRA=""
[ "${TARGET:-}" = "aarch64" ] && EXTRA="--iree-llvmcpu-target-triple=aarch64-unknown-linux-gnu --iree-llvmcpu-target-cpu-features=+neon"

RUN=0; RUN_ARGS=()
if [ "${1:-}" = "--run" ]; then RUN=1; shift; RUN_ARGS=("$@"); fi

cd "$COMPOSE_DIR"
docker compose run --rm iree bash -lc "
set -e
echo '>> iree-compile (llvm-cpu ${TARGET:-host}) /work/$REL'
iree-compile /work/$REL --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu $EXTRA -o /work/$VMFB_REL
echo '>> wrote /work/$VMFB_REL'
$( [ $RUN -eq 1 ] && echo "iree-run-module --device=local-task --module=/work/$VMFB_REL ${RUN_ARGS[*]}" )
"
