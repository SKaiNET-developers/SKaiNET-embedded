#!/usr/bin/env bash
# Compile the R1 smoke graphs with the pinned Torq-fork compiler image (sl2610-iree:v2.0.0,
# built from coraldevboard sl2610-function-calling scripts/.docker/Dockerfile.iree).
#   add_f32  -> llvm-cpu aarch64 (local-task)      add_bf16 -> torq (SL2610 NPU)
# Also compiles add_f32 with upstream IREE 3.11 to show the bytecode incompatibility.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd); mkdir -p "$D/out"
RUN=(docker run --rm --user "$(id -u):$(id -g)" -v "$D:/w" --entrypoint bash sl2610-iree:v2.0.0 -c)
M='$MLIRLIBS'
"${RUN[@]}" "$M/iree-compile --version"
"${RUN[@]}" "$M/iree-compile /w/mlir/add_f32.mlir --iree-input-type=stablehlo \
  --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=aarch64-unknown-linux-gnu --iree-llvmcpu-target-cpu=cortex-a55 \
  -o /w/out/add_f32.torqfork.cpu.vmfb"
docker run --rm --user "$(id -u):$(id -g)" -v "$D:/w" -w /w sl2610-iree:v2.0.0 \
  /w/mlir/add_bf16.mlir /w/out/add_bf16.torq.vmfb
docker run --rm --user "$(id -u):$(id -g)" -v "$D:/w" --entrypoint iree-compile skainet/iree-compiler:3.11.0 \
  /w/mlir/add_f32.mlir --iree-input-type=stablehlo \
  --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=aarch64-unknown-linux-gnu --iree-llvmcpu-target-cpu=cortex-a55 \
  -o /w/out/add_f32.up311.cpu.vmfb
ls -la "$D/out"
