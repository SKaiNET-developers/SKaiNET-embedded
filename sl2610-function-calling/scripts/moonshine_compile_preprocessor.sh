#!/usr/bin/env bash
# Compile the Moonshine ONNX audio preprocessor to an aarch64 llvm-cpu vmfb that
# the board's torq-run-module can load on --device=local-task. This is the one
# Moonshine stage that is not a prebuilt Torq vmfb (the encoder/decoder run on the
# NPU as shipped); the preprocessor ships only as preprocessor.onnx and the NPU
# path in MoonshineDecoder needs it as a CPU vmfb.
#
#   scripts/moonshine_compile_preprocessor.sh <preprocessor.onnx> <out.vmfb>
#
# TWO-STAGE, on purpose:
#  1. iree-import-onnx (stock IREE 3.11 toolchain, dockerized) : onnx -> torch-onnx MLIR.
#  2. iree-compile with the TORQ-FORK compiler                  : MLIR -> aarch64 llvm-cpu vmfb.
# Stage 2 must use the torq fork: the stock 3.11 compiler emits a "Ch" bytecode
# feature the board's (older) torq runtime rejects with INVALID_ARGUMENT. The torq
# fork's bytecode matches the board runtime — same reason scripts/iree-compile-torq.sh
# exists for the Torq NPU path.
set -euo pipefail

ONNX=${1:?usage: moonshine_compile_preprocessor.sh <preprocessor.onnx> <out.vmfb>}
OUT=${2:?usage: moonshine_compile_preprocessor.sh <preprocessor.onnx> <out.vmfb>}
IMPORT_IMAGE=${IMPORT_IMAGE:-iree-cpu-toolchain:3.11.0}
TORQ_PKG=${TORQ_PKG:-/home/miso/projects/coral/build-mlir/torqpkg}

WORK=$(mktemp -d)
cp "$ONNX" "$WORK/preprocessor.onnx"

echo ">> [1/2] iree-import-onnx (dockerized $IMPORT_IMAGE)"
docker run --rm -v "$WORK:/work" "$IMPORT_IMAGE" bash -lc '
  set -e
  pip install --quiet onnx 2>/dev/null || pip3 install --quiet onnx
  iree-import-onnx /work/preprocessor.onnx -o /work/preprocessor.mlir'

echo ">> [2/2] iree-compile (torq-fork, aarch64 llvm-cpu + neon)"
MLIRLIBS="$TORQ_PKG/iree/compiler/_mlir_libs"
SHIM=$(mktemp -d)
printf '#!/usr/bin/env bash\nexec "%s" -flavor gnu "$@"\n' "$MLIRLIBS/iree-lld" > "$SHIM/ld"
chmod +x "$SHIM/ld"
PATH="$SHIM:$PATH" LD_LIBRARY_PATH="$MLIRLIBS" "$MLIRLIBS/iree-compile" \
  --iree-input-type=onnx \
  --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=aarch64-unknown-linux-gnu \
  --iree-llvmcpu-target-cpu-features=+neon \
  "$WORK/preprocessor.mlir" -o "$OUT"
rm -rf "$SHIM" "$WORK"
echo ">> wrote $OUT ($(stat -c %s "$OUT") bytes)"
echo ">> deploy: adb push $OUT /home/root/moon/preprocessor_cpu.vmfb"
