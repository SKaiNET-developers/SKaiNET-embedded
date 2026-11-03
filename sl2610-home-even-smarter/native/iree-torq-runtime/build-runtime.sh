#!/usr/bin/env bash
# Fetch + cross-build the Torq-fork IREE runtime (local-sync, local-task, vulkan, torq HAL drivers)
# for the board with the Kotlin/Native bundled clang-21 + gcc-8.3/glibc-2.25 sysroot.
# Usage: build-runtime.sh <work-dir>      (needs: git, cmake >= 3.21, host clang, python3)
# Result: <work-dir>/build-a64 (static archives, torq-run-module, iree-run-module)
set -euo pipefail
W=$(mkdir -p "$1" && readlink -f "$1"); HERE=$(cd "$(dirname "$0")" && pwd)
# torq-compiler tag == the board's Torq runtime version (board/board.lock TORQ_VERSION). Override for an update:
#   TORQ_TAG=v2.2.1 build-runtime.sh <new-work-dir>   (IREE_PIN then defaults to that tag's third_party/iree pin)
TORQ_TAG=${TORQ_TAG:-v2.0.0}

fetch() { # <dir> <url> <commit>
  [ -e "$1/.git" ] && return 0
  mkdir -p "$1" && git -C "$1" init -q && git -C "$1" remote add origin "$2"
  git -C "$1" fetch -q --depth 1 origin "$3" && git -C "$1" checkout -q FETCH_HEAD
}
[ -d "$W/torq" ] || git clone -q --depth 1 --branch "$TORQ_TAG" https://github.com/synaptics-torq/torq-compiler "$W/torq"
# The IREE fork commit the torq tag pins (v2.0.0: 43014101c8ab, IREE 3.10.0.dev).
IREE_PIN=${IREE_PIN:-$(git -C "$W/torq" ls-tree HEAD third_party/iree | awk '{print $3}')}
echo "torq $TORQ_TAG, iree fork $IREE_PIN"
fetch "$W/iree" https://github.com/synaptics-torq/iree.git "$IREE_PIN"
for sm in flatcc benchmark vulkan_headers; do
  url=$(git -C "$W/iree" config -f .gitmodules "submodule.third_party/$sm.url")
  sha=$(git -C "$W/iree" ls-tree HEAD "third_party/$sm" | awk '{print $3}')
  rmdir "$W/iree/third_party/$sm" 2>/dev/null || true
  fetch "$W/iree/third_party/$sm" "$url" "$sha"
done
rmdir "$W/torq/third_party/iree" 2>/dev/null || true
ln -sfn "$W/iree" "$W/torq/third_party/iree"
# Optional experiment: accept exec-format v0 (prebuilt vendor encoder) when TORQ_ACCEPT_EXEC_V0=1. Only for runtimes
# that accept exactly version 2 (v2.0.0); v2.2.x accepts every version up to 2 by itself.
if [ "${ACCEPT_EXEC_V0:-1}" = 1 ] && grep -q 'runtimeVersion != 2)' "$W/torq/runtime/driver/TorqExecutable.cc"; then
  patch -d "$W/torq" -p1 < "$HERE/patch-accept-exec-v0.diff"
fi
command -v ninja >/dev/null || { python3 -m venv "$W/venv" && "$W/venv/bin/pip" -q install ninja; export PATH="$W/venv/bin:$PATH"; }

# 1) host tools (flatcc compiler + embed-data) from the same fork
cmake -S "$W/iree" -B "$W/build-host" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_ENABLE_CPUINFO=OFF
cmake --build "$W/build-host" --target iree-flatcc-cli iree-c-embed-data
# 2) aarch64 runtime + torq driver, mirrors torq scripts/configure_soc_build.sh astra_machina
cmake -S "$W/torq" -B "$W/build-a64" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$HERE/toolchain.konan-aarch64.cmake" \
  -DIREE_HOST_BIN_DIR="$W/build-host/tools" \
  -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_PYTHON_BINDINGS=OFF \
  -DTORQ_ENABLE_AWS_FPGA=OFF -DTORQ_ENABLE_SOC_FPGA=OFF -DTORQ_ENABLE_ASTRA_MACHINA=ON \
  -DTORQ_ENABLE_SIMULATOR=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_DRIVER_LOCAL_TASK=ON -DIREE_HAL_DRIVER_VULKAN=ON \
  -DIREE_ENABLE_CPUINFO=OFF
cmake --build "$W/build-a64" --target iree_runtime_impl torq-run-module iree-run-module
