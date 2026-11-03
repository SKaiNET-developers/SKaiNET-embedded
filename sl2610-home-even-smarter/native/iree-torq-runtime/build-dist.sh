#!/usr/bin/env bash
# Packages what the Kotlin/Native cinterop (:iree) links against, for the board (linuxArm64):
#   dist/include/hes_iree.h          the C API (self-contained, no IREE headers needed)
#   dist/lib/libhes_iree.a           the shim (hes_iree.c, hes_moonshine.c)
#   dist/lib/libiree_torq_runtime.a  the Torq-fork IREE runtime: local-sync, local-task, vulkan, torq
# Usage: [OUT=<dir>] build-dist.sh [work-dir]   (defaults: ./dist, ./work; built with build-runtime.sh first when missing)
# Uses the Kotlin/Native bundled clang and sysroot, so the archives match what K/N links for linuxArm64.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
W=$(mkdir -p "${1:-$D/work}" && readlink -f "${1:-$D/work}")
B="$W/build-a64"
[ -d "$B" ] || "$D/build-runtime.sh" "$W"
KONAN_LLVM=$HOME/.konan/dependencies/llvm-21-x86_64-linux-essentials-116
KONAN_GCC=$HOME/.konan/dependencies/aarch64-unknown-linux-gnu-gcc-8.3.0-glibc-2.25-kernel-4.9-2
SYSROOT=$KONAN_GCC/aarch64-unknown-linux-gnu/sysroot
OUT=$(mkdir -p "${OUT:-$D/dist}" && readlink -f "${OUT:-$D/dist}")   # absolute: step 1 runs from inside $B
mkdir -p "$OUT/lib" "$OUT/include" "$W/obj"

# 1) runtime: one fat archive of every runtime/driver archive (not --whole-archive: it holds duplicate objects)
rm -f "$OUT/lib/libiree_torq_runtime.a"
(cd "$B" && find . -name '*.a' ! -path "*/benchmark/*" ! -name libiree_runtime_unified.a | sort > "$W/archives.txt")
(cd "$B" && "$KONAN_LLVM/bin/llvm-ar" qcL "$OUT/lib/libiree_torq_runtime.a" $(cat "$W/archives.txt"))
"$KONAN_LLVM/bin/llvm-ar" s "$OUT/lib/libiree_torq_runtime.a"

# 2) shim
"$KONAN_LLVM/bin/clang" --target=aarch64-unknown-linux-gnu --sysroot="$SYSROOT" --gcc-toolchain="$KONAN_GCC" \
  -mcpu=cortex-a55 -O2 -fPIC -std=c11 -Wall -Werror=implicit-function-declaration \
  -DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl \
  -I"$W/iree/runtime/src" -I"$B/third_party/iree/runtime/src" \
  -c "$D/hes_iree.c" -o "$W/obj/hes_iree.o"
"$KONAN_LLVM/bin/clang" --target=aarch64-unknown-linux-gnu --sysroot="$SYSROOT" --gcc-toolchain="$KONAN_GCC" \
  -mcpu=cortex-a55 -O2 -fPIC -std=c11 -Wall -Werror=implicit-function-declaration \
  -DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl \
  -I"$W/iree/runtime/src" -I"$B/third_party/iree/runtime/src" \
  -c "$D/hes_moonshine.c" -o "$W/obj/hes_moonshine.o"
rm -f "$OUT/lib/libhes_iree.a"
"$KONAN_LLVM/bin/llvm-ar" rcs "$OUT/lib/libhes_iree.a" "$W/obj/hes_iree.o" "$W/obj/hes_moonshine.o"
cp "$D/hes_iree.h" "$OUT/include/"

# 3) provenance: which Torq tag / IREE fork commit this dist was built from
{
  echo "torq: $(git -C "$W/torq" describe --tags --always 2>/dev/null)"
  echo "iree: $(git -C "$W/iree" rev-parse HEAD 2>/dev/null)"
  echo "exec_v0_patch: $(grep -q TORQ_ACCEPT_EXEC_V0 "$W/torq/runtime/driver/TorqExecutable.cc" && echo applied || echo no)"
} > "$OUT/BUILD-INFO"
ls -la "$OUT/lib"; cat "$OUT/BUILD-INFO"
