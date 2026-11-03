#!/usr/bin/env bash
# Merge the cross-built runtime archives into one static lib and link r1_smoke against it.
# Usage: build-smoke.sh <work-dir>   (same <work-dir> as build-runtime.sh)
set -euo pipefail
W=$(readlink -f "$1"); D=$(cd "$(dirname "$0")" && pwd); B="$W/build-a64"
KONAN_LLVM=$HOME/.konan/dependencies/llvm-21-x86_64-linux-essentials-116
KONAN_GCC=$HOME/.konan/dependencies/aarch64-unknown-linux-gnu-gcc-8.3.0-glibc-2.25-kernel-4.9-2
SYSROOT=$KONAN_GCC/aarch64-unknown-linux-gnu/sysroot
mkdir -p "$D/out"
# One fat archive = what a K/N cinterop def would point staticLibraries at.
rm -f "$D/out/libiree_torq_runtime.a"
(cd "$B" && find . -name '*.a' ! -path "*/benchmark/*" ! -name libiree_runtime_unified.a | sort > "$D/out/archives.txt")
(cd "$B" && "$KONAN_LLVM/bin/llvm-ar" qcL "$D/out/libiree_torq_runtime.a" $(cat "$D/out/archives.txt"))
"$KONAN_LLVM/bin/llvm-ar" s "$D/out/libiree_torq_runtime.a"
"$KONAN_LLVM/bin/clang" --target=aarch64-unknown-linux-gnu --sysroot="$SYSROOT" --gcc-toolchain="$KONAN_GCC" \
  -mcpu=cortex-a55 -O2 -fuse-ld=lld \
  -DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl -I"$W/iree/runtime/src" -I"$B/third_party/iree/runtime/src" \
  "$D/r1_smoke.c" -o "$D/out/r1_smoke" \
  "$D/out/libiree_torq_runtime.a" \
  -Wl,-Bstatic -lstdc++ -lstdc++fs -Wl,-Bdynamic -static-libgcc -lm -ldl -lpthread
ls -la "$D/out/libiree_torq_runtime.a" "$D/out/r1_smoke"
