# Native IREE runtime with CPU + Torq NPU in one aarch64 process (from spike R1)

Verdict: **GO.** One native process on the board loads the Torq-fork IREE runtime as a static
library, creates one `iree_runtime_instance`, and runs vmfbs on `local-task` (CPU) and on `torq`
(NPU), interleaved, through the plain IREE C API (`iree/runtime/api.h`). No Python and no
`torq-run-module` subprocess.

## Findings

- **No reusable shared library on the board.** `torq2-stable` ships `iree/_runtime_libs/_runtime.cpython-312-aarch64-linux-gnu.so`
  (the IREE runtime, local-sync/local-task/torq linked in) plus statically linked tools. The `.so` is stripped
  and exports only `PyInit__runtime` plus some C++ RTTI/template symbols. It does not export
  `iree_runtime_*` or `iree_hal_*` C functions, so it cannot be dlopened as a C API.
- **The Torq HAL driver is open source** (Apache-2.0): `github.com/synaptics-torq/torq-compiler` tag `v2.0.0`
  (`runtime/driver`, `runtime/torq_hw`, `src/TorqAstraMachina.cpp` talks to `/dev/torq`). Strings in the
  board `.so` (`/workspace/code/runtime/driver/torq_driver.c` ...) match this tree.
- **IREE base:** the torq submodule pins `synaptics-torq/iree@43014101c8ab` (2026-06-08), `package-version 3.10.0.dev`,
  forked from upstream `ae97779` (2026-01-06) plus about 19 fork commits. VM bytecode module version 0, `hal` module version >= 6.
- **Upstream 3.11 vmfbs do not load** on the fork runtime (stock or self-built). `iree-compile 3.11.0rc20260316`
  output fails in `vm/bytecode/verifier.c:40` with `required module features [Ch] are not available ... have [EXT_F32|EXT_F64]`.
  CPU graphs must therefore be compiled with the fork compiler (`sl2610-iree:v2.0.0`, `--iree-hal-target-device=local
  --iree-hal-local-target-device-backends=llvm-cpu`, which works) or with an upstream compiler no newer than the fork base.
  The existing board CPU vmfbs (Gemma `gemma-gen.vmfb` + `.irpa`, Moonshine `encoder_cpu`, `enc_arm64`, `frontend_v2`,
  `decoder_prefill_cpu`) already load. Gemma output is identical between the self-built and the stock runtime.
- **Torq exec format:** the v2.0.0 runtime accepts only exec-format v2 (`TorqExecutable.cc:666`). The prebuilt vendor
  Moonshine encoder is exec v0. With the opt-in `patch-accept-exec-v0.diff` (`TORQ_ACCEPT_EXEC_V0=1`) it runs, and its
  output is **bit-identical** to the 2.0.0a1 runtime the app uses today (cmp on a 1x288x207 bf16 input).

## Build recipe (host x86_64 to board aarch64)

```
./build-runtime.sh <work>   # fetch torq v2.0.0 + iree fork pin + flatcc/benchmark/vulkan_headers; host tools; cross build (~40 s on 12 cores)
./compile-vmfbs.sh          # add_f32 (llvm-cpu, fork compiler), add_bf16 (torq), add_f32 with upstream 3.11 (negative test)
./build-smoke.sh <work>     # out/libiree_torq_runtime.a (one fat archive) + out/r1_smoke
./run-on-board.sh           # ENCODER=<board path to encoder.vmfb> also runs the vendor encoder
```

Toolchain (`toolchain.konan-aarch64.cmake`): the Kotlin/Native bundled clang-21/lld
(`~/.konan/dependencies/llvm-21-x86_64-linux-essentials-116`) with the K/N gcc-8.3/glibc-2.25 sysroot. This is
the same sysroot K/N uses for linuxArm64, so the archive can go straight into cinterop. Gotchas:
- The kernel-4.9 sysroot has no `linux/dma-buf.h` or `linux/dma-heap.h`. Copies of these uapi headers are in `compat-include/`.
- gcc-8 libstdc++ keeps `std::filesystem` in `libstdc++fs.a`, so link `-lstdc++fs`.
- Consumers must define `-DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl`. Include dirs: `iree/runtime/src` and
  `build-a64/third_party/iree/runtime/src` (generated flatbuffer headers).
- Do not use `--whole-archive`, because the archive holds duplicate libbacktrace and allocator objects. Link
  `-lstdc++ -lstdc++fs` statically plus `-lm -ldl -lpthread`.

Result: `r1_smoke` NEEDED is only libc/libm/libdl/libpthread, max symbol version `GLIBC_2.17`, with no libstdc++.so and no
libcrypt. The board has glibc 2.39.

## Runtime knobs (process-global IREE flags)
`--torq_hw_type` (default `astra_machina`), `--torq_device_allocator=cpu|dmabuf` (default `cpu`), `--task_topology_*`.
In-process, pass them through `iree_flags_parse_checked()` with a synthetic argv.

## Files
`r1_smoke.c` (C API demo), `mlir/` (smoke graphs), `out/` (artifacts), `bf16dump.py` (BusyBox has no `od -A`).
