# iree-tools

StableHLO MLIR to Coral NPU pipeline. Takes MLIR exported from SKaiNET (or any StableHLO source), transpiles it to C, builds a bare-metal RISC-V ELF via the `coralnpu/` Bazel toolchain, and runs it on the MPACT simulator.

## Prerequisites

- Python 3.13+ with [uv](https://docs.astral.sh/uv/)
- [Bazel](https://bazel.build/) (for building ELFs and the simulator)
- Clang 19 (`clang` must resolve to clang-19; the `coralnpu/` toolchain requires it)
- The `coralnpu/` repository checked out at `~/projects/coral/coralnpu` (or set `CORALNPU_ROOT`)

Install Python dependencies:

```bash
uv sync
```

## Pipeline overview

```
StableHLO MLIR (.mlir)
    |
    |  1. uv run python main.py generate-c
    v
C source (.cc)
    |
    |  2. uv run python main.py build-elf
    |     (writes .cc + BUILD.bazel to coralnpu/examples/generated/,
    |      invokes bazel build with coralnpu_v2_binary rule)
    v
Bare-metal RISC-V ELF (.elf)
    |
    |  3. bazel run //examples/generated:sim_runner
    |     (runs ELF on MPACT behavioral simulator)
    v
Simulation output (cycle count + output arrays)
```

Optional: verify the MLIR on the host CPU via IREE before transpiling.

## Quick start

Using the included `rgb2grayscale.mlir` as an example:

### 1. Verify on host (optional)

Compile the MLIR with IREE and run on the host CPU to get a reference output:

```bash
uv run python main.py verify rgb2grayscale.mlir
```

Output:

```
EXEC @rgb2grayscale
result[0]: hal.buffer_view
1x1x4x4xf32=[[[0 0 0 0][0 0 0 0][0 0 0 0][0 0 0 0]]]
```

(All zeros because the default input is zeros.)

### 2. Generate C source

Transpile StableHLO MLIR to C code compatible with `coralnpu_v2_binary`:

```bash
uv run python main.py generate-c rgb2grayscale.mlir
```

Output goes to `out/rgb2grayscale.cc`. The generated C looks like:

```c
float input_0[48] __attribute__((section(".data")));
float output_0[16] __attribute__((section(".data")));

static const float v0[3] = {0.2989f, 0.587f, 0.114f};

int main() {
  // 1x1 convolution: 3 input channels -> 1 output channels
  for (int i = 0; i < 16; i++) {
    float sum = 0.0f;
    for (int c = 0; c < 3; c++) {
      sum += input_0[c * 16 + i] * v0[c];
    }
    output_0[i] = sum;
  }
  return 0;
}
```

### 3. Build ELF

Generate C and build the bare-metal RISC-V ELF via Bazel:

```bash
uv run python main.py build-elf rgb2grayscale.mlir
```

This writes `rgb2grayscale.cc` and `BUILD.bazel` to `$CORALNPU_ROOT/examples/generated/` and runs `bazel build`. The ELF path is printed at the end.

### 4. Run on MPACT simulator

Run the ELF on the Coral NPU behavioral simulator. This step uses `bazel run` from the `coralnpu/` repo:

```bash
cd ~/projects/coral/coralnpu

# Find the ELF path (the coralnpu_v2_binary rule uses a platform transition,
# so the ELF ends up in a config-specific bazel-out/ dir, not bazel-bin/)
ELF=$(pwd)/$(bazel cquery //examples/generated:coralnpu_v2_rgb2grayscale \
  --output=files 2>&1 | grep '\.elf$')

# Run on simulator (must pass absolute path — sim_runner runs in a Bazel sandbox)
bazel run //examples/generated:sim_runner -- \
  $ELF \
  --output output_0=16
```

The sim_runner prints JSON with the output arrays and cycle count.

Note: the first run builds the MPACT simulator pybind `.so` (~2 min). Subsequent runs are fast.

## CLI reference

All commands are run with `uv run python main.py <command>`.

| Command | Description |
|---------|-------------|
| `generate-c <mlir>` | Transpile MLIR to C source (writes to `out/` or `-o path`) |
| `build-elf <mlir>` | Generate C + build bare-metal ELF via Bazel |
| `compile <mlir>` | Compile MLIR to host + rv32 VMFB via IREE |
| `verify <mlir>` | Compile for host and run with `iree-run-module` |
| `simulate <elf>` | Run ELF on MPACT simulator (via `bazel run`) |
| `run-all <mlir>` | Full pipeline: compile, verify, generate-c, build, simulate |

## Project files

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point with subcommands |
| `mlir_parser.py` | Regex-based StableHLO text parser |
| `ir.py` | Dataclasses: `Module`, `FuncDef`, `ConstantOp`, `ConvertOp`, `ConvolutionOp`, `BinaryOp` |
| `codegen.py` | IR to C code generator (coralnpu_v2_binary compatible) |
| `iree_commands.py` | Wrappers for `iree-compile` and `iree-run-module` |
| `bazel_builder.py` | Writes .cc + sim_runner.py + BUILD.bazel to coralnpu repo, invokes bazel |
| `simulator.py` | Drives MPACT simulator via `bazel run` |
| `rgb2grayscale.mlir` | Example StableHLO MLIR (1x1 convolution, RGB to grayscale) |
| `mlir2npu.md` | Detailed notes on the MLIR to Coral NPU pipeline |
| `mlir2coral.md` | Additional research notes |

## Supported StableHLO ops

The parser and codegen handle:

- `stablehlo.constant` — dense tensor constants
- `stablehlo.convert` — f16/f32 type conversion (no-op, Coral NPU uses f32)
- `stablehlo.convolution` — 1x1 (optimized) and general convolutions (with stride, padding, dilation)
- `stablehlo.add`, `multiply`, `subtract`, `divide` — element-wise binary ops

All f16 values are promoted to f32 (Coral NPU has hardware f32, no f16 support).

## How the generated C works

The codegen produces C code that follows the `coralnpu_v2_binary` conventions:

- **Input/output arrays** are global `float[]` with `__attribute__((section(".data")))` so they land in DTCM
- **Constants** are `static const float[]`
- **`main()`** runs the computation and returns 0
- The simulator writes input data to `input_N` symbol addresses, runs until `ebreak`, then reads from `output_N` symbol addresses

Memory layout (from the linker script):

```
0x00000000  ITCM   8KB  — .text, .rodata
0x00010000  DTCM  32KB  — .data, .bss, .heap, .stack
0x20000000  EXTMEM 4MB  — overflow
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CORALNPU_ROOT` | `~/projects/coral/coralnpu` | Path to the coralnpu repository |

## Building the simulator

The MPACT simulator pybind `.so` is built automatically on first `bazel run //examples/generated:sim_runner`. Requirements:

- Clang 19 as default `clang` (`sudo ln -sf /usr/bin/clang-19 /usr/bin/clang`)
- The simulator uses Bazel's Python 3.11 toolchain (not your system Python)

If you get `_PyThreadState_UncheckedGet` errors, it means Python version mismatch — always run the simulator through `bazel run`, not by importing the `.so` directly.
