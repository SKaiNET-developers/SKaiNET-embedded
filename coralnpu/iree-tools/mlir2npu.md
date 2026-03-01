# MLIR to Coral NPU

## Coral NPU Architecture

- **ISA**: `rv32imf_zve32x_zicsr_zifencei_zbb`
- **ABI**: `ilp32`, code model `medany`
- **Memory**: ITCM 8KB (code, .text) + DTCM 32KB (data, .data/.bss) + EXTMEM 4MB
- **Runtime**: newlib-nano, no OS, bare-metal
- **Binary format**: ELF, built via `coralnpu_v2_binary` Bazel rule
- **I/O convention**: Global arrays with `__attribute__((section(".data")))`, accessed by symbol address

## The Intended Architecture (per Google)

According to Google's official documentation, the Coral NPU toolchain includes an **IREE compiler with a hardware-specific plug-in** that produces binaries directly:

> "A model from a framework like JAX is first imported into the MLIR format using the StableHLO dialect. This intermediate file is then fed into the IREE compiler, which applies a hardware-specific plug-in to recognize the Coral NPU's architecture. From there, the compiler performs progressive lowering — a critical optimization step where the code is systematically translated through a series of dialects, moving closer to the machine's native language. After optimization, the toolchain generates a final, compact binary file ready for efficient execution on the edge device."
>
> — [Introducing Coral NPU: A full-stack platform for Edge AI](https://developers.googleblog.com/introducing-coral-npu-a-full-stack-platform-for-edge-ai/)

The intended pipeline:

```
JAX / PyTorch / TensorFlow
    │
    v
StableHLO MLIR
    │
    v
IREE compiler + Coral NPU hardware plug-in
    │  (progressive lowering: StableHLO → linalg → loops → LLVM → RV32)
    │  (Coral-specific custom MLIR dialects for matrix/vector ops)
    v
Compact binary (.elf) → runs on Coral NPU simulator or hardware
```

The Synaptics Torq NPU (first commercial SoC with Coral NPU) has "custom MLIR dialects for both low-level assembly and high-level matrix operations" — the plug-in likely targets the matrix/vector execution units directly.

### What's NOT in the open-source repo

**The IREE Coral NPU plug-in does not exist in the open-source `coralnpu/` repository** (released October 2025). Zero references to IREE anywhere in the repo. The plug-in is either:

- Internal to Google (not yet open-sourced)
- Part of Synaptics' proprietary toolchain for the Torq SoC
- Still under development

What IS available in the open-source repo is the bare-metal C compilation path (see below), and the IREE plug-in exists in the **Synaptics Torq compiler** (see next section).

## Synaptics Torq Compiler

The missing IREE Coral NPU plug-in lives in the **Synaptics Torq compiler** — an open-source IREE compiler plugin at [`github.com/synaptics-torq/torq-compiler`](https://github.com/synaptics-torq/torq-compiler).

**What it is**: An IREE plugin that registers a HAL target device and backend named `"torq"`. It targets the **Synaptics SL2610 SoC** — the first commercial chip with a Coral NPU core — not the generic open-source Coral NPU directly.

**Custom MLIR dialects**:
- `TorqHL` — high-level operations (tensor-level)
- `TorqHW` — hardware-level operations (maps to SL2610 execution units)

**Lowering pipeline**:
```
TOSA / Torch / ONNX / Linalg
    │
    v
TorqHL dialect
    │
    v
TorqHW dialect
    │
    v
Codegen → binary for SL2610
```

**Input types**: `tosa-torq`, `torch-torq`, `onnx-torq`, `linalg-torq`

**Key file**: `compiler/torq/PluginRegistration.cpp` — registers the HAL target device, backend, and dialect passes.

**Also forks**: `iree`, `llvm-project`, `torch-mlir` — the full build pulls in these custom forks alongside the plugin.

**Limitation**: The Torq compiler targets the SL2610's specific hardware, not the generic open-source Coral NPU ISA. It may not produce binaries compatible with the `coralnpu/` simulator without adaptation.

## How CoralNPU Examples Run Today

The open-source `coralnpu/` repo has only the C compilation path — no IREE integration:

```
rgb2grayscale.cc
    │
    │  Bazel: coralnpu_v2_binary()
    │  ├── Platform transition → //platforms:coralnpu_v2
    │  ├── Clang/GCC cross-compiler (riscv32-unknown-elf, rv32imf, -O3)
    │  ├── CRT startup (coralnpu_start.S): _start → clear BSS → init FP/Vector → call main()
    │  ├── Linker script (coralnpu_tcm.ld): .text → ITCM, .data → DTCM
    │  └── newlib-nano (libc, libm, lstdc++, lgcc)
    │
    v
coralnpu_v2_rgb2grayscale.elf  (bare-metal RISC-V binary)
    │
    v
Simulator loads ELF → CPU starts at 0x0 → _start → main() → ebreak → halt
```

### Key files in `coralnpu/`

| File | Role |
|------|------|
| `rules/coralnpu_v2.bzl` | Bazel macro: compile C → ELF + BIN + VMEM |
| `toolchain/cc_toolchain_config.bzl` | Clang flags: `-march=rv32imf_zve32x...`, `-O3`, `-nostdlib` |
| `toolchain/crt/coralnpu_start.S` | Assembly startup: BSS clear, FP/Vector enable, call `main()`, `ebreak` |
| `toolchain/crt/coralnpu_gloss.cc` | newlib stubs: `_write()`, `_sbrk()`, `_exit()` |
| `toolchain/coralnpu_tcm.ld.tpl` | Linker script template: ITCM/DTCM/EXTMEM layout |
| `rules/linker.bzl` | `generate_linker_script()` — substitutes memory sizes into template |
| `platforms/BUILD.bazel` | Platform constraint: `coralnpu_v2` (bare-metal) or `coralnpu_v2_semihosting` |

### CRT startup sequence (`coralnpu_start.S`)

1. Load stack pointer and global pointer from linker symbols
2. Zero out `.bss` section
3. Run C++ constructors (`.init_array`)
4. Set exception handler CSR
5. Enable FP and Vector extensions in `mstatus`
6. Write sentinel `0x0badd00d` to `_ret` symbol
7. Call `main(0, 0)`
8. Save return value, run C++ destructors
9. Store return value at `_ret`, `ebreak` to halt
10. Read cycle counters, `mpause`, infinite loop

### Memory map (from linker script)

```
0x00000000  ┌─────────────┐
            │   ITCM      │  8KB — .text, .rodata, .init_array
0x00002000  ├─────────────┤
            │   (gap)     │
0x00010000  ├─────────────┤
            │   DTCM      │  32KB — .data, .bss, .heap, .stack
0x00018000  ├─────────────┤
            │   (gap)     │
0x20000000  ├─────────────┤
            │   EXTMEM    │  4MB — .extdata, .extbss (overflow)
0x20400000  └─────────────┘
```

### Example: `rgb2grayscale.cc`

```c
float input[C_IN * HW] __attribute__((section(".data")));   // in DTCM
float output[C_OUT * HW] __attribute__((section(".data")));  // in DTCM
static const float weights[C_IN] = {0.2989f, 0.587f, 0.114f};

int main() {
  for (int i = 0; i < HW; i++) {
    float sum = 0.0f;
    for (int c = 0; c < C_IN; c++)
      sum += input[c * HW + i] * weights[c];
    output[i] = sum;
  }
  return 0;
}
```

Simulator writes input floats to `input` symbol address, runs, reads back from `output` symbol address.

## Current Status

### What works

```
SKaiNET (Kotlin)
    │
    ├── --backend=hlo-export → StableHLO MLIR (.mlir)
    │
    ├── iree-compile (host) → .vmfb → iree-run-module  ✓  reference output
    ├── iree-compile (rv32)  → .vmfb                    ✗  can't run, needs IREE runtime
    │
    └── Python transpiler → .cc → bazel build → .elf   ✓  runs on MPACT simulator
```

Verified on both MLIR variants:
- `rgb2grayscale.mlir` — function args, all f32
- `rgb2grayscale_skainet.mlir` — no args, baked-in constants, f16+f32 mix


## The Gap

Google describes an IREE plug-in that does `StableHLO → progressive lowering → compact binary`. This plug-in is not available in the open-source release. The generic IREE compiler (from PyPI) compiles to VMFB, which wraps kernels in a VM/HAL runtime that does not fit on the NPU.

What we have access to:

```
StableHLO MLIR  →  ???  →  C source or .o  →  coralnpu_v2_binary  →  .elf  →  simulator
```

The `coralnpu_v2_binary` Bazel rule handles everything after C source: toolchain, linker script, CRT, memory layout. The missing piece is the `???` — translating StableHLO ops into bare-metal code without the unreleased IREE plug-in.

### What generic IREE produces (not usable)

| IREE output mode | Result | Runs on Coral NPU? |
|---|---|---|
| `--output-format=vm-bytecode` | `.vmfb` — FlatBuffer with RV32 code + VM orchestration | No — needs IREE runtime |
| `--output-format=vm-c` | 4600-line C needing `iree/vm/api.h`, HAL | No — needs IREE runtime library |
| `--iree-llvmcpu-static-library-output-path` | Should emit `.o` + `.h` | Did not produce output for rv32 |

The RISC-V machine code exists inside the VMFB, but it's not extractable as a standalone bare-metal binary.

## Options to Close the Gap

### Option A: Use Synaptics Torq compiler

The IREE plug-in exists in the [Torq compiler](https://github.com/synaptics-torq/torq-compiler) (see above). However, it targets the **Synaptics SL2610 SoC**, not the generic open-source Coral NPU. The generated binaries may not run on the `coralnpu/` simulator without adaptation to the different memory map and hardware peripherals.

Worth investigating whether the Torq codegen can be retargeted to the generic Coral NPU ISA, or whether a future `coralnpu/` release will include its own IREE plugin.

### Option B: Direct MLIR lowering (no IREE)

Use the MLIR toolchain to lower StableHLO → LLVM IR → RISC-V object code:

```
mlir-opt  (StableHLO → linalg → loops → LLVM dialect)
    → mlir-translate  (LLVM dialect → LLVM IR)
    → llc  (LLVM IR → riscv32 .o)
    → coralnpu_v2_binary  (link with CRT + linker script → .elf)
```

Pro: Standard MLIR toolchain, no custom code, no IREE runtime needed.
Con: Need `mlir-opt` with StableHLO dialect support. Pass pipeline is tricky. I/O buffer convention needs handling. Does not target matrix/vector units.

### Option C: Transpiler in Kotlin/Native

Same approach as the current Python transpiler but in Kotlin Multiplatform. Fits the SKaiNET ecosystem.

```
StableHLO MLIR  →  Kotlin CLI (parse + codegen)  →  .cc  →  coralnpu_v2_binary  →  .elf
```

Pro: No Python dependency, integrates with SKaiNET, compiles to native binary.
Con: Reimplementation effort, limited to ops we implement, does not target matrix/vector units.

### Option D: SKaiNET emits C directly

Skip MLIR entirely for the Coral NPU target. SKaiNET already knows the computation graph — add a `--backend=coralnpu-c` that emits C source matching `coralnpu_v2_binary` conventions.

```
SKaiNET  →  --backend=coralnpu-c  →  .cc  →  coralnpu_v2_binary  →  .elf
```

Pro: Simplest path, no MLIR parsing, full control, works today.
Con: Bypasses MLIR, loses optimization interop. Does not target matrix/vector units.

## IREE Commands (Reference)

### Compile for host (verification)

```bash
uv run iree-compile \
  --output-format=vm-bytecode \
  --iree-input-type=stablehlo \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  input.mlir -o out/output_host.vmfb
```

### Compile for rv32 (produces VMFB, not runnable on NPU)

```bash
uv run iree-compile \
  --output-format=vm-bytecode \
  --iree-input-type=stablehlo \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=riscv32-pc-linux-elf \
  --iree-llvmcpu-target-cpu=generic-rv32 \
  --iree-llvmcpu-target-cpu-features=+m,+f \
  --iree-llvmcpu-target-abi=ilp32 \
  --iree-llvmcpu-debug-symbols=false \
  --iree-stream-partitioning-favor=min-peak-memory \
  input.mlir -o out/output_rv32.vmfb
```

### Verify on host

```bash
uv run iree-run-module --module=out/output_host.vmfb --function=function_name
```

## Simulators

### MPACT-CoralNPU (behavioral, faster)

```bash
cd coralnpu/
bazel run //sim:coralnpu_v2_sim -- path/program.elf
```

### Verilator (cycle-accurate)

```bash
cd coralnpu/
bazel build //tests/verilator_sim:rvv_core_mini_axi_sim
bazel-bin/tests/verilator_sim/rvv_core_mini_axi_sim --binary program.elf
```

### Simulator Python API (`coralnpu_v2_sim_utils.py`)

```python
sim = CoralNPUV2Simulator()
entry, symbols = sim.get_elf_entry_and_symbol(elf, ["input", "output"])
sim.load_program(elf, entry)
sim.write_memory(symbols["input"], data.astype(np.float32).view(np.uint8))
sim.run()
sim.wait()
result = sim.read_memory(symbols["output"], num_bytes).view(np.float32)
```

Note: Building the Python simulator requires the host Clang toolchain constraint (`//toolchain/host_clang:clang_compiler`). The pybind `.so` is at `bazel-bin/sw/coralnpu_sim/coralnpu_v2_sim_pybind.so`.

## Files in `iree-tools/`

| File | Purpose |
|------|---------|
| `ir.py` | Dataclasses: `Module`, `FuncDef`, `ConstantOp`, `ConvertOp`, `ConvolutionOp`, `BinaryOp` |
| `mlir_parser.py` | Regex-based StableHLO text parser → IR dataclasses |
| `codegen.py` | IR → C code generator (coralnpu_v2_binary compatible) |
| `iree_commands.py` | Wrappers for `iree-compile` and `iree-run-module` |
| `bazel_builder.py` | Writes .cc + BUILD.bazel to `coralnpu/examples/generated/`, invokes bazel |
| `simulator.py` | Drives MPACT simulator via Python API |
| `main.py` | CLI: `compile`, `verify`, `generate-c`, `build-elf`, `simulate`, `run-all` |
