# Compiling MLIR Models for Google Coral NPU

## Coral NPU Architecture

- **ISA**: `rv32imf_zve32x_zicsr_zifencei_zbb`
  - RV32I: Base 32-bit integer
  - M: Multiply/divide
  - F: Single-precision hardware float (FP32)
  - Zve32x: 128-bit SIMD vector extension
  - Zicsr: Control/status registers
  - Zifencei: Instruction-fetch fence
  - Zbb: Bit manipulation
- **ABI**: `ilp32`
- **Code model**: `medany`
- **Runtime**: newlib-nano

## IREE Compilation Command

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
  --iree-vm-bytecode-module-strip-source-map=true \
  --iree-vm-emit-polyglot-zip=false \
  input.mlir \
  -o out/output.vmfb
```

### Critical Flags

| Flag | Value | Why |
|------|-------|-----|
| `--iree-llvmcpu-target-cpu-features` | `+m,+f` | **Required.** Without `+f`, the compiler emits soft-float code needing `__mulsf3`/`__addsf3` builtins that the embedded ELF linker can't resolve. |
| `--iree-llvmcpu-target-triple` | `riscv32-pc-linux-elf` | Bare-metal embedded ELF format for RISC-V 32-bit. |
| `--iree-llvmcpu-target-abi` | `ilp32` | 32-bit integers, longs, and pointers. |
| `--iree-stream-partitioning-favor` | `min-peak-memory` | Recommended for memory-constrained edge devices. |

For full Coral NPU feature support:
```
--iree-llvmcpu-target-cpu-features=+m,+f,+zve32x,+zicsr,+zifencei,+zbb
```

## Simulators

Two simulators are available for testing without hardware:

### MPACT-CoralNPU (Behavioral, faster)

```bash
bazel build //sim:coralnpu_v2_sim
bazel run //sim:coralnpu_v2_sim -- path/program.elf
# Interactive mode:
bazel run //sim:coralnpu_v2_sim -- --i path/program.elf
```

### Verilator (Cycle-accurate)

```bash
bazel build //tests/verilator_sim:rvv_core_mini_axi_sim
bazel-bin/tests/verilator_sim/rvv_core_mini_axi_sim --binary program.elf
```

Both accept `.elf` and `.bin` formats. ELF is preferred (contains debug symbols).

## Compilation Flow

```
JAX / PyTorch / TensorFlow
        |
        v
  StableHLO MLIR (.mlir)
        |
        v
  iree-compile (with Coral NPU target flags)
        |
        v
  VM Bytecode (.vmfb)
        |
        v
  Coral NPU Simulator or Hardware
```

## References

- Repository: https://github.com/google-coral/coralnpu
- Simulator docs: https://developers.google.com/coral/guides/software/simulator
- IREE compiler guide: https://developers.google.com/coral/guides/software/mlir-iree-compilers
- Coral NPU datasheet: https://developers.google.com/coral/guides/hardware/datasheet
- IREE bare-metal guide: https://iree.dev/guides/deployment-configurations/bare-metal/
