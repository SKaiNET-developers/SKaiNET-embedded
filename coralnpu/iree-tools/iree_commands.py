"""Wrappers for iree-compile and iree-run-module subprocess calls."""

from __future__ import annotations

import subprocess
from pathlib import Path


def compile_host(input_mlir: str, output_vmfb: str) -> str:
    """Compile StableHLO MLIR to host VMFB for verification."""
    Path(output_vmfb).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "iree-compile",
        "--output-format=vm-bytecode",
        "--iree-input-type=stablehlo",
        "--iree-hal-target-device=local",
        "--iree-hal-local-target-device-backends=llvm-cpu",
        input_mlir,
        "-o",
        output_vmfb,
    ]
    subprocess.run(cmd, check=True)
    return output_vmfb


def compile_rv32(input_mlir: str, output_vmfb: str) -> str:
    """Compile StableHLO MLIR to RISC-V 32-bit VMFB."""
    Path(output_vmfb).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "iree-compile",
        "--output-format=vm-bytecode",
        "--iree-input-type=stablehlo",
        "--iree-hal-target-device=local",
        "--iree-hal-local-target-device-backends=llvm-cpu",
        "--iree-llvmcpu-target-triple=riscv32-pc-linux-elf",
        "--iree-llvmcpu-target-cpu=generic-rv32",
        "--iree-llvmcpu-target-cpu-features=+m,+f",
        "--iree-llvmcpu-target-abi=ilp32",
        "--iree-llvmcpu-debug-symbols=false",
        "--iree-stream-partitioning-favor=min-peak-memory",
        "--iree-vm-bytecode-module-strip-source-map=true",
        "--iree-vm-emit-polyglot-zip=false",
        input_mlir,
        "-o",
        output_vmfb,
    ]
    subprocess.run(cmd, check=True)
    return output_vmfb


def verify_host(
    vmfb_path: str,
    function_name: str,
    inputs: list[str] | None = None,
) -> str:
    """Run iree-run-module on host CPU and return stdout.

    Args:
        vmfb_path: Path to the compiled VMFB.
        function_name: Name of the function to invoke.
        inputs: Optional list of input specifications, e.g.
                ["1x3x4x4xf32=1,2,3,..."].
    """
    cmd = [
        "uv",
        "run",
        "iree-run-module",
        f"--module={vmfb_path}",
        f"--function={function_name}",
    ]
    if inputs:
        for inp in inputs:
            cmd.append(f"--input={inp}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout
