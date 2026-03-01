"""CLI for the SKaiNET -> Coral NPU simulator pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from codegen import generate_c
from mlir_parser import parse_module


def _stem(mlir_path: str) -> str:
    """Get the stem name from an MLIR file path."""
    return Path(mlir_path).stem


def _read_mlir(path: str) -> str:
    return Path(path).read_text()


def _parse_and_get_func(mlir_text: str):
    module = parse_module(mlir_text)
    if not module.functions:
        print("Error: no functions found in MLIR module", file=sys.stderr)
        sys.exit(1)
    return module, module.functions[0]


# -- Subcommands --


def cmd_compile(args: argparse.Namespace) -> None:
    """Compile MLIR to host + rv32 VMFB."""
    from iree_commands import compile_host, compile_rv32

    mlir = args.input
    stem = _stem(mlir)
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    host_vmfb = str(out_dir / f"{stem}_host.vmfb")
    rv32_vmfb = str(out_dir / f"{stem}_rv32.vmfb")

    print(f"Compiling {mlir} for host...")
    compile_host(mlir, host_vmfb)
    print(f"  -> {host_vmfb}")

    print(f"Compiling {mlir} for rv32...")
    compile_rv32(mlir, rv32_vmfb)
    print(f"  -> {rv32_vmfb}")


def cmd_verify(args: argparse.Namespace) -> None:
    """Compile for host and run iree-run-module to get reference output."""
    from iree_commands import compile_host, verify_host

    mlir = args.input
    stem = _stem(mlir)
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    host_vmfb = str(out_dir / f"{stem}_host.vmfb")
    print(f"Compiling {mlir} for host...")
    compile_host(mlir, host_vmfb)

    mlir_text = _read_mlir(mlir)
    _, func = _parse_and_get_func(mlir_text)

    # Build input specs for functions with arguments
    inputs = None
    if func.args:
        inputs = []
        for _name, ttype in func.args:
            shape_str = "x".join(str(d) for d in ttype.shape)
            inputs.append(f"{shape_str}x{ttype.element_type}")

    print(f"Running {func.name} on host...")
    output = verify_host(host_vmfb, func.name, inputs)
    print(output)


def cmd_generate_c(args: argparse.Namespace) -> None:
    """Transpile MLIR to C source."""
    mlir_text = _read_mlir(args.input)
    module = parse_module(mlir_text)
    c_source = generate_c(module)

    if args.output:
        out_path = args.output
    else:
        out_path = str(Path("out") / f"{_stem(args.input)}.cc")
        Path("out").mkdir(exist_ok=True)

    Path(out_path).write_text(c_source)
    print(f"Generated C source: {out_path}")


def cmd_build_elf(args: argparse.Namespace) -> None:
    """Generate C and build ELF via Bazel."""
    from bazel_builder import build_elf, write_generated_files

    mlir_text = _read_mlir(args.input)
    module = parse_module(mlir_text)
    c_source = generate_c(module)

    name = _stem(args.input)
    cc_path, build_path = write_generated_files(name, c_source)
    print(f"Generated: {cc_path}")
    print(f"Generated: {build_path}")

    print(f"Building ELF...")
    elf_path = build_elf(name)
    print(f"ELF: {elf_path}")


def cmd_simulate(args: argparse.Namespace) -> None:
    """Run an ELF on the MPACT simulator."""
    from simulator import run_elf

    elf_path = args.input

    # Parse output sizes from --output-sizes flag (e.g. "output_0=16")
    output_symbols = []
    output_sizes = {}
    if args.output_sizes:
        for spec in args.output_sizes:
            sym, size = spec.split("=")
            output_symbols.append(sym)
            output_sizes[sym] = int(size)
    else:
        output_symbols = ["output_0"]
        output_sizes = {"output_0": 16}  # default for rgb2grayscale

    results = run_elf(
        elf_path,
        output_symbols=output_symbols,
        output_sizes=output_sizes,
    )

    for sym in output_symbols:
        print(f"{sym}: {results[sym]}")
    print(f"Cycle count: {results['_cycle_count']}")


def cmd_run_all(args: argparse.Namespace) -> None:
    """Full end-to-end pipeline: compile, verify, generate C, build, simulate, compare."""
    from bazel_builder import build_elf, write_generated_files
    from iree_commands import compile_host, verify_host
    from simulator import run_elf

    mlir = args.input
    stem = _stem(mlir)
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    mlir_text = _read_mlir(mlir)
    module, func = _parse_and_get_func(mlir_text)

    # Step 1: Compile for host
    print("=" * 60)
    print("Step 1: Compile for host")
    host_vmfb = str(out_dir / f"{stem}_host.vmfb")
    compile_host(mlir, host_vmfb)
    print(f"  -> {host_vmfb}")

    # Step 2: Verify on host
    print("=" * 60)
    print("Step 2: Verify on host (reference output)")
    inputs = None
    if func.args:
        inputs = []
        for _name, ttype in func.args:
            shape_str = "x".join(str(d) for d in ttype.shape)
            inputs.append(f"{shape_str}x{ttype.element_type}")
    host_output = verify_host(host_vmfb, func.name, inputs)
    print(host_output)

    # Step 3: Generate C
    print("=" * 60)
    print("Step 3: Generate C source")
    c_source = generate_c(module)
    cc_path, build_path = write_generated_files(stem, c_source)
    print(f"  -> {cc_path}")

    # Step 4: Build ELF
    print("=" * 60)
    print("Step 4: Build ELF via Bazel")
    elf_path = build_elf(stem)
    print(f"  -> {elf_path}")

    # Step 5: Simulate
    print("=" * 60)
    print("Step 5: Run on MPACT simulator")

    # Determine output sizes from the IR
    from ir import ReturnOp

    output_sizes = {}
    output_symbols = []
    for op in func.body:
        if isinstance(op, ReturnOp):
            for i, typ in enumerate(op.types):
                sym = f"output_{i}"
                output_symbols.append(sym)
                output_sizes[sym] = typ.num_elements
            break

    # Determine input data (if function has args, use zeros as default)
    input_data = None
    if func.args:
        input_data = {}
        for i, (_name, ttype) in enumerate(func.args):
            sym = f"input_{i}"
            input_data[sym] = np.zeros(ttype.num_elements, dtype=np.float32)

    results = run_elf(
        elf_path,
        input_data=input_data,
        output_symbols=output_symbols,
        output_sizes=output_sizes,
    )

    for sym in output_symbols:
        print(f"  {sym}: {results[sym]}")
    print(f"  Cycle count: {results['_cycle_count']}")

    # Step 6: Compare
    print("=" * 60)
    print("Step 6: Compare host vs simulator output")
    print("  Host output (from iree-run-module):")
    print(f"  {host_output.strip()}")
    print("  Simulator output:")
    for sym in output_symbols:
        print(f"  {sym}: {results[sym]}")
    print("  (Manual comparison — automated parsing TODO)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SKaiNET -> Coral NPU simulator pipeline"
    )
    subparsers = parser.add_subparsers(dest="command")

    # compile
    p = subparsers.add_parser("compile", help="Compile MLIR to host + rv32 VMFB")
    p.add_argument("input", help="Input MLIR file")

    # verify
    p = subparsers.add_parser(
        "verify", help="Compile for host and run iree-run-module"
    )
    p.add_argument("input", help="Input MLIR file")

    # generate-c
    p = subparsers.add_parser("generate-c", help="Transpile MLIR to C source")
    p.add_argument("input", help="Input MLIR file")
    p.add_argument("-o", "--output", help="Output C file path")

    # build-elf
    p = subparsers.add_parser(
        "build-elf", help="Generate C and build ELF via Bazel"
    )
    p.add_argument("input", help="Input MLIR file")

    # simulate
    p = subparsers.add_parser("simulate", help="Run ELF on MPACT simulator")
    p.add_argument("input", help="Input ELF file")
    p.add_argument(
        "--output-sizes",
        nargs="*",
        help="Output symbol sizes, e.g. output_0=16",
    )

    # run-all
    p = subparsers.add_parser(
        "run-all", help="Full end-to-end pipeline"
    )
    p.add_argument("input", help="Input MLIR file")

    args = parser.parse_args()

    commands = {
        "compile": cmd_compile,
        "verify": cmd_verify,
        "generate-c": cmd_generate_c,
        "build-elf": cmd_build_elf,
        "simulate": cmd_simulate,
        "run-all": cmd_run_all,
    }

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands[args.command](args)


if __name__ == "__main__":
    main()
