"""Writes generated .cc + BUILD.bazel into coralnpu repo, invokes bazel build."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

CORALNPU_ROOT = os.environ.get(
    "CORALNPU_ROOT", "/home/miso/projects/coral/coralnpu"
)

_SIM_RUNNER_PY = '''\
"""Run an ELF on the MPACT simulator and print results as JSON."""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.join(
    __import__("os").path.dirname(__file__), "..", "..", "sw", "coralnpu_sim"))
from coralnpu_v2_sim_utils import CoralNPUV2Simulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("elf", help="Path to .elf file")
    parser.add_argument("--output", action="append", default=[],
                        help="output_name=num_elements (repeatable)")
    parser.add_argument("--input-npy", action="append", default=[],
                        help="symbol=path.npy (repeatable)")
    args = parser.parse_args()

    output_specs = {}
    for spec in args.output:
        sym, n = spec.split("=")
        output_specs[sym] = int(n)

    input_data = {}
    for spec in args.input_npy:
        sym, path = spec.split("=", 1)
        input_data[sym] = np.load(path).astype(np.float32)

    sim = CoralNPUV2Simulator()
    all_symbols = list(output_specs.keys()) + list(input_data.keys())
    seen = set()
    unique = [s for s in all_symbols if not (s in seen or seen.add(s))]

    entry, symbols = sim.get_elf_entry_and_symbol(args.elf, unique)
    sim.load_program(args.elf, entry)

    for sym, data in input_data.items():
        sim.write_memory(symbols[sym], data.view(np.uint8))

    sim.run()
    sim.wait()

    results = {}
    for sym, n in output_specs.items():
        raw = sim.read_memory(symbols[sym], n * 4)
        results[sym] = np.frombuffer(raw, dtype=np.float32).tolist()
    results["_cycle_count"] = sim.get_cycle_count()

    print(json.dumps(results))


if __name__ == "__main__":
    main()
'''


def write_generated_files(name: str, cc_source: str) -> tuple[str, str]:
    """Write .cc, sim_runner.py, and BUILD.bazel to coralnpu/examples/generated/.

    Returns (cc_path, build_path).
    """
    gen_dir = Path(CORALNPU_ROOT) / "examples" / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    cc_path = gen_dir / f"{name}.cc"
    cc_path.write_text(cc_source)

    runner_path = gen_dir / "sim_runner.py"
    runner_path.write_text(_SIM_RUNNER_PY)

    build_path = gen_dir / "BUILD.bazel"
    build_path.write_text(_generate_build_file(gen_dir))

    return str(cc_path), str(build_path)


def _generate_build_file(gen_dir: Path) -> str:
    """Generate BUILD.bazel with targets for all .cc files in the directory."""
    lines = [
        'load("//rules:coralnpu_v2.bzl", "coralnpu_v2_binary")',
        'load("@coralnpu_hw//third_party/python:requirements.bzl", "requirement")',
        "",
        'package(default_visibility = ["//visibility:public"])',
        "",
    ]

    for cc_file in sorted(gen_dir.glob("*.cc")):
        stem = cc_file.stem
        target_name = f"coralnpu_v2_{stem}"
        lines.append(f"coralnpu_v2_binary(")
        lines.append(f'    name = "{target_name}",')
        lines.append(f'    srcs = ["{cc_file.name}"],')
        lines.append(f")")
        lines.append("")

    lines.append("py_binary(")
    lines.append('    name = "sim_runner",')
    lines.append('    srcs = ["sim_runner.py"],')
    lines.append("    deps = [")
    lines.append('        "//sw/coralnpu_sim:coralnpu_v2_sim_utils_lib",')
    lines.append('        requirement("numpy"),')
    lines.append("    ],")
    lines.append(")")
    lines.append("")

    return "\n".join(lines)


def build_elf(name: str) -> str:
    """Build the ELF using bazel. Returns path to the .elf file."""
    target = f"//examples/generated:coralnpu_v2_{name}"
    subprocess.run(
        ["bazel", "build", target],
        cwd=CORALNPU_ROOT,
        check=True,
    )
    # The coralnpu_v2_binary rule uses a platform transition, so the ELF
    # ends up in a config-specific output dir, not the default bazel-bin/.
    # Use `bazel cquery --output=files` to find the real path.
    result = subprocess.run(
        ["bazel", "cquery", target, "--output=files"],
        cwd=CORALNPU_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        if line.strip().endswith(".elf"):
            return os.path.join(CORALNPU_ROOT, line.strip())
    # Fallback to conventional path
    return os.path.join(
        CORALNPU_ROOT,
        "bazel-bin",
        "examples",
        "generated",
        f"coralnpu_v2_{name}.elf",
    )
