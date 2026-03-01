"""Drives MPACT simulator via bazel run."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile

import numpy as np

CORALNPU_ROOT = os.environ.get(
    "CORALNPU_ROOT", "/home/miso/projects/coral/coralnpu"
)


def run_elf(
    elf_path: str,
    input_data: dict[str, np.ndarray] | None = None,
    output_symbols: list[str] | None = None,
    output_sizes: dict[str, int] | None = None,
) -> dict[str, np.ndarray | int]:
    """Run an ELF on the MPACT simulator via bazel run and read back outputs.

    Args:
        elf_path: Path to the .elf file.
        input_data: Map of symbol name -> float32 numpy array to write
                    before execution. May be None if inputs are baked in.
        output_symbols: List of output symbol names to read back.
                        Defaults to ["output_0"].
        output_sizes: Map of symbol name -> number of float32 elements.
                      Required for each output symbol.

    Returns:
        Dict with output symbol names -> np.ndarray (float32),
        plus "_cycle_count" -> int.
    """
    if output_symbols is None:
        output_symbols = ["output_0"]
    if output_sizes is None:
        output_sizes = {}

    elf_path = os.path.abspath(elf_path)

    cmd = [
        "bazel", "run", "//examples/generated:sim_runner", "--",
        elf_path,
    ]

    for sym in output_symbols:
        n = output_sizes.get(sym)
        if n is None or n == 0:
            raise ValueError(
                f"output_sizes must specify element count for '{sym}'"
            )
        cmd.extend(["--output", f"{sym}={n}"])

    tmp_files = []
    try:
        if input_data:
            for sym, data in input_data.items():
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".npy", delete=False
                )
                np.save(tmp, data.astype(np.float32))
                tmp.close()
                tmp_files.append(tmp.name)
                cmd.extend(["--input-npy", f"{sym}={tmp.name}"])

        result = subprocess.run(
            cmd,
            cwd=CORALNPU_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        for f in tmp_files:
            os.unlink(f)

    # Parse JSON from last line of stdout (bazel run may print other lines)
    stdout_lines = result.stdout.strip().splitlines()
    json_line = stdout_lines[-1]
    raw = json.loads(json_line)

    results: dict[str, np.ndarray | int] = {}
    for sym in output_symbols:
        results[sym] = np.array(raw[sym], dtype=np.float32)
    results["_cycle_count"] = raw["_cycle_count"]
    return results
