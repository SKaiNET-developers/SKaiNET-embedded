"""Regex-based StableHLO MLIR text parser."""

from __future__ import annotations

import re

from ir import (
    BinaryOp,
    ConstantOp,
    ConvertOp,
    ConvolutionOp,
    FuncDef,
    Module,
    Op,
    ReturnOp,
    TensorType,
)


def parse_tensor_type(text: str) -> TensorType:
    """Parse 'tensor<1x3x4x4xf32>' into TensorType."""
    m = re.match(r"tensor<([^>]+)>", text.strip())
    if not m:
        raise ValueError(f"Cannot parse tensor type: {text}")
    parts = m.group(1).split("x")
    element_type = parts[-1]
    shape = [int(d) for d in parts[:-1]]
    return TensorType(shape=shape, element_type=element_type)


def parse_dense_values(text: str) -> list[float]:
    """Extract flat list of float values from a dense literal.

    Handles nested brackets like '[[[[0.2989]], [[0.587]], [[0.114]]]]'.
    Values are extracted in row-major order (which matches the bracket nesting).
    """
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]


def parse_constant(line: str) -> ConstantOp | None:
    """Parse: %v0 = stablehlo.constant dense<...> : tensor<...>"""
    m = re.match(
        r"\s*(%\w+)\s*=\s*stablehlo\.constant\s+dense<(.+)>\s*:\s*(tensor<[^>]+>)",
        line,
    )
    if not m:
        return None
    return ConstantOp(
        result_name=m.group(1),
        values=parse_dense_values(m.group(2)),
        result_type=parse_tensor_type(m.group(3)),
    )


def parse_convert(line: str) -> ConvertOp | None:
    """Parse: %v3 = stablehlo.convert %v1 : (tensor<...>) -> tensor<...>"""
    m = re.match(
        r"\s*(%\w+)\s*=\s*stablehlo\.convert\s+(%\w+)\s*:\s*"
        r"\(([^)]+)\)\s*->\s*(tensor<[^>]+>)",
        line,
    )
    if not m:
        return None
    return ConvertOp(
        result_name=m.group(1),
        operand=m.group(2),
        input_type=parse_tensor_type(m.group(3)),
        result_type=parse_tensor_type(m.group(4)),
    )


def parse_convolution(line: str) -> ConvolutionOp | None:
    """Parse: %v1 = stablehlo.convolution(%arg0, %v0) dim_numbers = ... : (...) -> tensor<...>"""
    m = re.match(
        r"\s*(%\w+)\s*=\s*stablehlo\.convolution\((%\w+),\s*(%\w+)\)",
        line,
    )
    if not m:
        return None

    result_name = m.group(1)
    lhs = m.group(2)
    rhs = m.group(3)

    # Extract strides
    stride_m = re.search(r"stride\s*=\s*\[([^\]]+)\]", line)
    strides = [int(x) for x in stride_m.group(1).split(",")] if stride_m else [1, 1]

    # Extract padding
    pad_m = re.search(r"pad\s*=\s*\[\[([^\]]*)\],\s*\[([^\]]*)\]\]", line)
    if pad_m:
        padding = [
            [int(x) for x in pad_m.group(1).split(",")],
            [int(x) for x in pad_m.group(2).split(",")],
        ]
    else:
        padding = [[0, 0], [0, 0]]

    # Extract rhs_dilate
    dilate_m = re.search(r"rhs_dilate\s*=\s*\[([^\]]+)\]", line)
    rhs_dilate = (
        [int(x) for x in dilate_m.group(1).split(",")] if dilate_m else [1, 1]
    )

    # Extract group counts
    batch_gc_m = re.search(r"batch_group_count\s*=\s*(\d+)", line)
    batch_group_count = int(batch_gc_m.group(1)) if batch_gc_m else 1

    feature_gc_m = re.search(r"feature_group_count\s*=\s*(\d+)", line)
    feature_group_count = int(feature_gc_m.group(1)) if feature_gc_m else 1

    # Extract types: (lhs_type, rhs_type) -> result_type
    type_m = re.search(
        r":\s*\((tensor<[^>]+>),\s*(tensor<[^>]+>)\)\s*->\s*(tensor<[^>]+>)\s*$",
        line,
    )
    if not type_m:
        raise ValueError(f"Cannot parse convolution types: {line}")

    return ConvolutionOp(
        result_name=result_name,
        lhs=lhs,
        rhs=rhs,
        lhs_type=parse_tensor_type(type_m.group(1)),
        rhs_type=parse_tensor_type(type_m.group(2)),
        result_type=parse_tensor_type(type_m.group(3)),
        strides=strides,
        padding=padding,
        rhs_dilate=rhs_dilate,
        batch_group_count=batch_group_count,
        feature_group_count=feature_group_count,
    )


def parse_binary_op(line: str) -> BinaryOp | None:
    """Parse: %v3 = stablehlo.add %v1, %v2 : tensor<...>"""
    m = re.match(
        r"\s*(%\w+)\s*=\s*stablehlo\.(add|multiply|subtract|divide)"
        r"\s+(%\w+),\s*(%\w+)\s*:\s*(tensor<[^>]+>)",
        line,
    )
    if not m:
        return None
    return BinaryOp(
        op=m.group(2),
        result_name=m.group(1),
        lhs=m.group(3),
        rhs=m.group(4),
        result_type=parse_tensor_type(m.group(5)),
    )


def parse_return(line: str) -> ReturnOp | None:
    """Parse: return %v1 : tensor<...>"""
    m = re.match(r"\s*return\s+(.+)", line)
    if not m:
        return None
    rest = m.group(1)
    # Split at ' : ' separating values from types
    colon_idx = rest.index(" : ")
    values_str = rest[:colon_idx]
    types_str = rest[colon_idx + 3 :]

    values = [v.strip() for v in values_str.split(",")]
    types = [parse_tensor_type(t) for t in re.findall(r"tensor<[^>]+>", types_str)]

    return ReturnOp(values=values, types=types)


def parse_func(lines: list[str], start: int) -> tuple[FuncDef, int]:
    """Parse a func.func definition starting at the given line index."""
    header = lines[start]

    m = re.match(
        r"\s*func\.func\s+@(\w+)\(([^)]*)\)\s*->\s*\(([^)]+)\)\s*\{",
        header,
    )
    if not m:
        raise ValueError(f"Cannot parse function header: {header}")

    name = m.group(1)
    args_str = m.group(2).strip()
    returns_str = m.group(3).strip()

    # Parse args
    args: list[tuple[str, TensorType]] = []
    if args_str:
        for arg_match in re.finditer(r"(%\w+):\s*(tensor<[^>]+>)", args_str):
            args.append((arg_match.group(1), parse_tensor_type(arg_match.group(2))))

    # Parse return types
    return_types = [
        parse_tensor_type(t) for t in re.findall(r"tensor<[^>]+>", returns_str)
    ]

    # Parse body
    body: list[Op] = []
    i = start + 1
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped == "}":
            break
        if not stripped or stripped.startswith("//"):
            i += 1
            continue

        op: Op | None = None
        for parser in (
            parse_constant,
            parse_convert,
            parse_convolution,
            parse_binary_op,
            parse_return,
        ):
            op = parser(line)
            if op is not None:
                break

        if op is not None:
            body.append(op)

        i += 1

    return FuncDef(name=name, args=args, return_types=return_types, body=body), i


def parse_module(text: str) -> Module:
    """Parse a StableHLO MLIR module."""
    lines = text.split("\n")
    functions: list[FuncDef] = []

    i = 0
    while i < len(lines):
        if "func.func" in lines[i]:
            func, end_i = parse_func(lines, i)
            functions.append(func)
            i = end_i + 1
        else:
            i += 1

    return Module(functions=functions)
