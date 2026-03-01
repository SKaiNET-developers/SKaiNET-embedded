"""IR -> C code generator for coralnpu_v2_binary."""

from __future__ import annotations

from ir import (
    BinaryOp,
    ConstantOp,
    ConvertOp,
    ConvolutionOp,
    FuncDef,
    Module,
    ReturnOp,
)


def generate_c(module: Module) -> str:
    """Generate C source code from the IR module."""
    if not module.functions:
        raise ValueError("Module has no functions")
    return _generate_func(module.functions[0])


def _format_float(v: float) -> str:
    """Format a float value for C code."""
    s = f"{v}"
    if "." not in s and "e" not in s.lower():
        s += ".0"
    return s + "f"


def _resolve(name: str, name_map: dict[str, str]) -> str:
    """Resolve an SSA name to its C variable name."""
    return name_map.get(name, name.lstrip("%"))


def _result_c_name(
    op_result: str,
    name_map: dict[str, str],
    return_names: dict[str, str],
) -> str:
    """Get the C variable name for an op result."""
    if op_result in return_names:
        c_name = return_names[op_result]
    else:
        c_name = op_result.lstrip("%")
    name_map[op_result] = c_name
    return c_name


def _generate_func(func: FuncDef) -> str:
    lines: list[str] = []
    lines.append(f"// Generated from StableHLO MLIR function @{func.name}")
    lines.append(f"// f16 promoted to f32 (Coral NPU has hardware f32, no f16)")
    lines.append("")

    # Identify return values
    return_op: ReturnOp | None = None
    for op in func.body:
        if isinstance(op, ReturnOp):
            return_op = op
            break

    return_names: dict[str, str] = {}
    if return_op:
        for i, val in enumerate(return_op.values):
            return_names[val] = f"output_{i}"

    # SSA name -> C variable name
    name_map: dict[str, str] = {}

    # Generate input arrays from function args
    for i, (arg_name, arg_type) in enumerate(func.args):
        c_name = f"input_{i}"
        name_map[arg_name] = c_name
        n = arg_type.num_elements
        lines.append(
            f"float {c_name}[{n}] __attribute__((section(\".data\")));"
        )

    # Generate output arrays
    if return_op:
        for i, typ in enumerate(return_op.types):
            c_name = f"output_{i}"
            n = typ.num_elements
            lines.append(
                f"float {c_name}[{n}] __attribute__((section(\".data\")));"
            )

    if func.args or return_op:
        lines.append("")

    # Generate constants
    has_constants = False
    for op in func.body:
        if isinstance(op, ConstantOp):
            has_constants = True
            c_name = op.result_name.lstrip("%")
            name_map[op.result_name] = c_name
            n = op.result_type.num_elements
            vals = ", ".join(_format_float(v) for v in op.values)
            lines.append(f"static const float {c_name}[{n}] = {{{vals}}};")

    if has_constants:
        lines.append("")

    lines.append("int main() {")

    # Generate compute ops
    for op in func.body:
        if isinstance(op, ConstantOp):
            continue  # already handled above
        elif isinstance(op, ConvertOp):
            # f16<->f32 is no-op: alias the operand
            name_map[op.result_name] = _resolve(op.operand, name_map)
        elif isinstance(op, ConvolutionOp):
            _generate_convolution(op, name_map, return_names, lines)
        elif isinstance(op, BinaryOp):
            _generate_binary_op(op, name_map, return_names, lines)
        elif isinstance(op, ReturnOp):
            pass  # output arrays already declared

    lines.append("  return 0;")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def _generate_convolution(
    op: ConvolutionOp,
    name_map: dict[str, str],
    return_names: dict[str, str],
    lines: list[str],
) -> None:
    lhs_name = _resolve(op.lhs, name_map)
    rhs_name = _resolve(op.rhs, name_map)
    out_name = _result_c_name(op.result_name, name_map, return_names)

    # lhs shape: [N, C_IN, IH, IW] (NCHW)
    # rhs shape: [C_OUT, C_IN, KH, KW] (OIHW per dim_numbers [o, i, 0, 1])
    c_in = op.lhs_type.shape[1]
    ih = op.lhs_type.shape[2]
    iw = op.lhs_type.shape[3]
    c_out = op.rhs_type.shape[0]
    kh = op.rhs_type.shape[2]
    kw = op.rhs_type.shape[3]
    oh = op.result_type.shape[2]
    ow = op.result_type.shape[3]
    n = op.lhs_type.shape[0]

    stride_h, stride_w = op.strides[0], op.strides[1]
    dil_h, dil_w = op.rhs_dilate[0], op.rhs_dilate[1]

    is_1x1 = (
        kh == 1
        and kw == 1
        and stride_h == 1
        and stride_w == 1
        and op.padding == [[0, 0], [0, 0]]
    )

    # Declare intermediate array if not an output
    if op.result_name not in return_names:
        lines.append(f"  float {out_name}[{op.result_type.num_elements}];")

    hw = oh * ow

    if is_1x1:
        lines.append(
            f"  // 1x1 convolution: {c_in} input channels -> {c_out} output channels"
        )
        if c_out == 1:
            lines.append(f"  for (int i = 0; i < {hw}; i++) {{")
            lines.append(f"    float sum = 0.0f;")
            lines.append(f"    for (int c = 0; c < {c_in}; c++) {{")
            lines.append(f"      sum += {lhs_name}[c * {hw} + i] * {rhs_name}[c];")
            lines.append(f"    }}")
            lines.append(f"    {out_name}[i] = sum;")
            lines.append(f"  }}")
        else:
            lines.append(f"  for (int oc = 0; oc < {c_out}; oc++) {{")
            lines.append(f"    for (int i = 0; i < {hw}; i++) {{")
            lines.append(f"      float sum = 0.0f;")
            lines.append(f"      for (int ic = 0; ic < {c_in}; ic++) {{")
            lines.append(
                f"        sum += {lhs_name}[ic * {hw} + i]"
                f" * {rhs_name}[oc * {c_in} + ic];"
            )
            lines.append(f"      }}")
            lines.append(f"      {out_name}[oc * {hw} + i] = sum;")
            lines.append(f"    }}")
            lines.append(f"  }}")
    else:
        # General convolution: 7-deep loop nest
        lines.append(
            f"  // General convolution:"
            f" [{n},{c_in},{ih},{iw}] * [{c_out},{c_in},{kh},{kw}]"
            f" -> [{n},{c_out},{oh},{ow}]"
        )
        lines.append(f"  for (int n_idx = 0; n_idx < {n}; n_idx++) {{")
        lines.append(f"    for (int oc = 0; oc < {c_out}; oc++) {{")
        lines.append(f"      for (int oh_idx = 0; oh_idx < {oh}; oh_idx++) {{")
        lines.append(f"        for (int ow_idx = 0; ow_idx < {ow}; ow_idx++) {{")
        lines.append(f"          float sum = 0.0f;")
        lines.append(f"          for (int ic = 0; ic < {c_in}; ic++) {{")
        lines.append(f"            for (int kh_idx = 0; kh_idx < {kh}; kh_idx++) {{")
        lines.append(
            f"              for (int kw_idx = 0; kw_idx < {kw}; kw_idx++) {{"
        )
        lines.append(
            f"                int ih_idx = oh_idx * {stride_h} + kh_idx * {dil_h};"
        )
        lines.append(
            f"                int iw_idx = ow_idx * {stride_w} + kw_idx * {dil_w};"
        )
        lines.append(
            f"                sum += {lhs_name}"
            f"[((n_idx * {c_in} + ic) * {ih} + ih_idx) * {iw} + iw_idx]"
        )
        lines.append(
            f"                     * {rhs_name}"
            f"[((oc * {c_in} + ic) * {kh} + kh_idx) * {kw} + kw_idx];"
        )
        lines.append(f"              }}")
        lines.append(f"            }}")
        lines.append(f"          }}")
        lines.append(
            f"          {out_name}"
            f"[((n_idx * {c_out} + oc) * {oh} + oh_idx) * {ow} + ow_idx] = sum;"
        )
        lines.append(f"        }}")
        lines.append(f"      }}")
        lines.append(f"    }}")
        lines.append(f"  }}")


def _generate_binary_op(
    op: BinaryOp,
    name_map: dict[str, str],
    return_names: dict[str, str],
    lines: list[str],
) -> None:
    lhs_name = _resolve(op.lhs, name_map)
    rhs_name = _resolve(op.rhs, name_map)
    out_name = _result_c_name(op.result_name, name_map, return_names)

    n = op.result_type.num_elements
    op_symbol = {
        "add": "+",
        "multiply": "*",
        "subtract": "-",
        "divide": "/",
    }[op.op]

    if op.result_name not in return_names:
        lines.append(f"  float {out_name}[{n}];")

    lines.append(f"  // element-wise {op.op}")
    lines.append(f"  for (int i = 0; i < {n}; i++) {{")
    lines.append(f"    {out_name}[i] = {lhs_name}[i] {op_symbol} {rhs_name}[i];")
    lines.append(f"  }}")
