#!/usr/bin/env python3
"""Decompose `onnx.LayerNormalization` nodes into primitive ops (ReduceMean/Sub/Mul/Sqrt/Div/Add).

IREE's torch-onnx path (3.11.0) marks `onnx.LayerNormalization` (opset-17 fused op) illegal to legalize,
so the Moonshine v2 `decoder_kv` graph fails to compile as-is. This surgically replaces each LN node with
its mathematical expansion — leaving every other node untouched — so the graph lowers. Matches the scale-only
(and scale+bias) forms the v2 graphs use.

    uv run python onnx_decompose_layernorm.py <in.onnx> <out.onnx>
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper


def decompose(in_path: str, out_path: str) -> int:
    m = onnx.load(in_path)
    g = m.graph
    inits = list(g.initializer)
    axes_name = "__ln_axes_neg1"
    inits.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), axes_name))

    new_nodes, count = [], 0
    for i, n in enumerate(g.node):
        if n.op_type != "LayerNormalization":
            new_nodes.append(n)
            continue
        count += 1
        X, scale = n.input[0], n.input[1]
        bias = n.input[2] if len(n.input) > 2 and n.input[2] else None
        Y = n.output[0]
        eps = next((a.f for a in n.attribute if a.name == "epsilon"), 1e-5)
        p = f"__ln{i}_"
        eps_name = p + "eps"
        inits.append(numpy_helper.from_array(np.array(eps, dtype=np.float32), eps_name))
        nn = [
            helper.make_node("ReduceMean", [X, axes_name], [p + "mean"], keepdims=1),
            helper.make_node("Sub", [X, p + "mean"], [p + "d"]),
            helper.make_node("Mul", [p + "d", p + "d"], [p + "sq"]),
            helper.make_node("ReduceMean", [p + "sq", axes_name], [p + "var"], keepdims=1),
            helper.make_node("Add", [p + "var", eps_name], [p + "vare"]),
            helper.make_node("Sqrt", [p + "vare"], [p + "std"]),
            helper.make_node("Div", [p + "d", p + "std"], [p + "norm"]),
        ]
        if bias:
            nn.append(helper.make_node("Mul", [p + "norm", scale], [p + "scaled"]))
            nn.append(helper.make_node("Add", [p + "scaled", bias], [Y]))
        else:
            nn.append(helper.make_node("Mul", [p + "norm", scale], [Y]))
        new_nodes.extend(nn)

    del g.node[:]
    g.node.extend(new_nodes)
    del g.initializer[:]
    g.initializer.extend(inits)
    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    return count


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    n = decompose(sys.argv[1], sys.argv[2])
    print(f"decomposed {n} LayerNormalization node(s) -> {sys.argv[2]}")
