#!/usr/bin/env python3
"""Freeze a Moonshine-encoder ONNX to a static batch=1 shape and simplify.

    static_onnx.py <in.onnx> <out.onnx> <frames>

Folds the dynamic pos-id Range/Shape (which the Torq compiler otherwise keeps dynamic →
"incorrect number of dynamic sizes") by fixing the input to [1, frames, feat] and running
onnxsim. Prints the node-count reduction.
"""
import sys
import onnx
import onnxsim


def main():
    in_path, out_path, frames = sys.argv[1], sys.argv[2], int(sys.argv[3])
    m = onnx.load(in_path)
    name = m.graph.input[0].name
    dims = m.graph.input[0].type.tensor_type.shape.dim
    feat = (dims[2].dim_value or 288) if len(dims) >= 3 else 288
    shape = [1, frames, feat]
    ms, ok = onnxsim.simplify(m, overwrite_input_shapes={name: shape})
    assert ok, "onnxsim simplify failed"
    onnx.save(ms, out_path)
    print(f"[static] {name}={shape}  {len(m.graph.node)} -> {len(ms.graph.node)} nodes")


if __name__ == "__main__":
    main()
