#!/usr/bin/env python3
"""Extract Moonshine encoder weights from enc_xformer.onnx → per-tensor f32 .bin files
named for the DSL baker (voicecc.export.DirBinWeightSource + MoonshineWeights.hfNameFor).

    extract_moonshine_onnx_weights.py <enc_xformer.onnx> <out-dir> [--layers 6]

Then:  CHECKPOINT=<out-dir> ./gradlew moonshineEncoderMlir -PuseLocalStack=true

Mapping (validated): projection weights are anonymous initializers (`onnx::MatMul_*`) reached
via the named MatMul nodes `/layers.L/self_attn/{q,k,v,dense}_proj/MatMul` + `/mlp/{fc1,fc2}/MatMul`;
LN weights + fc biases are named initializers. ONNX uses input_layernorm / post_attention_layernorm /
dense — we emit the HF names the baker expects (self_attn_layer_norm / final_layer_norm / o_proj).
Moonshine LN is unbiased, so the LN bias tensors are written as zeros. The baker applies the
`onnxᵀ` transpose for the 2-D weights, so we write the ONNX tensors as-is.
"""
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

DIM = 288


def main():
    onnx_path, out = sys.argv[1], sys.argv[2]
    layers = int(sys.argv[sys.argv.index("--layers") + 1]) if "--layers" in sys.argv else 6
    os.makedirs(out, exist_ok=True)

    m = onnx.load(onnx_path)
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    # named MatMul/Gemm node -> its weight-initializer array
    node_w = {}
    for n in m.graph.node:
        if n.op_type in ("MatMul", "Gemm"):
            for x in n.input:
                if x in inits:
                    node_w[n.name] = inits[x]
                    break

    def write(name, arr):
        np.asarray(arr).astype("<f4").tofile(os.path.join(out, name + ".bin"))

    for L in range(layers):
        for dsl, onx in [("q_proj", "q_proj"), ("k_proj", "k_proj"), ("v_proj", "v_proj"), ("o_proj", "dense")]:
            write(f"encoder.layers.{L}.self_attn.{dsl}.weight", node_w[f"/layers.{L}/self_attn/{onx}/MatMul"])
        write(f"encoder.layers.{L}.fc1.weight", node_w[f"/layers.{L}/mlp/fc1/MatMul"])
        write(f"encoder.layers.{L}.fc2.weight", node_w[f"/layers.{L}/mlp/fc2/MatMul"])
        write(f"encoder.layers.{L}.self_attn_layer_norm.weight", inits[f"layers.{L}.input_layernorm.weight"])
        write(f"encoder.layers.{L}.final_layer_norm.weight", inits[f"layers.{L}.post_attention_layernorm.weight"])
        write(f"encoder.layers.{L}.fc1.bias", inits[f"layers.{L}.mlp.fc1.bias"])
        write(f"encoder.layers.{L}.fc2.bias", inits[f"layers.{L}.mlp.fc2.bias"])
        write(f"encoder.layers.{L}.self_attn_layer_norm.bias", np.zeros(DIM, "f4"))
        write(f"encoder.layers.{L}.final_layer_norm.bias", np.zeros(DIM, "f4"))
    write("encoder.layer_norm.weight", inits["layer_norm.weight"])
    write("encoder.layer_norm.bias", np.zeros(DIM, "f4"))

    print(f"wrote {len(os.listdir(out))} .bin files to {out}")


if __name__ == "__main__":
    main()
