#!/usr/bin/env python3
"""Bake the Moonshine **v2** DECODER weights (decoder_kv.onnx + cross_kv.onnx) to per-tensor f32 .bin.

Companion to `convert_moonshine_v2_weights.py` (encoder + adapter). Maps onto the SKaiNET DSL decoder param
names (`MoonshineDecoderModel` / `moonshineV2Decoder`), NUMERICALLY VALIDATED to cos-sim 1.0 vs onnxruntime
(`validate_moonshine_v2_decoder.py`).

Topology (obfuscated names in decoder_kv → recovered by order; cross_kv keeps semantic names):
  per layer, the 8 ordered 2-D MatMul weights are [self_q, self_k, self_v, self_o, cross_q, cross_o,
  fc1[dim,2·ffn], fc2[ffn,dim]]; the 19 learned LayerNorm scales are [self, cross, mlp]×6 + final; the FF
  biases keep `cross_attn_blocks.N.ff.project_{in.proj,out}.bias`; `embedding.weight` is the tied lm_head;
  cross K/V are `cross_attn_blocks.N.attn.{k,v}_proj.weight` in cross_kv.onnx.

CRITICAL orientation (validated): decoder_kv linears are ONNX **`MatMul` [in,out]** (used as `x@W`) → the DSL's
`linearProject` wants `[out,in]` (`x@Wᵀ`), so **transpose=true**. cross_kv linears are ONNX **`Gemm` [out,in]**
(transB) → already `[out,in]`, so **transpose=false**. Norms/biases 1-D (no transpose); `embedding.weight`
`[vocab,dim]=[out,in]` → transpose=false.
"""
import argparse
import json
import os

import numpy as np
import onnx
from onnx import numpy_helper

DIM = 320


def _ordered(path):
    g = onnx.load(path).graph
    inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    nodes = list(g.node)
    mm, norms = [], []
    for i, n in enumerate(nodes):
        if n.op_type in ("MatMul", "Gemm"):
            w = [x for x in n.input if x in inits and inits[x].ndim == 2]
            if w:
                mm.append(w[0])
        if n.op_type == "LayerNormalization":
            for j in range(i + 1, min(i + 4, len(nodes))):
                if nodes[j].op_type == "Mul":
                    s = [x for x in nodes[j].input if x in inits and inits[x].shape == (DIM,)]
                    if s:
                        norms.append(s[0])
                        break
    return inits, mm, norms


def bake(onnx_dir, out_dir, layers=6):
    di, dmm, dn = _ordered(os.path.join(onnx_dir, "decoder_kv.onnx"))
    ci, _, _ = _ordered(os.path.join(onnx_dir, "cross_kv.onnx"))
    if len(dmm) != layers * 8 or len(dn) != layers * 3 + 1:
        raise SystemExit(f"topology mismatch: matmuls={len(dmm)} norms={len(dn)} (layers={layers})")
    os.makedirs(out_dir, exist_ok=True)
    manifest = []

    def emit(dsl, arr, transpose):
        arr = np.ascontiguousarray(arr, dtype="<f4")
        with open(os.path.join(out_dir, dsl + ".bin"), "wb") as f:
            f.write(arr.reshape(-1).tobytes())
        manifest.append({"dsl": dsl, "shape": list(arr.shape), "transpose": transpose})

    roles = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "cross_attn.q_proj", "cross_attn.o_proj"]
    for l in range(layers):
        b = l * 8
        for r, name in enumerate(roles):                       # decoder MatMul [in,out] → transpose
            emit(f"dec.{l}.{name}.weight", di[dmm[b + r]], True)
        emit(f"dec.{l}.mlp_fc1.weight", di[dmm[b + 6]], True)   # [dim,2ffn] → transpose
        emit(f"dec.{l}.mlp_fc1.bias", di[f"cross_attn_blocks.{l}.ff.project_in.proj.bias"], False)
        emit(f"dec.{l}.mlp_fc2.weight", di[dmm[b + 7]], True)
        emit(f"dec.{l}.mlp_fc2.bias", di[f"cross_attn_blocks.{l}.ff.project_out.bias"], False)
        emit(f"dec.{l}.self_attn_norm.weight", di[dn[l * 3 + 0]], False)
        emit(f"dec.{l}.cross_attn_norm.weight", di[dn[l * 3 + 1]], False)
        emit(f"dec.{l}.mlp_norm.weight", di[dn[l * 3 + 2]], False)
        # cross K/V from cross_kv.onnx — Gemm [out,in] → NO transpose
        emit(f"dec.{l}.cross_attn.k_proj.weight", ci[f"cross_attn_blocks.{l}.attn.k_proj.weight"], False)
        emit(f"dec.{l}.cross_attn.v_proj.weight", ci[f"cross_attn_blocks.{l}.attn.v_proj.weight"], False)
    emit("dec_out_norm.weight", di[dn[-1]], False)
    emit("lm_head.weight", di["embedding.weight"], False)       # tied; [vocab,dim]=[out,in]
    emit("dec_embed.weight", di["embedding.weight"], False)     # host-side token lookup

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"dim": DIM, "layers": layers, "tensors": manifest}, f, indent=2)
    print(f"baked {len(manifest)} decoder tensors -> {out_dir}")
    print(f"  per layer: self q,k,v,o + cross q,k,v,o + mlp fc1/fc2(+b) + 3 norms; + dec_out_norm + lm_head/embed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True, help="dir with decoder_kv.onnx + cross_kv.onnx (float ONNX)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=6)
    a = ap.parse_args()
    bake(a.onnx_dir, a.out, a.layers)
