#!/usr/bin/env python3
"""Bake Moonshine **v2** streaming weights (tiny-streaming) into per-tensor f32 .bin files.

Companion to `convert_moonshine_weights.py` (v1). The v2 model ships as ONNX/ORT from
`download.moonshine.ai/model/<arch>-streaming-en/{float,quantized}/`. Obtain the readable
float graphs with (Python via uv, per project convention):

    uv add moonshine-voice requests onnx
    # then fetch the float ONNX graphs (the .ort is an optimized flatbuffer, NOT readable by `onnx`):
    #   https://download.moonshine.ai/model/tiny-streaming-en/float/{encoder,adapter}.onnx
    #   https://download.moonshine.ai/model/tiny-streaming-en/quantized/streaming_config.json

Unlike v1's ONNX, the v2 exporter **obfuscates weight tensor names** (`val_14`, `add_25`, …); only
the FFN biases keep semantic names (`blocks.N.ff.project_*.bias`). So this baker recovers each
tensor's role by **graph topology**, which is fully regular (validated against tiny-streaming):

  per encoder block (6 total), in node order:
    LayerNorm(shared unit scale) -> Mul(learned scale add_N)   # attn_norm  (scale-only, NO bias)
    MatMul(q)[in,out] MatMul(k) MatMul(v)                       # attention: bias-free, position-free
    ... attention (scalar 1/sqrt(d) Mul, no RoPE) ...
    MatMul(o)
    LayerNorm(shared unit scale) -> Mul(learned scale add_N)   # ffn_norm
    MatMul(ff_up)[dim,ffn] Add(blocks.N.ff.project_in.0.bias)
    Gelu
    MatMul(ff_down)[ffn,dim] Add(blocks.N.ff.project_out.bias)
  final: LayerNorm -> Mul(learned scale)                       # enc_out_norm
  => 36 MatMul weights (6 x [q,k,v,o,ff_up,ff_down]) + 13 norm scales (6*2 + enc_out) + 12 FF biases.

Adapter (validated): op sequence is Gather + Add ONLY — `memory = encoded + pos_embed[offset:offset+seq]`.
There is **no LayerNorm** in the real adapter (the sole learned tensor is `pos_embed.weight [maxFrames,dim]`).

Output layout mirrors v1: `$out/<dsl-name>.bin` little-endian f32, row-major, in the tensor's RAW
ONNX orientation. Linear weights are ONNX `[in,out]` and must be transposed to the DSL's `[out,in]`
`linearProject` layout at load time — `manifest.json` records `transpose:true` for those (same
convention as the v1 encoder mapper). LayerNorm bias is absent (scale-only) → the Kotlin loader
synthesizes zeros, exactly as the v1 decoder mapper does for Moonshine's bias-free norms.
"""
import argparse
import json
import os
import struct
import sys

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    sys.exit("need onnx: `uv add onnx` (run under uv)")


def _write_bin(path, arr):
    flat = arr.reshape(-1).astype("<f4")
    with open(path, "wb") as f:
        f.write(flat.tobytes())
    return list(arr.shape)


def bake_encoder(model, out_dir, dim, layers):
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    dims = {n: list(a.shape) for n, a in inits.items()}
    nodes = list(model.graph.node)

    # ordered role recovery (see module docstring)
    mm = []       # 2D MatMul weights, in node order
    norms = []    # learned LN scales ([dim] Mul right after a LayerNormalization), in node order
    ffbias = {}   # semantic FF biases by name
    for idx, n in enumerate(nodes):
        ii = [x for x in n.input if x in inits]
        if n.op_type == "MatMul":
            w = [x for x in ii if len(dims[x]) == 2]
            if w:
                mm.append(w[0])
        elif n.op_type == "LayerNormalization":
            for j in range(idx + 1, min(idx + 4, len(nodes))):
                if nodes[j].op_type == "Mul":
                    s = [x for x in nodes[j].input if x in inits and dims[x] == [dim]]
                    if s:
                        norms.append(s[0])
                        break
        elif n.op_type == "Add":
            for x in ii:
                if x.startswith("blocks."):
                    ffbias[x] = x

    exp_mm, exp_norm = layers * 6, layers * 2 + 1
    if len(mm) != exp_mm or len(norms) != exp_norm:
        sys.exit(f"topology mismatch: matmuls={len(mm)} (exp {exp_mm}), "
                 f"norms={len(norms)} (exp {exp_norm}) — model layout changed, re-inspect")

    manifest = []

    def emit(dsl, src, transpose):
        shp = _write_bin(os.path.join(out_dir, dsl + ".bin"), inits[src])
        manifest.append({"dsl": dsl, "src": src, "shape": shp, "transpose": transpose})

    for l in range(layers):
        q, k, v, o, ff_up, ff_down = mm[l * 6:l * 6 + 6]
        emit(f"enc.{l}.attn_norm.weight", norms[l * 2], False)
        emit(f"enc.{l}.attn.q.weight", q, True)
        emit(f"enc.{l}.attn.k.weight", k, True)
        emit(f"enc.{l}.attn.v.weight", v, True)
        emit(f"enc.{l}.attn.o.weight", o, True)
        emit(f"enc.{l}.ffn_norm.weight", norms[l * 2 + 1], False)
        emit(f"enc.{l}.ffn_up.weight", ff_up, True)
        emit(f"enc.{l}.ffn_up.bias", f"blocks.{l}.ff.project_in.0.bias", False)
        emit(f"enc.{l}.ffn_down.weight", ff_down, True)
        emit(f"enc.{l}.ffn_down.bias", f"blocks.{l}.ff.project_out.bias", False)
    emit("enc_out_norm.weight", norms[-1], False)
    return manifest


def bake_adapter(model, out_dir):
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    if "pos_embed.weight" not in inits:
        sys.exit("adapter: pos_embed.weight not found — layout changed")
    ops = [n.op_type for n in model.graph.node]
    if "LayerNormalization" in ops:
        print("  WARNING: adapter has a LayerNormalization — MoonshineV2Adapter's norm may be real "
              "after all; re-check. (tiny-streaming had none.)", file=sys.stderr)
    shp = _write_bin(os.path.join(out_dir, "v2_adapter.pos_embed.weight.bin"), inits["pos_embed.weight"])
    return [{"dsl": "v2_adapter.pos_embed.weight", "src": "pos_embed.weight",
             "shape": shp, "transpose": False}]


def main():
    ap = argparse.ArgumentParser(description="Bake Moonshine v2 tiny-streaming weights to f32 .bin")
    ap.add_argument("--onnx-dir", required=True, help="dir with float encoder.onnx + adapter.onnx")
    ap.add_argument("--out", required=True, help="output dir for <dsl-name>.bin + manifest.json")
    ap.add_argument("--dim", type=int, default=320, help="encoder_dim (tiny=320)")
    ap.add_argument("--layers", type=int, default=6, help="encoder depth (tiny=6)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    enc = onnx.load(os.path.join(args.onnx_dir, "encoder.onnx"))
    ada = onnx.load(os.path.join(args.onnx_dir, "adapter.onnx"))

    manifest = bake_encoder(enc, args.out, args.dim, args.layers)
    manifest += bake_adapter(ada, args.out)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"dim": args.dim, "layers": args.layers, "tensors": manifest}, f, indent=2)
    total = sum(len(m["shape"]) and 1 for m in manifest)
    print(f"baked {total} tensors -> {args.out}")
    print(f"  encoder: {args.layers} blocks x (attn_norm,q,k,v,o,ffn_norm,ff_up+b,ff_down+b) + enc_out_norm")
    print(f"  adapter: pos_embed.weight {next(m['shape'] for m in manifest if m['dsl']=='v2_adapter.pos_embed.weight')}")


if __name__ == "__main__":
    main()
