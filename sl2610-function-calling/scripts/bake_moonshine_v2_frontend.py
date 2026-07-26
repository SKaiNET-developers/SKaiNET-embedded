#!/usr/bin/env python3
"""Bake the Moonshine **v2** frontend weights to per-tensor f32 .bin (weight-norm resolved to plain convs).

The v2 frontend (validated bit-exact by `validate_moonshine_v2_frontend.py`) is:
  frame audio into [N,80] → per-frame CMVN (eps=1e-6) → asinh(exp(log_k)·x) compression →
  MatMul[80,320] filterbank → SiLU → conv1d(320→640,k5,s2)+SiLU → conv1d(640→320,k5,s2) → features[·,320].
The two convs are **weight-normalized** in the ONNX (`parametrizations.weight.original0` = per-out magnitude
`[out,1,1]`, `original1` = direction `[out,in,k]`); this resolves them to a plain conv weight
`w = original0 · original1 / ‖original1‖₂(over in,k)` so the DSL uses a normal `Conv1d`.

Emits DSL-named `.bin` for `moonshineV2Frontend`: `fe_filterbank.weight [80,320]`, `fe_conv1.weight
[640,320,5]`+`.bias`, `fe_conv2.weight [320,640,5]`+`.bias`, and the scalar `fe_log_k`.
"""
import argparse
import json
import os
import numpy as np
import onnx
from onnx import numpy_helper


def bake(onnx_path, out_dir):
    I = {t.name: numpy_helper.to_array(t) for t in onnx.load(onnx_path).graph.initializer}
    os.makedirs(out_dir, exist_ok=True)
    manifest = []

    def emit(name, arr):
        arr = np.ascontiguousarray(arr, dtype="<f4")
        open(os.path.join(out_dir, name + ".bin"), "wb").write(arr.reshape(-1).tobytes())
        manifest.append({"dsl": name, "shape": list(arr.shape)})

    def wnorm(pfx):
        g0 = I[f"{pfx}.parametrizations.weight.original0"]      # [out,1,1]
        v = I[f"{pfx}.parametrizations.weight.original1"]       # [out,in,k]
        return g0 * v / np.sqrt((v ** 2).sum(axis=(1, 2), keepdims=True))

    emit("fe_filterbank.weight", I["onnx::MatMul_128"])         # [80,320]
    emit("fe_conv1.weight", wnorm("conv1.conv"))               # [640,320,5]
    emit("fe_conv1.bias", I["conv1.conv.bias"])
    emit("fe_conv2.weight", wnorm("conv2.conv"))               # [320,640,5]
    emit("fe_conv2.bias", I["conv2.conv.bias"])
    emit("fe_log_k", np.asarray(I["comp.log_k"], np.float32).reshape(1))

    json.dump({"tensors": manifest, "eps": 1e-6, "frame_len": 80}, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    print(f"baked {len(manifest)} frontend tensors -> {out_dir} (weight-norm resolved)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="frontend.onnx (float)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bake(a.onnx, a.out)
