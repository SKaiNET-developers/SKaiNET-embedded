#!/usr/bin/env python3
"""Numerically validate the DSL v2 frontend spec against the vendor ONNX (cos-sim vs onnxruntime).

Reimplements the frontend in numpy — frame(80) → per-frame CMVN(eps=1e-6) → asinh(exp(log_k)·x) → MatMul[80,320]
filterbank → SiLU → conv1d(320→640,k5,s2, weight-norm)+SiLU → conv1d(640→320,k5,s2) → features — and compares to
onnxruntime (zero streaming state = first chunk). RESULT (tiny-streaming): **cos 1.0, maxabsdiff 0.0** — the exact
spec `moonshineV2Frontend` implements. Convs are weight-normalized (resolved as in `bake_moonshine_v2_frontend.py`).

    uv run python validate_moonshine_v2_frontend.py --onnx <frontend.onnx>
"""
import argparse
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


def main(path):
    np.random.seed(1)
    I = {t.name: numpy_helper.to_array(t) for t in onnx.load(path).graph.initializer}
    log_k = I["comp.log_k"]
    W_fb = I["onnx::MatMul_128"]
    eps = 1e-6

    def wnorm(p):
        g0 = I[f"{p}.parametrizations.weight.original0"]
        v = I[f"{p}.parametrizations.weight.original1"]
        return g0 * v / np.sqrt((v ** 2).sum(axis=(1, 2), keepdims=True))

    W1, b1 = wnorm("conv1.conv"), I["conv1.conv.bias"]
    W2, b2 = wnorm("conv2.conv"), I["conv2.conv.bias"]
    silu = lambda x: x * (1 / (1 + np.exp(-x)))

    def conv1d(x, w, b, stride=2):
        Cout, Cin, K = w.shape
        return np.stack([(w * x[None, :, t:t + K]).sum((1, 2)) + b for t in range(0, x.shape[1] - K + 1, stride)], 1)

    def frontend(audio):
        n = len(audio) // 80
        fr = audio[:n * 80].reshape(n, 80).astype(np.float64)
        m = fr.mean(-1, keepdims=True)
        s = fr - m
        normed = s / np.sqrt((s ** 2).mean(-1, keepdims=True) + eps)
        x = silu(np.arcsinh(np.exp(log_k) * normed) @ W_fb)
        c1 = silu(conv1d(np.concatenate([np.zeros((320, 4)), x.T], 1), W1, b1))
        c2 = conv1d(np.concatenate([np.zeros((640, 4)), c1], 1), W2, b2)
        return c2.T

    n = 40
    audio = (np.random.randn(n * 80) * 0.1).astype(np.float32)
    ref = ort.InferenceSession(path).run(None, {
        "audio_chunk": audio[None], "sample_buffer": np.zeros((1, 79), np.float32),
        "sample_len": np.zeros(1, np.int64), "conv1_buffer": np.zeros((1, 320, 4), np.float32),
        "conv2_buffer": np.zeros((1, 640, 4), np.float32), "frame_count": np.zeros(1, np.int64)})[0][0]
    mine = frontend(audio)
    M = min(len(ref), len(mine))
    cos = float((mine[:M].ravel() @ ref[:M].ravel()) / (np.linalg.norm(mine[:M]) * np.linalg.norm(ref[:M])))
    diff = float(np.abs(mine[:M] - ref[:M]).max())
    print(f"cos={cos:.5f}  maxabsdiff={diff:.5f}  shape={ref.shape}")
    ok = cos > 0.999 and diff < 1e-2
    print("VALIDATION:", "PASS (DSL frontend spec reproduces ONNX)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    raise SystemExit(main(ap.parse_args().onnx))
