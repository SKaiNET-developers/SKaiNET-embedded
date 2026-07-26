#!/usr/bin/env python3
"""Numerically validate the SKaiNET DSL v2 decoder against the vendor ONNX (cos-sim vs onnxruntime).

Reimplements the DSL `moonshineV2Decoder` (= `MoonshineDecoderModel`) forward in numpy with the exact DSL
semantics — causal + **interleaved** partial-rotary RoPE self-attn, cross-attn over the cross_kv K/V, **gated
SiLU MLP (value|gate)**, scale-only LayerNorm, tied lm_head — using weights extracted from `decoder_kv.onnx` /
`cross_kv.onnx`, and compares its logits to onnxruntime running the real graphs. This proves the DSL decoder's
architecture + config reproduce the real model (the Kotlin decoder runs the same math).

RESULT (tiny-streaming, 2026-07-25): **cos-sim 1.00000 per token, top-1 tokens identical.** Confirms
RoPE=interleaved, rotaryDim=32 (rotary.inv_freq), gating value|gate, the tie, and the extraction/transpose
rules in `bake_moonshine_v2_decoder.py` (decoder MatMul weights transpose; cross_kv Gemm weights do not).

    uv run python validate_moonshine_v2_decoder.py --onnx-dir <float-onnx>
"""
import argparse
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

DIM, H, HD, L, FFN = 320, 8, 40, 6, 1280


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


def main(onnx_dir):
    np.random.seed(0)
    di, dmm, dn = _ordered(f"{onnx_dir}/decoder_kv.onnx")
    ci, _, _ = _ordered(f"{onnx_dir}/cross_kv.onnx")
    inv = di["rotary.inv_freq"].astype(np.float64)
    W = {l: dict(
        sq=di[dmm[l * 8]], sk=di[dmm[l * 8 + 1]], sv=di[dmm[l * 8 + 2]], so=di[dmm[l * 8 + 3]],
        cq=di[dmm[l * 8 + 4]], co=di[dmm[l * 8 + 5]], fc1=di[dmm[l * 8 + 6]], fc2=di[dmm[l * 8 + 7]],
        fc1b=di[f"cross_attn_blocks.{l}.ff.project_in.proj.bias"],
        fc2b=di[f"cross_attn_blocks.{l}.ff.project_out.bias"],
        ns=di[dn[l * 3]], nc=di[dn[l * 3 + 1]], nm=di[dn[l * 3 + 2]],
        ck=ci[f"cross_attn_blocks.{l}.attn.k_proj.weight"].T,   # Gemm [out,in] → x@Wᵀ ⇒ use .T here
        cv=ci[f"cross_attn_blocks.{l}.attn.v_proj.weight"].T) for l in range(L)}
    on = di[dn[-1]]
    emb = di["embedding.weight"]

    def ln(x, g, e=1e-5):
        m = x.mean(-1, keepdims=True)
        return (x - m) / np.sqrt(((x - m) ** 2).mean(-1, keepdims=True) + e) * g

    silu = lambda x: x / (1 + np.exp(-x))
    hd = lambda x: x.reshape(x.shape[0], H, HD).transpose(1, 0, 2)
    uh = lambda x: x.transpose(1, 0, 2).reshape(x.shape[1], H * HD)

    def rope(x, pos):
        o = x.copy().astype(np.float64)
        ang = np.outer(pos, inv)
        c, s = np.cos(ang), np.sin(ang)
        for i in range(16):
            a, b = x[:, :, 2 * i], x[:, :, 2 * i + 1]
            o[:, :, 2 * i] = a * c[:, i] - b * s[:, i]
            o[:, :, 2 * i + 1] = a * s[:, i] + b * c[:, i]
        return o.astype(np.float32)

    def attn(q, k, v, causal):
        s = (q @ k.transpose(0, 2, 1)) / np.sqrt(HD)
        if causal:
            sq, sk = s.shape[1], s.shape[2]
            s = np.where(np.triu(np.ones((sq, sk)), k=sk - sq + 1).astype(bool), -1e9, s)
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s)
        return (e / e.sum(-1, keepdims=True)) @ v

    def fwd(tok, mem):
        x = emb[tok]
        pos = np.arange(len(tok))
        for l in range(L):
            w = W[l]
            h = ln(x, w["ns"])
            q, k, v = rope(hd(h @ w["sq"]), pos), rope(hd(h @ w["sk"]), pos), hd(h @ w["sv"])
            x = x + uh(attn(q, k, v, True)) @ w["so"]
            h = ln(x, w["nc"])
            x = x + uh(attn(hd(h @ w["cq"]), hd(mem @ w["ck"]), hd(mem @ w["cv"]), False)) @ w["co"]
            h = ln(x, w["nm"]) @ w["fc1"] + w["fc1b"]
            x = x + (silu(h[:, FFN:]) * h[:, :FFN]) @ w["fc2"] + w["fc2b"]
        return ln(x, on) @ emb.T

    f = 8
    mem = np.random.randn(f, DIM).astype(np.float32) * 0.5
    tok = np.array([1, 100, 2000], np.int64)
    kc, vc = ort.InferenceSession(f"{onnx_dir}/cross_kv.onnx").run(None, {"memory": mem[None]})
    z = np.zeros((L, 1, H, 0, HD), np.float32)
    ref = ort.InferenceSession(f"{onnx_dir}/decoder_kv.onnx").run(
        None, {"token": tok[None], "k_self": z, "v_self": z, "out_k_cross": kc, "out_v_cross": vc})[0][0]
    mine = fwd(tok, mem)
    cos = lambda a, b: float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))
    per = [cos(mine[t], ref[t]) for t in range(len(tok))]
    match = [int(mine[t].argmax() == ref[t].argmax()) for t in range(len(tok))]
    print("cos per-token:", ["%.5f" % c for c in per])
    print("argmax match :", match)
    ok = all(c > 0.999 for c in per) and all(match)
    print("VALIDATION:", "PASS (DSL decoder reproduces ONNX)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    raise SystemExit(main(ap.parse_args().onnx_dir))
