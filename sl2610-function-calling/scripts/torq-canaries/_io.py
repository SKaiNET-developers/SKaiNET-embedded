#!/usr/bin/env python3
"""Canary I/O + assertions for torq-verify.sh.

  gen   <canary> <dir>          write raw-bf16 input bins <dir>/<canary>.in<N>.bin, print the
                                torq-run-module --input specs (one per line: SHAPExDTYPE)
  check <canary> <out.bin>      read raw-bf16 output, assert, print "PASS" or "FAIL: <reason>"

Raw-bin bf16 (little-endian uint16 view) because the board runtime's numpy writer rejects bf16.
"""
import sys, numpy as np, ml_dtypes

BF16 = ml_dtypes.bfloat16

# canary -> (list of input ndarrays, output shape, checker(out_ndarray)->(ok, reason))
def _uniform_ok(o, target, tol=0.06):
    m = float(np.nanmean(o))
    if not np.all(np.isfinite(o)):    return False, f"non-finite (NaN/Inf) in output, mean={m}"
    if np.count_nonzero(o) == 0:      return False, "ALL ZEROS (graph did not execute on device)"
    if abs(m - target) > tol * max(abs(target), 1.0): return False, f"mean {m:.4f} != {target} (tol {tol})"
    return True, f"mean={m:.4f}~{target}"

def _softmax_ok(o):
    m = float(np.nanmean(o))
    if not np.all(np.isfinite(o)):    return False, f"non-finite (CSS exp/reduce not executing?), mean={m}"
    if np.count_nonzero(o) == 0:      return False, "ALL ZEROS (CSS softmax kernels did not execute)"
    sd = float(np.nanstd(o))
    if not (0.5 <= m <= 2.0):         return False, f"mean {m:.4f} outside [0.5,2.0]"
    if m and sd/m > 0.25:             return False, f"non-uniform (std/mean={sd/m:.3f}) — softmax likely wrong"
    return True, f"mean={m:.4f} std={sd:.4f} (uniform, CSS ran)"

def spec(a):
    return "x".join(map(str, a.shape)) + "xbf16"

def inputs(canary):
    if canary == "identity_1x4":
        return [np.array([[1, 2, 3, 4]], dtype=BF16)]
    if canary == "chain2_64":
        return [np.ones((64, 64), dtype=BF16)] * 3
    if canary == "chain8_64":
        return [np.ones((64, 64), dtype=BF16)]
    if canary == "softmax_64":
        a = ((np.arange(4096).reshape(64, 64) % 7) * 0.1).astype(BF16)
        return [a, np.ones((64, 64), dtype=BF16), np.ones((64, 64), dtype=BF16)]
    raise SystemExit(f"unknown canary {canary}")

def out_shape(canary):
    return {"identity_1x4": (1, 4), "chain2_64": (64, 64),
            "chain8_64": (64, 64), "softmax_64": (64, 64)}[canary]

def check(canary, out):
    if canary == "identity_1x4":
        v = out.reshape(-1)[:4]
        return bool(np.allclose(v, [1, 2, 3, 4], atol=0.1)), f"got {v.tolist()} (want [1,2,3,4])"
    if canary == "chain2_64":  return _uniform_ok(out, 4096.0)
    if canary == "chain8_64":  return _uniform_ok(out, 1.0)
    if canary == "softmax_64": return _softmax_ok(out)

def main():
    cmd = sys.argv[1]
    if cmd == "gen":
        canary, d = sys.argv[2], sys.argv[3]
        specs = []
        for i, a in enumerate(inputs(canary)):
            a.view(np.uint16).tofile(f"{d}/{canary}.in{i}.bin")
            specs.append(spec(a))
        print("\n".join(specs))
    elif cmd == "check":
        canary, outbin = sys.argv[2], sys.argv[3]
        raw = np.fromfile(outbin, dtype=np.uint16).view(BF16).astype(np.float32)
        exp = int(np.prod(out_shape(canary)))
        if raw.size < exp:
            print(f"FAIL: short output ({raw.size} < {exp} elems)"); return
        ok, reason = check(canary, raw[:exp].reshape(out_shape(canary)))
        print(("PASS: " if ok else "FAIL: ") + reason)
    else:
        raise SystemExit("usage: _io.py gen|check ...")

if __name__ == "__main__":
    main()
