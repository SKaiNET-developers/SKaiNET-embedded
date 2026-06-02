"""Build f16 weights + an f16-global MLIR from the f32 baked artifacts, to halve
the board .irpa (1.74GB -> ~0.87GB). (1) rewrite gemma.safetensors f32->f16;
(2) rewrite gemma-argmax.mlir so every weight util.global is f16 and each load is
followed by stablehlo.convert f16->f32 (compute stays f32; board has no EXT_F16
but the convert lowers to int/f32 ops, proven on-device).
"""
import json, struct, re, sys
import numpy as np

B = "/home/miso/projects/coral/build-mlir"
MLIR_IN = sys.argv[1] if len(sys.argv) > 1 else f"{B}/gemma-argmax.mlir"
MLIR_OUT = sys.argv[2] if len(sys.argv) > 2 else f"{B}/gemma-argmax-f16.mlir"

# --- (1) safetensors f32 -> f16 ---
with open(f"{B}/gemma.safetensors", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    base = 8 + n
    items = [(k, v) for k, v in hdr.items() if k != "__metadata__"]
    new_hdr = {}
    off = 0
    blobs = []
    for k, v in items:
        s, e = v["data_offsets"]
        f.seek(base + s)
        a = np.frombuffer(f.read(e - s), dtype=np.float32).astype(np.float16)
        b = a.tobytes()
        new_hdr[k] = {"dtype": "F16", "shape": v["shape"], "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
hb = json.dumps(new_hdr).encode()
with open(f"{B}/gemma-f16.safetensors", "wb") as o:
    o.write(struct.pack("<Q", len(hb)))
    o.write(hb)
    for b in blobs:
        o.write(b)
print(f"wrote gemma-f16.safetensors ({len(items)} tensors, data {off//(1024*1024)}MiB)")

# --- (2) MLIR: f32 weight globals -> f16 + convert-on-load ---
mlir = open(MLIR_IN).read()

# global decls: only parameter.named ones, switch their elem type f32 -> f16
def glob_sub(m):
    return m.group(1) + "f16>"
mlir = re.sub(r'(util\.global private @\w+ = #flow\.parameter\.named<"[^"]*"::"[^"]*"> : tensor<[0-9x]*x)f32>',
              glob_sub, mlir)

# loads of those globals: load f16, then convert to f32 under the original SSA name
def load_sub(m):
    ssa, shape = m.group(1), m.group(2)
    return (f"{ssa}_h = util.global.load @{m.group(3)} : tensor<{shape}xf16>\n"
            f"    {ssa} = stablehlo.convert {ssa}_h : (tensor<{shape}xf16>) -> tensor<{shape}xf32>")
mlir = re.sub(r'(%\w+) = util\.global\.load @(\w+) : tensor<([0-9x]*)xf32>',
              lambda m: f"{m.group(1)}_h = util.global.load @{m.group(2)} : tensor<{m.group(3)}xf16>\n"
                        f"    {m.group(1)} = stablehlo.convert {m.group(1)}_h : (tensor<{m.group(3)}xf16>) -> tensor<{m.group(3)}xf32>",
              mlir)
open(MLIR_OUT, "w").write(mlir)
print("wrote gemma-argmax-f16.mlir; f16 globals:", mlir.count("xf16>"))
