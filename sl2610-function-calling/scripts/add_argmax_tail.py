"""Rewrite gemma-baked.mlir so @gemma returns the last-position argmax token id
(tensor<1xi32>) instead of the full [1,4,262153] logits — a tiny output the
board's iree-run-module can print without OOM. Tie-break to the lowest index
to match numpy.argmax / greedy top_k=1.
"""
import re, sys

src = open("/home/miso/projects/coral/build-mlir/gemma-baked.mlir").read()

# final logits SSA + return line
m = re.search(r"return (%v\d+) : tensor<1x4x262153xf32>", src)
assert m, "return not found"
logits = m.group(1)
VOCAB = 262153

tail = f"""    %am_last = stablehlo.slice {logits} [0:1:1, 3:4:1, 0:{VOCAB}:1] : (tensor<1x4x262153xf32>) -> tensor<1x1x262153xf32>
    %am_flat = stablehlo.reshape %am_last : (tensor<1x1x262153xf32>) -> tensor<{VOCAB}xf32>
    %am_iota = stablehlo.iota dim = 0 : tensor<{VOCAB}xi32>
    %am_ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %am_zero = stablehlo.constant dense<0> : tensor<i32>
    %am_mv, %am_mi = stablehlo.reduce(%am_flat init: %am_ninf), (%am_iota init: %am_zero) across dimensions = [0] : (tensor<{VOCAB}xf32>, tensor<{VOCAB}xi32>, tensor<f32>, tensor<i32>) -> (tensor<f32>, tensor<i32>)
     reducer(%lv: tensor<f32>, %rv: tensor<f32>) (%li: tensor<i32>, %ri: tensor<i32>) {{
       %gt = stablehlo.compare  GT, %lv, %rv : (tensor<f32>, tensor<f32>) -> tensor<i1>
       %eq = stablehlo.compare  EQ, %lv, %rv : (tensor<f32>, tensor<f32>) -> tensor<i1>
       %mn = stablehlo.minimum %li, %ri : tensor<i32>
       %sv = stablehlo.select %gt, %lv, %rv : tensor<i1>, tensor<f32>
       %i1 = stablehlo.select %eq, %mn, %ri : tensor<i1>, tensor<i32>
       %si = stablehlo.select %gt, %li, %i1 : tensor<i1>, tensor<i32>
       stablehlo.return %sv, %si : tensor<f32>, tensor<i32>
     }}
    %am_out = stablehlo.reshape %am_mi : (tensor<i32>) -> tensor<1xi32>
    return %am_out : tensor<1xi32>"""

src = src.replace(f"    return {logits} : tensor<1x4x262153xf32>", tail)
src = src.replace("-> (tensor<1x4x262153xf32>) {", "-> (tensor<1xi32>) {")

open("/home/miso/projects/coral/build-mlir/gemma-argmax.mlir", "w").write(src)
print("wrote gemma-argmax.mlir; logits=", logits)
