"""Rewrite the seq=24 baked gemma so @gemma returns the PER-POSITION argmax token
ids (tensor<24xi32>) — argmax over vocab at every sequence position. Tiny output
(24 ints) the board can emit; the host decode loop reads the argmax at the last
real position each greedy step. Tie-break to lowest index (numpy/greedy).
"""
import re
B = "/home/miso/projects/coral/build-mlir"
src = open(f"{B}/gemma-baked.mlir").read()
m = re.search(r"return (%v\d+) : tensor<1x24x262153xf32>", src)
assert m, "seq=24 return not found"
logits = m.group(1)
V = 262153
S = 24
tail = f"""    %am_flat = stablehlo.reshape {logits} : (tensor<1x{S}x{V}xf32>) -> tensor<{S}x{V}xf32>
    %am_iota = stablehlo.iota dim = 1 : tensor<{S}x{V}xi32>
    %am_ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %am_zero = stablehlo.constant dense<0> : tensor<i32>
    %am_mv, %am_mi = stablehlo.reduce(%am_flat init: %am_ninf), (%am_iota init: %am_zero) across dimensions = [1] : (tensor<{S}x{V}xf32>, tensor<{S}x{V}xi32>, tensor<f32>, tensor<i32>) -> (tensor<{S}xf32>, tensor<{S}xi32>)
     reducer(%lv: tensor<f32>, %rv: tensor<f32>) (%li: tensor<i32>, %ri: tensor<i32>) {{
       %gt = stablehlo.compare  GT, %lv, %rv : (tensor<f32>, tensor<f32>) -> tensor<i1>
       %eq = stablehlo.compare  EQ, %lv, %rv : (tensor<f32>, tensor<f32>) -> tensor<i1>
       %mn = stablehlo.minimum %li, %ri : tensor<i32>
       %sv = stablehlo.select %gt, %lv, %rv : tensor<i1>, tensor<f32>
       %i1 = stablehlo.select %eq, %mn, %ri : tensor<i1>, tensor<i32>
       %si = stablehlo.select %gt, %li, %i1 : tensor<i1>, tensor<i32>
       stablehlo.return %sv, %si : tensor<f32>, tensor<i32>
     }}
    return %am_mi : tensor<{S}xi32>"""
src = src.replace(f"    return {logits} : tensor<1x{S}x{V}xf32>", tail)
src = src.replace(f"-> (tensor<1x{S}x{V}xf32>) {{", f"-> (tensor<{S}xi32>) {{")
open(f"{B}/gemma-gen.mlir", "w").write(src)
print("wrote gemma-gen.mlir; logits=", logits)
