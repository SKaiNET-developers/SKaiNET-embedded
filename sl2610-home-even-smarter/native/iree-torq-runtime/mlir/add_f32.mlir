// R1 smoke graph: elementwise add + mul, f32[4]. out = (a + b) * a
func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.add %a, %b : tensor<4xf32>
  %1 = stablehlo.multiply %0, %a : tensor<4xf32>
  return %1 : tensor<4xf32>
}
