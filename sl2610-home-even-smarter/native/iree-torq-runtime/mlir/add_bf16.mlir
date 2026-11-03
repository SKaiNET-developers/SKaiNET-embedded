// R1 smoke graph for the NPU: elementwise add, bf16[1,64]. out = a + b
func.func @main(%a: tensor<1x64xbf16>, %b: tensor<1x64xbf16>) -> tensor<1x64xbf16> {
  %0 = stablehlo.add %a, %b : tensor<1x64xbf16>
  return %0 : tensor<1x64xbf16>
}
