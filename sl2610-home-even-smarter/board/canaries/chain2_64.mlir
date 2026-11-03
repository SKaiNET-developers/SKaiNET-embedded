// Canary: simple NSS execution. Two chained 64x64 matmuls of ones.
// Purpose: proves the NPU actually computes a fused simple graph correctly (not just loads).
//   (ones[64,64] @ ones[64,64]) = 64 everywhere; @ ones = 64*64 = 4096 everywhere.
// Input:  three 64x64xbf16 = 1.0
// Expect: 64x64xbf16 = 4096
module {
  func.func @main(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>, %c: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %1 = stablehlo.dot_general %0, %c, contracting_dims = [1] x [0] : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }
}
