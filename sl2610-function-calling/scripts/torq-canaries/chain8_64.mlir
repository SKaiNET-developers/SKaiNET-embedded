// Canary: SILENT-ZEROS probe (the important one). Eight chained 64x64 matmuls.
// Purpose: catches the failure where the board loads+runs a multi-dispatch graph but
//   silently returns all zeros. On this SL2610 hardware, <=2 chained matmuls compute
//   correctly but 3+ return zeros — the exact failure that made the encoder useless.
//   Weight is baked 1/64 so every matmul preserves the value (ones stay 1.0).
// Input:  1 tensor 64x64xbf16 = 1.0
// Expect: 64x64xbf16 = 1.0   (a FAIL here = zeros = the hardware won't execute this class of graph)
module {
  func.func @main(%a: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    %w = stablehlo.constant dense<1.562500e-02> : tensor<64x64xbf16>
    %v0 = stablehlo.dot_general %a,  %w, contracting_dims = [1] x [0] : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %v1 = stablehlo.dot_general %v0, %w, contracting_dims = [1] x [0] : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %v2 = stablehlo.dot_general %v1, %w, contracting_dims = [1] x [0] : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %v3 = stablehlo.dot_general %v2, %w, contracting_dims = [1] x [0] : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %v4 = stablehlo.dot_general %v3, %w, contracting_dims = [1] x [0] : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %v5 = stablehlo.dot_general %v4, %w, contracting_dims = [1] x [0] : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %v6 = stablehlo.dot_general %v5, %w, contracting_dims = [1] x [0] : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %v7 = stablehlo.dot_general %v6, %w, contracting_dims = [1] x [0] : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %v7 : tensor<64x64xbf16>
  }
}
