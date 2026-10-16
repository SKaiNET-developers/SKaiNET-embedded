// Canary: EXEC-FORMAT probe. Smallest torq graph — a 1x4 @ 4x4-identity matmul.
// Purpose: does the board runtime LOAD this compiler's executable format at all?
//   A version mismatch fails here with "executable runtime version N does not match expected".
// Input:  1x4xbf16 = [1 2 3 4]   (identity weight is baked)
// Expect: 1x4xbf16 = [1 2 3 4]
module {
  func.func @main(%x: tensor<1x4xbf16>) -> tensor<1x4xbf16> {
    %I = stablehlo.constant dense<[[1.0, 0.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0, 0.0],
                                   [0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0]]> : tensor<4x4xbf16>
    %0 = stablehlo.dot_general %x, %I, contracting_dims = [1] x [0] : (tensor<1x4xbf16>, tensor<4x4xbf16>) -> tensor<1x4xbf16>
    return %0 : tensor<1x4xbf16>
  }
}
