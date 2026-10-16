// Canary: CSS-kernel execution. matmul -> softmax(last dim) -> matmul.
// Purpose: proves the CSS (RISC-V) kernel path (exp/reduce/divide) executes, not just NSS matmuls.
//   With a=(i*j%7)*0.1, b=c=ones: scores row-varies, softmax normalizes to sum 1 across 64 keys,
//   then @ones sums the row -> 64 everywhere (a normalized softmax row sums to 1; times 64 ones = 64).
// Input:  %a 64x64xbf16 (test pattern, supplied by torq-verify.sh), %b %c 64x64xbf16 = 1.0
// Expect: 64x64xbf16 ~= 64   (nonzero, ~constant; a FAIL = CSS kernels not executing)
module {
  func.func @main(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>, %c: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    %ninf = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %z = stablehlo.constant dense<0.0> : tensor<bf16>
    %s = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %mx = stablehlo.reduce(%s init: %ninf) applies stablehlo.maximum across dimensions = [1] : (tensor<64x64xbf16>, tensor<bf16>) -> tensor<64xbf16>
    %mxb = stablehlo.broadcast_in_dim %mx, dims = [0] : (tensor<64xbf16>) -> tensor<64x64xbf16>
    %sub = stablehlo.subtract %s, %mxb : tensor<64x64xbf16>
    %e = stablehlo.exponential %sub : tensor<64x64xbf16>
    %sum = stablehlo.reduce(%e init: %z) applies stablehlo.add across dimensions = [1] : (tensor<64x64xbf16>, tensor<bf16>) -> tensor<64xbf16>
    %sumb = stablehlo.broadcast_in_dim %sum, dims = [0] : (tensor<64xbf16>) -> tensor<64x64xbf16>
    %sm = stablehlo.divide %e, %sumb : tensor<64x64xbf16>
    %o = stablehlo.dot_general %sm, %c, contracting_dims = [1] x [0] : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %o : tensor<64x64xbf16>
  }
}
