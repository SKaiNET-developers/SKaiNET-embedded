package sk.ainet.embedded.coralnpu.codegen.emitters

import sk.ainet.embedded.coralnpu.codegen.VariableNamer
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphNode

/**
 * Emits C code for convolution operations.
 *
 * Directly ports `codegen.py:_generate_convolution()`:
 * - **1x1 fast path**: kh=kw=1, stride=[1,1], padding=[[0,0],[0,0]]
 *   - Single output channel → 2-loop nest (spatial x input channels)
 *   - Multiple output channels → 3-loop nest (oc x spatial x ic)
 * - **General path**: 7-deep loop nest with stride/dilation index computation
 *
 * Memory layout: NCHW row-major, index = `((n*C+c)*H+h)*W+w`
 */
object ConvolutionEmitter {

    fun canEmit(node: GraphNode): Boolean = node.operation.name == "convolution"

    /**
     * Emit C code for a convolution node.
     *
     * Expects:
     * - 2 input nodes: lhs (activation) and rhs (kernel)
     * - lhs shape: [N, C_IN, IH, IW]  (NCHW)
     * - rhs shape: [C_OUT, C_IN, KH, KW]  (OIHW)
     * - node.operation.parameters: strides, padding, rhs_dilate
     */
    fun emit(node: GraphNode, graph: ComputeGraph, namer: VariableNamer): List<String> {
        val inputNodes = graph.getInputNodes(node)
        require(inputNodes.size == 2) { "Convolution requires 2 inputs, got ${inputNodes.size}" }

        val lhsNode = inputNodes[0]
        val rhsNode = inputNodes[1]
        val lhsName = namer.resolve(lhsNode.id)
        val rhsName = namer.resolve(rhsNode.id)
        val outName = namer.resolve(node.id)

        // Extract shapes
        val lhsShape = lhsNode.outputs.first().shape
            ?: throw IllegalStateException("LHS shape unknown for convolution ${node.id}")
        val rhsShape = rhsNode.outputs.first().shape
            ?: throw IllegalStateException("RHS shape unknown for convolution ${node.id}")
        val outShape = node.outputs.first().shape
            ?: throw IllegalStateException("Output shape unknown for convolution ${node.id}")

        // NCHW layout
        val n = lhsShape[0]
        val cIn = lhsShape[1]
        val ih = lhsShape[2]
        val iw = lhsShape[3]
        val cOut = rhsShape[0]
        val kh = rhsShape[2]
        val kw = rhsShape[3]
        val oh = outShape[2]
        val ow = outShape[3]

        // Parameters (with defaults matching StableHLO conventions)
        val strides = extractIntList(node.operation.parameters["strides"]) ?: listOf(1, 1)
        val padding = extractPadding(node.operation.parameters["padding"]) ?: listOf(listOf(0, 0), listOf(0, 0))
        val rhsDilate = extractIntList(node.operation.parameters["rhs_dilate"]) ?: listOf(1, 1)

        val strideH = strides[0]
        val strideW = strides[1]
        val dilH = rhsDilate[0]
        val dilW = rhsDilate[1]

        val is1x1 = kh == 1 && kw == 1
            && strideH == 1 && strideW == 1
            && padding == listOf(listOf(0, 0), listOf(0, 0))

        val hw = oh * ow
        val numElements = outShape.fold(1) { acc, d -> acc * d }
        val lines = mutableListOf<String>()

        // Declare intermediate array if not an output
        if (!namer.isOutput(node.id)) {
            lines.add("  float $outName[$numElements];")
        }

        if (is1x1) {
            lines.add("  // 1x1 convolution: $cIn input channels -> $cOut output channels")
            if (cOut == 1) {
                // Single output channel: 2-loop nest
                lines.add("  for (int i = 0; i < $hw; i++) {")
                lines.add("    float sum = 0.0f;")
                lines.add("    for (int c = 0; c < $cIn; c++) {")
                lines.add("      sum += ${lhsName}[c * $hw + i] * ${rhsName}[c];")
                lines.add("    }")
                lines.add("    ${outName}[i] = sum;")
                lines.add("  }")
            } else {
                // Multiple output channels: 3-loop nest
                lines.add("  for (int oc = 0; oc < $cOut; oc++) {")
                lines.add("    for (int i = 0; i < $hw; i++) {")
                lines.add("      float sum = 0.0f;")
                lines.add("      for (int ic = 0; ic < $cIn; ic++) {")
                lines.add("        sum += ${lhsName}[ic * $hw + i] * ${rhsName}[oc * $cIn + ic];")
                lines.add("      }")
                lines.add("      ${outName}[oc * $hw + i] = sum;")
                lines.add("    }")
                lines.add("  }")
            }
        } else {
            // General convolution: 7-deep loop nest
            lines.add("  // General convolution: [$n,$cIn,$ih,$iw] * [$cOut,$cIn,$kh,$kw] -> [$n,$cOut,$oh,$ow]")
            lines.add("  for (int n_idx = 0; n_idx < $n; n_idx++) {")
            lines.add("    for (int oc = 0; oc < $cOut; oc++) {")
            lines.add("      for (int oh_idx = 0; oh_idx < $oh; oh_idx++) {")
            lines.add("        for (int ow_idx = 0; ow_idx < $ow; ow_idx++) {")
            lines.add("          float sum = 0.0f;")
            lines.add("          for (int ic = 0; ic < $cIn; ic++) {")
            lines.add("            for (int kh_idx = 0; kh_idx < $kh; kh_idx++) {")
            lines.add("              for (int kw_idx = 0; kw_idx < $kw; kw_idx++) {")
            lines.add("                int ih_idx = oh_idx * $strideH + kh_idx * $dilH;")
            lines.add("                int iw_idx = ow_idx * $strideW + kw_idx * $dilW;")
            lines.add("                sum += ${lhsName}[((n_idx * $cIn + ic) * $ih + ih_idx) * $iw + iw_idx]")
            lines.add("                     * ${rhsName}[((oc * $cIn + ic) * $kh + kh_idx) * $kw + kw_idx];")
            lines.add("              }")
            lines.add("            }")
            lines.add("          }")
            lines.add("          ${outName}[((n_idx * $cOut + oc) * $oh + oh_idx) * $ow + ow_idx] = sum;")
            lines.add("        }")
            lines.add("      }")
            lines.add("    }")
            lines.add("  }")
        }

        return lines
    }

    @Suppress("UNCHECKED_CAST")
    private fun extractIntList(value: Any?): List<Int>? {
        return when (value) {
            is List<*> -> (value as? List<Number>)?.map { it.toInt() }
            is IntArray -> value.toList()
            else -> null
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun extractPadding(value: Any?): List<List<Int>>? {
        if (value == null) return null
        return when (value) {
            is List<*> -> {
                (value as? List<List<Number>>)?.map { inner -> inner.map { it.toInt() } }
            }
            else -> null
        }
    }
}
