package sk.ainet.embedded.coralnpu.codegen.emitters

import sk.ainet.embedded.coralnpu.codegen.VariableNamer
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphNode

/**
 * Emits element-wise binary operations: add, multiply, subtract, divide.
 *
 * Generated C (matching Python codegen exactly):
 * ```c
 *   // element-wise add
 *   for (int i = 0; i < N; i++) {
 *     out[i] = lhs[i] + rhs[i];
 *   }
 * ```
 */
object BinaryOpEmitter {

    private val OP_SYMBOLS = mapOf(
        "add" to "+",
        "multiply" to "*",
        "subtract" to "-",
        "divide" to "/"
    )

    fun canEmit(node: GraphNode): Boolean = node.operation.name in OP_SYMBOLS

    /**
     * Emit C code lines for an element-wise binary op.
     *
     * @return list of C source lines (with 2-space indentation for body)
     */
    fun emit(node: GraphNode, graph: ComputeGraph, namer: VariableNamer): List<String> {
        val opName = node.operation.name
        val symbol = OP_SYMBOLS[opName]
            ?: throw IllegalArgumentException("Unsupported binary op: $opName")

        val inputNodes = graph.getInputNodes(node)
        require(inputNodes.size == 2) { "Binary op $opName requires 2 inputs, got ${inputNodes.size}" }

        val lhsName = namer.resolve(inputNodes[0].id)
        val rhsName = namer.resolve(inputNodes[1].id)
        val outName = namer.resolve(node.id)

        val numElements = node.outputs.firstOrNull()?.shape?.fold(1) { acc, d -> acc * d }
            ?: throw IllegalStateException("Cannot determine output size for node ${node.id}")

        val lines = mutableListOf<String>()

        // Declare intermediate array if not an output
        if (!namer.isOutput(node.id)) {
            lines.add("  float $outName[$numElements];")
        }

        lines.add("  // element-wise $opName")
        lines.add("  for (int i = 0; i < $numElements; i++) {")
        lines.add("    $outName[i] = $lhsName[i] $symbol $rhsName[i];")
        lines.add("  }")

        return lines
    }
}
