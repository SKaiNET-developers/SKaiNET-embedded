package sk.ainet.embedded.coralnpu.codegen.emitters

import sk.ainet.embedded.coralnpu.codegen.VariableNamer
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphNode

/**
 * Handles `convert` operations (e.g., f16 → f32).
 *
 * On the Coral NPU these are no-ops: the output aliases the input variable.
 * No C code is emitted — only a name alias is registered in the [VariableNamer].
 */
object ConvertEmitter {

    /**
     * Register an alias in [namer] so that references to the convert result
     * resolve to its input variable. Returns `true` if this node was a convert.
     */
    fun handle(node: GraphNode, graph: ComputeGraph, namer: VariableNamer): Boolean {
        if (node.operation.name != "convert") return false

        // The single input node is the operand
        val inputNodes = graph.getInputNodes(node)
        if (inputNodes.isNotEmpty()) {
            namer.alias(node.id, inputNodes[0].id)
        }
        return true
    }
}
