package sk.ainet.embedded.coralnpu.codegen.emitters

import sk.ainet.embedded.coralnpu.codegen.VariableNamer
import sk.ainet.lang.graph.GraphNode

/**
 * Emits `static const float vN[K] = {0.2989f, ...};` for constant nodes.
 */
object ConstantEmitter {

    /**
     * Emit a top-level constant declaration.
     * Returns the C source line, or null if this node is not a constant.
     */
    fun emit(node: GraphNode, namer: VariableNamer): String? {
        if (node.operation.name != "constant") return null

        val cName = namer.resolve(node.id)
        val values = extractValues(node) ?: return null
        val numElements = values.size
        val formatted = values.joinToString(", ") { formatFloat(it) }

        return "static const float $cName[$numElements] = {$formatted};"
    }

    @Suppress("UNCHECKED_CAST")
    fun extractValues(node: GraphNode): List<Float>? {
        val raw = node.operation.parameters["values"] ?: return null
        return when (raw) {
            is List<*> -> (raw as? List<Number>)?.map { it.toFloat() }
            is FloatArray -> raw.toList()
            is DoubleArray -> raw.map { it.toFloat() }
            else -> null
        }
    }

    /**
     * Format a float value for C code, matching Python's `_format_float`.
     *
     * Produces strings like `0.2989f`, `1.0f`, `3.14f`.
     */
    fun formatFloat(v: Float): String {
        val s = v.toString()
        val withDot = if ('.' !in s && 'e' !in s.lowercase() && 'E' !in s) "${s}.0" else s
        return "${withDot}f"
    }
}
