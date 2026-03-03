package sk.ainet.embedded.coralnpu.codegen

/**
 * Maps graph node IDs to C variable names.
 *
 * Handles three naming conventions:
 * - Input nodes: `input_0`, `input_1`, ...
 * - Output nodes: `output_0`, `output_1`, ...
 * - Intermediates / constants: node ID stripped of '%' prefix (e.g., `v0`, `v1`)
 *
 * Also supports aliasing for no-op operations like `convert` (f16→f32).
 */
class VariableNamer {
    private val nameMap = mutableMapOf<String, String>()

    /** Register an input node. */
    fun registerInput(nodeId: String, index: Int) {
        nameMap[nodeId] = "input_$index"
    }

    /** Register an output node. */
    fun registerOutput(nodeId: String, index: Int) {
        nameMap[nodeId] = "output_$index"
    }

    /** Register a constant or intermediate node using its ID as the C name. */
    fun registerNode(nodeId: String) {
        if (nodeId !in nameMap) {
            nameMap[nodeId] = nodeId.removePrefix("%")
        }
    }

    /** Create an alias: [aliasId] resolves to whatever [targetId] resolves to. */
    fun alias(aliasId: String, targetId: String) {
        nameMap[aliasId] = resolve(targetId)
    }

    /** Resolve a node ID to its C variable name. */
    fun resolve(nodeId: String): String {
        return nameMap[nodeId] ?: nodeId.removePrefix("%")
    }

    /** Check whether a node ID has been registered as an output. */
    fun isOutput(nodeId: String): Boolean {
        return nameMap[nodeId]?.startsWith("output_") == true
    }
}
