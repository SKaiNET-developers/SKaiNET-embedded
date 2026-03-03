package sk.ainet.embedded.coralnpu.codegen

import sk.ainet.embedded.coralnpu.codegen.emitters.BinaryOpEmitter
import sk.ainet.embedded.coralnpu.codegen.emitters.ConstantEmitter
import sk.ainet.embedded.coralnpu.codegen.emitters.ConvertEmitter
import sk.ainet.embedded.coralnpu.codegen.emitters.ConvolutionEmitter
import sk.ainet.lang.graph.ComputeGraph

/**
 * Generates bare-metal C source code for the Coral NPU from a [ComputeGraph].
 *
 * Replicates the output of `codegen.py` character-for-character for the
 * supported operations: constant, convert, convolution, and binary ops.
 *
 * Memory conventions (matching `coralnpu_v2_binary`):
 * | Kind          | Declaration                                              | Placement     |
 * |---------------|----------------------------------------------------------|---------------|
 * | I/O arrays    | `float input_0[N] __attribute__((section(".data")));`    | DTCM (32 KB)  |
 * | Constants     | `static const float v0[K] = {...};`                      | ITCM / flash  |
 * | Intermediates | `float vN[M];` inside `main()`                           | Stack         |
 */
class CoralNpuCodeGenerator(
    private val config: CoralNpuConfig = CoralNpuConfig.DEFAULT,
    private val funcName: String = "main"
) {

    /**
     * Generate C source from the given [graph].
     *
     * @param graph A validated [ComputeGraph] representing one function.
     * @param modelName Name used in header comments (e.g., "rgb2grayscale").
     * @return Complete C source code as a string.
     */
    fun generate(graph: ComputeGraph, modelName: String = funcName): String {
        val lines = mutableListOf<String>()
        val namer = VariableNamer()

        val topoOrder = graph.getTopologicalOrder()

        // Input nodes = nodes with no incoming edges EXCLUDING constants.
        // In StableHLO, function arguments are inputs; constants are separate.
        val allInputNodes = graph.getInputNodes()
        val inputNodes = allInputNodes.filter { it.operation.name != "constant" }
        val outputNodes = graph.getOutputNodes()
        val inputIds = inputNodes.map { it.id }.toSet()

        // Register input names
        inputNodes.forEachIndexed { i, node ->
            namer.registerInput(node.id, i)
        }

        // Register output names.
        outputNodes.forEachIndexed { i, node ->
            namer.registerOutput(node.id, i)
        }

        // Register all other nodes (constants, intermediates)
        for (node in topoOrder) {
            if (!namer.isOutput(node.id) && node.id !in inputIds) {
                namer.registerNode(node.id)
            }
        }

        // 1. Header comment
        for (comment in config.headerComment(modelName)) {
            lines.add(comment)
        }
        lines.add("")

        // 2. Input array declarations
        for (node in inputNodes) {
            val numElements = node.outputs.firstOrNull()?.shape?.fold(1) { acc, d -> acc * d }
                ?: throw IllegalStateException("Cannot determine size for input node ${node.id}")
            val cName = namer.resolve(node.id)
            lines.add("float $cName[$numElements] ${config.dataSectionAttr};")
        }

        // 3. Output array declarations
        for (node in outputNodes) {
            val outputSpec = node.outputs.firstOrNull() ?: node.inputs.firstOrNull()
                ?: throw IllegalStateException("Cannot determine size for output node ${node.id}")
            val numElements = outputSpec.shape?.fold(1) { acc, d -> acc * d }
                ?: throw IllegalStateException("Cannot determine size for output node ${node.id}")
            val cName = namer.resolve(node.id)
            lines.add("float $cName[$numElements] ${config.dataSectionAttr};")
        }

        if (inputNodes.isNotEmpty() || outputNodes.isNotEmpty()) {
            lines.add("")
        }

        // 4. Constant declarations (top-level, before main)
        var hasConstants = false
        for (node in topoOrder) {
            val constLine = ConstantEmitter.emit(node, namer)
            if (constLine != null) {
                hasConstants = true
                lines.add(constLine)
            }
        }
        if (hasConstants) {
            lines.add("")
        }

        // 5. main() function
        lines.add("int main() {")

        // 6. Walk topo order, emit compute ops
        for (node in topoOrder) {
            when {
                node.operation.name == "constant" -> {
                    // Already emitted as global constant, skip
                }
                node.operation.name == "convert" -> {
                    ConvertEmitter.handle(node, graph, namer)
                }
                ConvolutionEmitter.canEmit(node) -> {
                    lines.addAll(ConvolutionEmitter.emit(node, graph, namer))
                }
                BinaryOpEmitter.canEmit(node) -> {
                    lines.addAll(BinaryOpEmitter.emit(node, graph, namer))
                }
                node.id in inputIds -> {
                    // Input node — no code needed
                }
                else -> {
                    lines.add("  // TODO: unsupported operation '${node.operation.name}' (node ${node.id})")
                }
            }
        }

        lines.add("  return 0;")
        lines.add("}")
        lines.add("")

        return lines.joinToString("\n")
    }
}
