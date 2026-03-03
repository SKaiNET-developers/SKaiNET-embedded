package sk.ainet.embedded.coralnpu.codegen

import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Integration test: builds the rgb2grayscale model as a ComputeGraph,
 * generates C code, and verifies it matches the Python codegen output
 * character-for-character.
 *
 * The Python codegen produces (from rgb2grayscale.mlir):
 * ```c
 * // Generated from StableHLO MLIR function @rgb2grayscale
 * // f16 promoted to f32 (Coral NPU has hardware f32, no f16)
 *
 * float input_0[48] __attribute__((section(".data")));
 * float output_0[16] __attribute__((section(".data")));
 *
 * static const float v0[3] = {0.2989f, 0.587f, 0.114f};
 *
 * int main() {
 *   // 1x1 convolution: 3 input channels -> 1 output channels
 *   for (int i = 0; i < 16; i++) {
 *     float sum = 0.0f;
 *     for (int c = 0; c < 3; c++) {
 *       sum += input_0[c * 16 + i] * v0[c];
 *     }
 *     output_0[i] = sum;
 *   }
 *   return 0;
 * }
 * ```
 */
class Rgb2GrayscaleTest {

    private fun spec(name: String, shape: List<Int>) =
        TensorSpec(name = name, shape = shape, dtype = "float32")

    /**
     * Build the rgb2grayscale ComputeGraph equivalent to the MLIR:
     *
     * ```
     * func.func @rgb2grayscale(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x1x4x4xf32> {
     *   %v0 = stablehlo.constant dense<[[[[0.2989]], [[0.587]], [[0.114]]]]> : tensor<1x3x1x1xf32>
     *   %v1 = stablehlo.convolution(%arg0, %v0) ... : tensor<1x1x4x4xf32>
     *   return %v1
     * }
     * ```
     */
    private fun buildRgb2GrayscaleGraph(): DefaultComputeGraph {
        val graph = DefaultComputeGraph()

        // %arg0: input tensor [1, 3, 4, 4]
        val input = graph.addNode(GraphNode(
            id = "arg0",
            operation = GenericOperation("input"),
            inputs = emptyList(),
            outputs = listOf(spec("arg0", listOf(1, 3, 4, 4)))
        ))

        // %v0: constant weights [1, 3, 1, 1] = [0.2989, 0.587, 0.114]
        val weights = graph.addNode(GraphNode(
            id = "v0",
            operation = GenericOperation("constant", mapOf(
                "values" to listOf(0.2989f, 0.587f, 0.114f)
            ), "constant"),
            inputs = emptyList(),
            outputs = listOf(spec("v0", listOf(1, 3, 1, 1)))
        ))

        // %v1: convolution output [1, 1, 4, 4]
        val conv = graph.addNode(GraphNode(
            id = "v1",
            operation = GenericOperation("convolution", mapOf(
                "strides" to listOf(1, 1),
                "padding" to listOf(listOf(0, 0), listOf(0, 0)),
                "rhs_dilate" to listOf(1, 1),
                "batch_group_count" to 1,
                "feature_group_count" to 1
            )),
            inputs = listOf(spec("lhs", listOf(1, 3, 4, 4)), spec("rhs", listOf(1, 3, 1, 1))),
            outputs = listOf(spec("v1", listOf(1, 1, 4, 4)))
        ))

        graph.addEdge(GraphEdge("e1", input, conv, tensorSpec = spec("activation", listOf(1, 3, 4, 4))))
        graph.addEdge(GraphEdge("e2", weights, conv, destinationInputIndex = 1, tensorSpec = spec("kernel", listOf(1, 3, 1, 1))))

        return graph
    }

    @Test
    fun generatedCodeMatchesPythonCodegen() {
        val graph = buildRgb2GrayscaleGraph()
        val gen = CoralNpuCodeGenerator()
        val actual = gen.generate(graph, "rgb2grayscale")

        val expected = buildString {
            appendLine("// Generated from StableHLO MLIR function @rgb2grayscale")
            appendLine("// f16 promoted to f32 (Coral NPU has hardware f32, no f16)")
            appendLine()
            appendLine("""float input_0[48] __attribute__((section(".data")));""")
            appendLine("""float output_0[16] __attribute__((section(".data")));""")
            appendLine()
            appendLine("static const float v0[3] = {0.2989f, 0.587f, 0.114f};")
            appendLine()
            appendLine("int main() {")
            appendLine("  // 1x1 convolution: 3 input channels -> 1 output channels")
            appendLine("  for (int i = 0; i < 16; i++) {")
            appendLine("    float sum = 0.0f;")
            appendLine("    for (int c = 0; c < 3; c++) {")
            appendLine("      sum += input_0[c * 16 + i] * v0[c];")
            appendLine("    }")
            appendLine("    output_0[i] = sum;")
            appendLine("  }")
            appendLine("  return 0;")
            appendLine("}")
            append("")
        }

        assertEquals(expected, actual)
    }
}
