package sk.ainet.embedded.coralnpu.codegen

import sk.ainet.embedded.coralnpu.codegen.emitters.ConvolutionEmitter
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertTrue

class ConvolutionEmitterTest {

    private fun spec(name: String, shape: List<Int>) =
        TensorSpec(name = name, shape = shape, dtype = "float32")

    private fun build1x1ConvGraph(): Triple<DefaultComputeGraph, GraphNode, VariableNamer> {
        // rgb2grayscale: 1x1 conv, 3 input channels → 1 output channel
        // lhs: [1, 3, 4, 4], rhs: [1, 3, 1, 1], output: [1, 1, 4, 4]
        val graph = DefaultComputeGraph()

        val input = graph.addNode(GraphNode(
            id = "input_0",
            operation = GenericOperation("input"),
            inputs = emptyList(),
            outputs = listOf(spec("activation", listOf(1, 3, 4, 4)))
        ))

        val kernel = graph.addNode(GraphNode(
            id = "v0",
            operation = GenericOperation("constant", mapOf(
                "values" to listOf(0.2989f, 0.587f, 0.114f)
            ), "constant"),
            inputs = emptyList(),
            outputs = listOf(spec("kernel", listOf(1, 3, 1, 1)))
        ))

        val conv = graph.addNode(GraphNode(
            id = "v1",
            operation = GenericOperation("convolution", mapOf(
                "strides" to listOf(1, 1),
                "padding" to listOf(listOf(0, 0), listOf(0, 0)),
                "rhs_dilate" to listOf(1, 1)
            )),
            inputs = listOf(spec("lhs", listOf(1, 3, 4, 4)), spec("rhs", listOf(1, 3, 1, 1))),
            outputs = listOf(spec("result", listOf(1, 1, 4, 4)))
        ))

        graph.addEdge(GraphEdge("e1", input, conv, tensorSpec = spec("activation", listOf(1, 3, 4, 4))))
        graph.addEdge(GraphEdge("e2", kernel, conv, destinationInputIndex = 1, tensorSpec = spec("kernel", listOf(1, 3, 1, 1))))

        val namer = VariableNamer()
        namer.registerInput(input.id, 0)
        namer.registerOutput(conv.id, 0)
        namer.registerNode(kernel.id)

        return Triple(graph, conv, namer)
    }

    @Test
    fun emits1x1ConvSingleOutputChannel() {
        val (graph, conv, namer) = build1x1ConvGraph()
        val lines = ConvolutionEmitter.emit(conv, graph, namer)
        val code = lines.joinToString("\n")

        assertContains(code, "1x1 convolution: 3 input channels -> 1 output channels")
        assertContains(code, "for (int i = 0; i < 16; i++)")
        assertContains(code, "float sum = 0.0f;")
        assertContains(code, "for (int c = 0; c < 3; c++)")
        assertContains(code, "input_0[c * 16 + i] * v0[c]")
        assertContains(code, "output_0[i] = sum;")
    }

    @Test
    fun emitsGeneralConvWith7LoopNest() {
        val graph = DefaultComputeGraph()

        // 3x3 conv: lhs [1,1,8,8], rhs [1,1,3,3], strides=[2,2], padding=[[1,1],[1,1]]
        val input = graph.addNode(GraphNode(
            id = "input_0",
            operation = GenericOperation("input"),
            inputs = emptyList(),
            outputs = listOf(spec("activation", listOf(1, 1, 8, 8)))
        ))

        val kernel = graph.addNode(GraphNode(
            id = "v0",
            operation = GenericOperation("constant", mapOf(
                "values" to (1..9).map { it.toFloat() }
            ), "constant"),
            inputs = emptyList(),
            outputs = listOf(spec("kernel", listOf(1, 1, 3, 3)))
        ))

        val conv = graph.addNode(GraphNode(
            id = "v1",
            operation = GenericOperation("convolution", mapOf(
                "strides" to listOf(2, 2),
                "padding" to listOf(listOf(1, 1), listOf(1, 1)),
                "rhs_dilate" to listOf(1, 1)
            )),
            inputs = listOf(spec("lhs", listOf(1, 1, 8, 8)), spec("rhs", listOf(1, 1, 3, 3))),
            outputs = listOf(spec("result", listOf(1, 1, 4, 4)))
        ))

        graph.addEdge(GraphEdge("e1", input, conv, tensorSpec = spec("a", listOf(1, 1, 8, 8))))
        graph.addEdge(GraphEdge("e2", kernel, conv, destinationInputIndex = 1, tensorSpec = spec("k", listOf(1, 1, 3, 3))))

        val namer = VariableNamer()
        namer.registerInput(input.id, 0)
        namer.registerOutput(conv.id, 0)
        namer.registerNode(kernel.id)

        val lines = ConvolutionEmitter.emit(conv, graph, namer)
        val code = lines.joinToString("\n")

        assertContains(code, "General convolution:")
        assertContains(code, "for (int n_idx = 0; n_idx < 1; n_idx++)")
        assertContains(code, "for (int oc = 0; oc < 1; oc++)")
        assertContains(code, "for (int oh_idx = 0; oh_idx < 4; oh_idx++)")
        assertContains(code, "for (int ow_idx = 0; ow_idx < 4; ow_idx++)")
        assertContains(code, "for (int ic = 0; ic < 1; ic++)")
        assertContains(code, "for (int kh_idx = 0; kh_idx < 3; kh_idx++)")
        assertContains(code, "for (int kw_idx = 0; kw_idx < 3; kw_idx++)")
        assertContains(code, "oh_idx * 2 + kh_idx * 1")  // stride=2, dilation=1
    }

    @Test
    fun emits1x1ConvMultipleOutputChannels() {
        val graph = DefaultComputeGraph()

        // lhs [1, 3, 4, 4], rhs [2, 3, 1, 1] → output [1, 2, 4, 4]
        val input = graph.addNode(GraphNode(
            id = "input_0",
            operation = GenericOperation("input"),
            inputs = emptyList(),
            outputs = listOf(spec("activation", listOf(1, 3, 4, 4)))
        ))

        val kernel = graph.addNode(GraphNode(
            id = "v0",
            operation = GenericOperation("constant", mapOf(
                "values" to (1..6).map { it.toFloat() }
            ), "constant"),
            inputs = emptyList(),
            outputs = listOf(spec("kernel", listOf(2, 3, 1, 1)))
        ))

        val conv = graph.addNode(GraphNode(
            id = "v1",
            operation = GenericOperation("convolution", mapOf(
                "strides" to listOf(1, 1),
                "padding" to listOf(listOf(0, 0), listOf(0, 0)),
                "rhs_dilate" to listOf(1, 1)
            )),
            inputs = listOf(spec("lhs", listOf(1, 3, 4, 4)), spec("rhs", listOf(2, 3, 1, 1))),
            outputs = listOf(spec("result", listOf(1, 2, 4, 4)))
        ))

        graph.addEdge(GraphEdge("e1", input, conv, tensorSpec = spec("a", listOf(1, 3, 4, 4))))
        graph.addEdge(GraphEdge("e2", kernel, conv, destinationInputIndex = 1, tensorSpec = spec("k", listOf(2, 3, 1, 1))))

        val namer = VariableNamer()
        namer.registerInput(input.id, 0)
        namer.registerOutput(conv.id, 0)
        namer.registerNode(kernel.id)

        val lines = ConvolutionEmitter.emit(conv, graph, namer)
        val code = lines.joinToString("\n")

        assertContains(code, "1x1 convolution: 3 input channels -> 2 output channels")
        assertContains(code, "for (int oc = 0; oc < 2; oc++)")
        assertContains(code, "for (int i = 0; i < 16; i++)")
        assertContains(code, "for (int ic = 0; ic < 3; ic++)")
        assertContains(code, "input_0[ic * 16 + i] * v0[oc * 3 + ic]")
        assertContains(code, "output_0[oc * 16 + i] = sum;")
    }
}
