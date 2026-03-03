package sk.ainet.embedded.coralnpu.codegen

import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertTrue

class CoralNpuCodeGeneratorTest {

    private fun spec(name: String, shape: List<Int>, dtype: String = "float32") =
        TensorSpec(name = name, shape = shape, dtype = dtype)

    @Test
    fun emptyGraphProducesMinimalMain() {
        val graph = DefaultComputeGraph()
        // With no nodes, we can't really generate useful code, but it shouldn't crash.
        // At minimum we'd get an empty graph with no topo order.
        val gen = CoralNpuCodeGenerator()
        val code = gen.generate(graph, "empty")
        assertContains(code, "int main()")
        assertContains(code, "return 0;")
    }

    @Test
    fun binaryAddGeneratesElementwiseLoop() {
        val graph = DefaultComputeGraph()

        val inputA = graph.addNode(GraphNode(
            id = "input_a",
            operation = GenericOperation("input"),
            inputs = emptyList(),
            outputs = listOf(spec("a", listOf(1, 4)))
        ))
        val inputB = graph.addNode(GraphNode(
            id = "input_b",
            operation = GenericOperation("input"),
            inputs = emptyList(),
            outputs = listOf(spec("b", listOf(1, 4)))
        ))
        val add = graph.addNode(GraphNode(
            id = "add_out",
            operation = GenericOperation("add"),
            inputs = listOf(spec("a", listOf(1, 4)), spec("b", listOf(1, 4))),
            outputs = listOf(spec("out", listOf(1, 4)))
        ))

        graph.addEdge(GraphEdge("e1", inputA, add, tensorSpec = spec("a", listOf(1, 4))))
        graph.addEdge(GraphEdge("e2", inputB, add, destinationInputIndex = 1, tensorSpec = spec("b", listOf(1, 4))))

        val gen = CoralNpuCodeGenerator()
        val code = gen.generate(graph, "test_add")

        assertContains(code, "float input_0[4]")
        assertContains(code, "float input_1[4]")
        assertContains(code, "element-wise add")
        assertContains(code, "for (int i = 0; i < 4; i++)")
        assertContains(code, "input_0[i] + input_1[i]")
    }

    @Test
    fun constantEmitsStaticArray() {
        val graph = DefaultComputeGraph()

        val input = graph.addNode(GraphNode(
            id = "input_a",
            operation = GenericOperation("input"),
            inputs = emptyList(),
            outputs = listOf(spec("a", listOf(3)))
        ))

        val constNode = graph.addNode(GraphNode(
            id = "v0",
            operation = GenericOperation("constant", mapOf("values" to listOf(0.2989f, 0.587f, 0.114f)), "constant"),
            inputs = emptyList(),
            outputs = listOf(spec("weights", listOf(3)))
        ))

        // Binary add consuming both input and constant
        val output = graph.addNode(GraphNode(
            id = "out",
            operation = GenericOperation("add"),
            inputs = listOf(spec("a", listOf(3)), spec("w", listOf(3))),
            outputs = listOf(spec("result", listOf(3)))
        ))
        graph.addEdge(GraphEdge("e1", input, output, tensorSpec = spec("a", listOf(3))))
        graph.addEdge(GraphEdge("e2", constNode, output, destinationInputIndex = 1, tensorSpec = spec("w", listOf(3))))

        val gen = CoralNpuCodeGenerator()
        val code = gen.generate(graph, "test_const")

        assertContains(code, "static const float v0[3]")
        assertContains(code, "0.2989f")
        assertContains(code, "0.587f")
        assertContains(code, "0.114f")
    }
}
