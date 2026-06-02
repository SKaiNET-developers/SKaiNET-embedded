package voicecc.export

import sk.ainet.compile.hlo.toStableHlo
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.ReluOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType
import java.io.File

/**
 * Phase-2 bridge bring-up (host tooling): build a tiny SKaiNET ComputeGraph
 * (a + b -> relu) and export it to StableHLO MLIR. The MLIR is then fed to
 * `iree-compile --iree-hal-target-device=local ... llvm-cpu` (NEON for aarch64)
 * and run via the IREE CPU runtime — see scripts/iree-compile-cpu.sh.
 *
 * This proves DAG -> StableHLO -> IREE-CPU end-to-end before scaling to the
 * gemma3 graph (which additionally needs gather/RoPE/RMSNorm converters).
 */
fun main(args: Array<String>) {
    val outPath = args.getOrElse(0) { "build/mlir/addrelu.mlir" }
    val shape = listOf(1, 4)

    val graph = DefaultComputeGraph()
    val a = GraphNode("a", InputOperation<DType, Any>(), emptyList(), listOf(TensorSpec("a", shape, "FP32")))
    val b = GraphNode("b", InputOperation<DType, Any>(), emptyList(), listOf(TensorSpec("b", shape, "FP32")))
    val add = GraphNode(
        "add1", AddOperation<DType, Any>(),
        inputs = listOf(TensorSpec("a", shape, "FP32"), TensorSpec("b", shape, "FP32")),
        outputs = listOf(TensorSpec("c", shape, "FP32")),
    )
    val relu = GraphNode(
        "relu1", ReluOperation<DType, Any>(),
        inputs = listOf(TensorSpec("c", shape, "FP32")),
        outputs = listOf(TensorSpec("d", shape, "FP32")),
    )
    graph.addNode(a); graph.addNode(b); graph.addNode(add); graph.addNode(relu)
    graph.addEdge(GraphEdge("e1", a, add, 0, 0, a.outputs[0]))
    graph.addEdge(GraphEdge("e2", b, add, 0, 1, b.outputs[0]))
    graph.addEdge(GraphEdge("e3", add, relu, 0, 0, add.outputs[0]))

    val mlir = toStableHlo(graph, "main").content
    File(outPath).apply { parentFile?.mkdirs() }.writeText(mlir)
    println(mlir)
    println("[hloExport] wrote $outPath")
}
