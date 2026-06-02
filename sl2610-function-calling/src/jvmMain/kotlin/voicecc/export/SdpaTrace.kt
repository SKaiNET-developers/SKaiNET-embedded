package voicecc.export

import sk.ainet.compile.hlo.toStableHlo
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.tape.Execution

/**
 * "Trace SDPA first": build a tiny 1-block gemma3 graph, record its forward with
 * VoidTensorOps (captures op structure, no real math, no weights needed), and
 * print the recorded operation node types. This reveals whether
 * `scaledDotProductAttention` is an ATOMIC tape node (-> needs a transformer-side
 * StableHLO converter in skainet-transformers) or DECOMPOSES into matmul/softmax/
 * transpose (-> nothing transformer-specific to add). Then it attempts toStableHlo
 * and prints which op types have no converter (the core primitives to add).
 */
fun main() {
    // Standalone attention block (no embedding/gather) — isolates SDPA.
    val dim = 64
    val seqLen = 4
    val model = MultiHeadAttention<FP32, Float>(
        dim = dim,
        nHeads = 2,
        nKVHeads = 1,
        causal = true,
    )

    val input = VoidOpsTensor(
        object : TensorData<FP32, Float> {
            override val shape = Shape(seqLen, dim)
            override fun get(vararg indices: Int): Float = 0.0f
            override fun set(vararg indices: Int, value: Float) {}
        },
        FP32::class,
    )

    val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
    val (tape, _) = ctx.record {
        val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
        Execution.tapeStack.pushTape(ct)
        try {
            model.forward(input, this as ExecutionContext)
        } finally {
            Execution.tapeStack.popTape()
        }
    }
    val graph = (tape as DefaultExecutionTape).toComputeGraph()

    val opTypes = graph.nodes.map { it.operation::class.simpleName ?: "?" }
    println("=== recorded op node types (${graph.nodes.size} nodes) ===")
    opTypes.groupingBy { it }.eachCount().toSortedMap().forEach { (k, v) -> println("  $k x$v") }
    println("SDPA atomic? -> " + opTypes.any { it.contains("ScaledDot", true) || it.contains("Attention", true) })

    println("=== toStableHlo attempt ===")
    try {
        val mlir = toStableHlo(graph, "gemma_block").content
        println(mlir.lines().take(40).joinToString("\n"))
        println("... (${mlir.lines().size} lines total)")
    } catch (e: Throwable) {
        println("toStableHlo failed: ${e::class.simpleName}: ${e.message}")
    }
}
