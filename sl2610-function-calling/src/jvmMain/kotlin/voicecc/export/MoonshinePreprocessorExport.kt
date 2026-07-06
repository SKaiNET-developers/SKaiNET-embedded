package voicecc.export

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.FP32
import sk.ainet.models.moonshine.MoonshineConfig
import sk.ainet.models.moonshine.moonshinePreprocessor
import sk.ainet.tape.Execution
import java.io.File

/**
 * Self-compile the Moonshine audio FRONTEND: author `moonshinePreprocessor()` (board layout) and emit
 * StableHLO with baked weights — a drop-in for the vendor `preprocessor_cpu.vmfb`:
 *   `input_values [1, samples] f32  →  features [1, dim, frames] f32`   (80000 → [1,288,207])
 * The demo runtime casts that f32 to bf16 and feeds the (board-layout) encoder. FP32 to match.
 *
 *   PP_CHECKPOINT=<hf .bin dir> ./gradlew moonshinePreprocessorMlir   → build/mlir/moonshine-preprocessor.mlir
 * then scripts/iree-compile-cpu-ish → aarch64 llvm-cpu vmfb → deploy as preprocessor_cpu.vmfb.
 */
fun main() {
    val cfg = MoonshineConfig()
    val samples = System.getenv("PP_SAMPLES")?.toInt() ?: 80000
    val outDir = System.getenv("MOONSHINE_DECODER_OUT_DIR") ?: "build/mlir"
    val checkpoint = System.getenv("PP_CHECKPOINT")?.let { DirBinWeightSource(it) }
    if (checkpoint == null) println("[moonshinePreprocessorMlir] no PP_CHECKPOINT — weights stay as args")

    val model = moonshinePreprocessor<FP32, Float>(cfg, FP32::class, boardLayout = true)
    val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
    if (checkpoint != null) {
        val n = bakeMoonshineWeights(model, checkpoint, ::preprocessorHfNameFor, FP32::class, ctx as ExecutionContext)
        println("[moonshinePreprocessorMlir] baked $n weight tensors")
    }
    val input = VoidOpsTensor(
        object : TensorData<FP32, Float> {
            override val shape = Shape(1, samples)
            override fun get(vararg indices: Int): Float = 0.0f
            override fun set(vararg indices: Int, value: Float) {}
        },
        FP32::class,
    )
    val tape = ctx.record {
        val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
        Execution.tapeStack.pushTape(ct)
        try { model.forward(input, this as ExecutionContext) } finally { Execution.tapeStack.popTape() }
    }.first
    val graph = (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true, embedConstants = checkpoint != null)
    val g = sk.ainet.compile.opt.passes.DtypeForwardPropagationPass(targetFloatDtype = "FP32").apply(graph).graph
    // conv1d only has a converter in the EXTENDED registry (NeuralNetOperationsConverter); the plain
    // toStableHlo uses createBasic, which lacks it. The transformer encoder/decoder never needed convs.
    val mlir = sk.ainet.compile.hlo.StableHloConverterFactory.createExtended().convert(g, "moonshine_preprocessor").content
    File("$outDir/moonshine-preprocessor.mlir").apply { parentFile?.mkdirs() }.writeText(mlir)
    println("[moonshinePreprocessorMlir] wrote $outDir/moonshine-preprocessor.mlir (${mlir.lines().size} lines)")
}
