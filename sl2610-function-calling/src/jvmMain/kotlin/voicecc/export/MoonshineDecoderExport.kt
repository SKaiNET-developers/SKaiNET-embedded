package voicecc.export

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.models.moonshine.MoonshineConfig
import sk.ainet.models.moonshine.MoonshineDecoderModel
import sk.ainet.models.moonshine.moonshineDecoder
import sk.ainet.tape.Execution
import java.io.File
import kotlin.reflect.KClass

/**
 * Self-compile the Moonshine DECODER: author the published `moonshineDecoder()` DSL, bake real
 * weights, and emit the TWO board KV-cache graphs as StableHLO (weight constants → runnable vmfbs):
 *   moonshine-decoder.mlir            prefill: (embeds[1,1,288], memory[1,F,288])
 *                                     → logits + per-layer self_k/v[1,8,1,36] + cross_k/v[1,8,F,36]
 *   moonshine-decoder-with-past.mlir  step:    (token[1,1,288], cos[1,36], sin[1,36],
 *                                              self_k/v[1,8,P,36], cross_k/v[1,8,F,36])
 *                                     → logits + extended self_k/v[1,8,P+1,36]
 *
 * RoPE position is a RUNTIME input (cos/sin tables, host-built via RoPE.buildInterleavedCosSin) —
 * one with_past vmfb serves every position, no in-graph gather.
 *
 * Usage:
 *   DECODER_CHECKPOINT=weights ./gradlew moonshineDecoderMlir           # bf16, F=207, P=1
 *   DEC_DTYPE=FP32 DECODER_CHECKPOINT=weights ./gradlew moonshineDecoderMlir   # f32, numeric bring-up
 * where `weights/` is the per-tensor HF `.bin` dir from scripts/convert_moonshine_weights.py.
 *
 * NEXT (see .claude/plans/moonshine-demo-rewrite-3DEC-e.md): iree-compile both → decoder.vmfb /
 * decoder_with_past.vmfb (CPU then Torq); DEC_PAST is a fixed trace length — a single board vmfb over
 * a GROWING self-cache needs a dynamic seq dim (`1x8x?x36`) or the vendor-style fixed-pad rework.
 */
fun main() {
    val cfg = MoonshineConfig()
    val frames = System.getenv("DEC_FRAMES")?.toInt() ?: 207
    val past = System.getenv("DEC_PAST")?.toInt() ?: 1
    val useF32 = System.getenv("DEC_DTYPE") == "FP32"
    val outDir = System.getenv("MOONSHINE_DECODER_OUT_DIR") ?: "build/mlir"
    val checkpoint = System.getenv("DECODER_CHECKPOINT")?.let { DirBinWeightSource(it) }
    if (checkpoint == null) println("[moonshineDecoderMlir] no DECODER_CHECKPOINT — weights stay as args (structure only)")

    val prefill = if (useF32) tracePrefill(cfg, frames, FP32::class, checkpoint)
    else tracePrefill(cfg, frames, BF16::class, checkpoint)
    val withPast = if (useF32) traceWithPast(cfg, frames, past, FP32::class, checkpoint)
    else traceWithPast(cfg, frames, past, BF16::class, checkpoint)

    write("$outDir/moonshine-decoder.mlir", prefill)
    write("$outDir/moonshine-decoder-with-past.mlir", withPast)
}

private fun write(path: String, mlir: String) {
    File(path).apply { parentFile?.mkdirs() }.writeText(mlir)
    println("[moonshineDecoderMlir] wrote $path — ${mlir.lines().size} lines")
}

private fun <T : DType> tracePrefill(cfg: MoonshineConfig, frames: Int, dtype: KClass<T>, w: WeightSource?): String {
    val model = moonshineDecoder<T, Float>(cfg, dtype)
    val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
    if (w != null) bakeDecoderWeights(model, w, dtype, ctx as ExecutionContext)
    val embeds = void(Shape(1, 1, cfg.dim), dtype)
    val memory = void(Shape(1, frames, cfg.dim), dtype)
    val tape = ctx.record {
        val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
        Execution.tapeStack.pushTape(ct)
        try { model.forwardPrefill(embeds, memory, this as ExecutionContext) } finally { Execution.tapeStack.popTape() }
    }.first
    return emit(tape, "moonshine_decoder_prefill", w != null, dtypeTag(dtype))
}

private fun <T : DType> traceWithPast(cfg: MoonshineConfig, frames: Int, past: Int, dtype: KClass<T>, w: WeightSource?): String {
    val model: MoonshineDecoderModel<T, Float> = moonshineDecoder(cfg, dtype)
    val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
    if (w != null) bakeDecoderWeights(model, w, dtype, ctx as ExecutionContext)
    val token = void(Shape(1, 1, cfg.dim), dtype)
    val cos = void(Shape(1, cfg.headDim), dtype)
    val sin = void(Shape(1, cfg.headDim), dtype)
    val selfK = List(cfg.decoderLayers) { void(Shape(1, cfg.nHeads, past, cfg.headDim), dtype) }
    val selfV = List(cfg.decoderLayers) { void(Shape(1, cfg.nHeads, past, cfg.headDim), dtype) }
    val crossK = List(cfg.decoderLayers) { void(Shape(1, cfg.nHeads, frames, cfg.headDim), dtype) }
    val crossV = List(cfg.decoderLayers) { void(Shape(1, cfg.nHeads, frames, cfg.headDim), dtype) }
    val tape = ctx.record {
        val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
        Execution.tapeStack.pushTape(ct)
        try {
            model.forwardWithPast(token, cos, sin, selfK, selfV, crossK, crossV, this as ExecutionContext)
        } finally { Execution.tapeStack.popTape() }
    }.first
    return emit(tape, "moonshine_decoder_with_past", w != null, dtypeTag(dtype))
}

private fun emit(tape: Any?, fn: String, baked: Boolean, floatTag: String): String {
    val graph = (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true, embedConstants = baked)
    val dt = if (System.getenv("DEC_SKIP_DTYPE") == "1") null else floatTag
    val g = sk.ainet.compile.opt.passes.DtypeForwardPropagationPass(targetFloatDtype = dt).apply(graph).graph
    return sk.ainet.compile.hlo.toStableHlo(g, fn).content
}

private fun <T : DType> dtypeTag(dtype: KClass<T>): String = if (dtype == FP32::class) "FP32" else "BF16"

private fun <T : DType> void(shape: Shape, dtype: KClass<T>): Tensor<T, Float> =
    VoidOpsTensor(object : TensorData<T, Float> {
        override val shape = shape
        override fun get(vararg indices: Int): Float = 0.0f
        override fun set(vararg indices: Int, value: Float) {}
    }, dtype)
