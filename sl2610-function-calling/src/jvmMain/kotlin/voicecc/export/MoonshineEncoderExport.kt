package voicecc.export

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.models.moonshine.MoonshineConfig
import sk.ainet.models.moonshine.moonshineEncoder
import sk.ainet.tape.Execution
import java.io.File
import kotlin.reflect.KClass

/**
 * Self-compile the Moonshine encoder: author the published `moonshineEncoder()` DSL
 * (sk.ainet.transformers:skainet-transformers-inference-moonshine), trace it, and emit
 * portable StableHLO for the Torq NPU — replacing the *opaque* prebuilt vendor
 * `encoder.vmfb` with a graph WE produce from the model definition.
 *
 * Pipeline:  DSL → tape → ComputeGraph → DtypeForwardPropagation(bf16) → StableHLO
 * Then:      scripts/iree-compile-torq-docker.sh <mlir> encoder.vmfb   (dockerized iree-compile)
 *
 * Usage (host):
 *   ./gradlew moonshineEncoderMlir                       # 6 layers, bf16 → build/mlir/moonshine-encoder.mlir
 *   ENC_LAYERS=1 ENC_DTYPE=FP32 ./gradlew moonshineEncoderMlir   # 1-layer f32, for numeric bring-up
 *
 * ── What this DOES emit: the exact encoder *graph* (attention/RoPE/LayerNorm/FFN),
 *    HW-agnostic, bf16-native, ready for iree-compile. Verified structurally here.
 *
 * ── What still needs the checkpoint to be a RUNNABLE drop-in (see docs how-to):
 *    1. WEIGHTS. `moonshineEncoder()` builds placeholder parameters; a runnable vmfb needs
 *       the moonshine-tiny weights baked in (safetensors → the DSL module's parameters,
 *       the validated mapping: projections arg = onnxᵀ, fc1/fc2 transposed, LN weight-only).
 *       Wire a real [weightsFor] below once the checkpoint is on the build machine.
 *    2. FRAMES / LAYOUT. The board's vendor preprocessor emits [1, dim, 207] (5 s of audio);
 *       this DSL encoder takes [1, frames, dim]. Set ENC_FRAMES=207 (below) and add the
 *       [dim,frames]→[frames,dim] transpose so it drops in for `MoonshineDecoder.encVmfb`.
 */
fun main(args: Array<String>) {
    val layers = System.getenv("ENC_LAYERS")?.toInt() ?: 6
    val frames = System.getenv("ENC_FRAMES")?.toInt()          // null → config default (165)
    val useF32 = System.getenv("ENC_DTYPE") == "FP32"
    val outPath = args.getOrElse(0) { "build/mlir/moonshine-encoder.mlir" }

    val cfg = MoonshineConfig(encoderLayers = layers).let {
        if (frames != null) it.copy(maxFrames = frames) else it
    }

    // DUMP_PARAMS=1 — print the DSL's parameter names+shapes and exit. Used to ground the
    // checkpoint→DSL name mapping in MoonshineWeights.
    if (System.getenv("DUMP_PARAMS") == "1") {
        val model = moonshineEncoder<BF16, Float>(cfg, BF16::class)
        collectParams(model).forEach { p ->
            println("PARAM ${p.name}  ${p.value.shape.dimensions.toList()}")
        }
        return
    }
    // CHECKPOINT=<dir> — bake real weights (per-tensor f32 .bin, HF-named) so the graph emits
    // weight CONSTANTS (runnable). Absent → placeholder weights, structure only.
    val weights: WeightSource? = System.getenv("CHECKPOINT")?.let { DirBinWeightSource(it) }
    // ENC_BOARD_LAYOUT=1 — accept the board's [1, dim, frames] and transpose to [1, frames, dim]
    // so the vmfb drops in for MoonshineDecoder's preprocessor output ([1,288,207]).
    val boardLayout = System.getenv("ENC_BOARD_LAYOUT") == "1"

    val mlir = if (useF32) traceEncoder(cfg, FP32::class, "FP32", weights, boardLayout)
    else traceEncoder(cfg, BF16::class, "BF16", weights, boardLayout)

    File(outPath).apply { parentFile?.mkdirs() }.writeText(mlir)
    println("[moonshineEncoderMlir] wrote $outPath — ${mlir.lines().size} lines, " +
        "${layers} layers, frames=${cfg.maxFrames}, dtype=${if (useF32) "f32" else "bf16"}, " +
        "weights=${if (weights != null) "baked" else "placeholder"}, boardLayout=$boardLayout")
}

/**
 * Trace [moonshineEncoder] to StableHLO. Mirrors the transformers module's
 * MoonshineEncoderMlirDumpTest recipe, kept here so the demo can self-compile the encoder
 * from the published artifact without the (unpublished) test on the classpath.
 */
private fun <T : DType> traceEncoder(
    cfg: MoonshineConfig,
    dtypeClass: KClass<T>,
    floatDtype: String,
    weights: WeightSource?,
    boardLayout: Boolean,
): String {
    val model = moonshineEncoder<T, Float>(cfg, dtypeClass)

    val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())

    // Bake real weights into the module's (mutable) parameters BEFORE tracing, so they record
    // as constants rather than synthesized inputs.
    if (weights != null) {
        val n = bakeWeights(model, weights, dtypeClass, ctx as ExecutionContext)
        println("[moonshineEncoderMlir] baked $n weight tensors from the checkpoint")
    }

    // Encoder input: [batch, frames, dim] (default) or the board's [batch, dim, frames] which
    // we transpose to [batch, frames, dim] inside the graph.
    val inShape = if (boardLayout) Shape(1, cfg.dim, cfg.maxFrames) else Shape(1, cfg.maxFrames, cfg.dim)
    val input = VoidOpsTensor(
        object : TensorData<T, Float> {
            override val shape = inShape
            override fun get(vararg indices: Int): Float = weightsFor(indices)
            override fun set(vararg indices: Int, value: Float) {}
        },
        dtypeClass,
    )

    val tape = ctx.record {
        val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
        Execution.tapeStack.pushTape(ct)
        try {
            val ectx = this as ExecutionContext
            val x = if (boardLayout) ectx.ops.transpose(input) else input
            model.forward(x, ectx)
        } finally {
            Execution.tapeStack.popTape()
        }
    }.first

    val rawGraph = (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true)

    // Core, HW-agnostic dtype unification (bf16-native traces record norms/reductions as f32).
    // Target-specific passes (Torq tiling) live in the synaptics-torq vendor plugin and are NOT
    // applied here — the emitted StableHLO stays portable.
    val dtypeTarget = if (System.getenv("ENC_SKIP_DTYPE") == "1") null else floatDtype
    val graph = sk.ainet.compile.opt.passes.DtypeForwardPropagationPass(targetFloatDtype = dtypeTarget).apply(rawGraph).graph
    return sk.ainet.compile.hlo.toStableHlo(graph, "moonshine_encoder").content
}

/**
 * Placeholder input/weight values (zeros) — enough to emit the encoder *structure*.
 * Replace with a real moonshine-tiny weight lookup (per parameter name) to bake a
 * RUNNABLE encoder; see the class KDoc, item 1.
 */
private fun weightsFor(@Suppress("UNUSED_PARAMETER") indices: IntArray): Float = 0.0f

/** Recursively collect every parameter (weights + biases) of [m] and its submodules. */
private fun <T : DType, V> collectParams(
    m: sk.ainet.lang.nn.Module<T, V>,
    out: MutableList<sk.ainet.lang.nn.topology.ModuleParameter<*, *>> = mutableListOf(),
): List<sk.ainet.lang.nn.topology.ModuleParameter<*, *>> {
    out.addAll(m.params)
    for (child in m.modules) collectParams(child, out)
    return out
}
