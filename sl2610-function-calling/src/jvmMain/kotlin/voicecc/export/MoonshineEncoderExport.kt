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
    val mlir = if (useF32) traceEncoder(cfg, FP32::class, "FP32") else traceEncoder(cfg, BF16::class, "BF16")

    File(outPath).apply { parentFile?.mkdirs() }.writeText(mlir)
    println("[moonshineEncoderMlir] wrote $outPath — ${mlir.lines().size} lines, " +
        "${layers} layers, frames=${cfg.maxFrames}, dtype=${if (useF32) "f32" else "bf16"}")
}

/**
 * Trace [moonshineEncoder] to StableHLO. Mirrors the transformers module's
 * MoonshineEncoderMlirDumpTest recipe, kept here so the demo can self-compile the encoder
 * from the published artifact without the (unpublished) test on the classpath.
 */
private fun <T : DType> traceEncoder(cfg: MoonshineConfig, dtypeClass: KClass<T>, floatDtype: String): String {
    val model = moonshineEncoder<T, Float>(cfg, dtypeClass)

    // Encoder input = conv-frontend output [batch, frames, dim].
    val input = VoidOpsTensor(
        object : TensorData<T, Float> {
            override val shape = Shape(1, cfg.maxFrames, cfg.dim)
            override fun get(vararg indices: Int): Float = weightsFor(indices)
            override fun set(vararg indices: Int, value: Float) {}
        },
        dtypeClass,
    )

    val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
    val tape = ctx.record {
        val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
        Execution.tapeStack.pushTape(ct)
        try {
            model.forward(input, this as ExecutionContext)
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
