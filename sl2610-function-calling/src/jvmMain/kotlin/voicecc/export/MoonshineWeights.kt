package voicecc.export

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import java.io.File
import kotlin.reflect.KClass

/**
 * Bakes real moonshine-tiny weights into the `moonshineEncoder()` DSL so the traced graph
 * emits weight CONSTANTS (a runnable vmfb) instead of weight arguments.
 *
 * Mechanism: [ModuleParameter.value] is mutable, so we walk the module tree, map each DSL
 * parameter to its checkpoint tensor by NAME, apply the layout convention, and overwrite the
 * value before tracing.
 *
 * Requires transformers **0.34.1+**, where every encoder parameter is uniquely layer-qualified
 * (`enc.$layer.attn.q_proj.weight`, `enc.$layer.attn_norm.weight`, …). (0.34.0 left the attn/norm
 * names un-qualified and needed a positional workaround; that's gone.)
 *
 * CHECKPOINT: supply a directory of per-tensor little-endian f32 `.bin` files named by the HF
 * tensor name (e.g. `encoder.layers.0.self_attn.q_proj.weight.bin`). The HF names + the transpose
 * conventions in [hfNameFor] are the validated moonshine-tiny mapping; verify against your
 * checkpoint (the values are untestable without it). Missing tensors fail fast.
 */
internal interface WeightSource {
    /** Flat row-major f32 values for [hfName], or null if absent. */
    fun weight(hfName: String): FloatArray?
}

/** Reads `$dir/<hfName>.bin` as little-endian f32. Matches the demo's raw-bin I/O. */
internal class DirBinWeightSource(private val dir: String) : WeightSource {
    override fun weight(hfName: String): FloatArray? {
        val f = File(dir, "$hfName.bin")
        if (!f.exists()) return null
        val bytes = f.readBytes()
        val out = FloatArray(bytes.size / 4)
        for (i in out.indices) {
            var bits = 0
            for (b in 0 until 4) bits = bits or ((bytes[i * 4 + b].toInt() and 0xFF) shl (8 * b))
            out[i] = Float.fromBits(bits)
        }
        return out
    }
}

private val ENC_LAYER = Regex("""^enc\.(\d+)\.(.+)$""")

/**
 * Map a DSL parameter name to its moonshine-tiny HF tensor name, plus whether the 2-D weight
 * must be transposed to match the DSL's `[out, in]` (linear) / `input @ weightᵀ` convention.
 * Relies on the 0.34.1 layer-qualified names (`enc.$layer.*`).
 *
 * VERIFY these HF names against your checkpoint before trusting the output.
 */
private fun hfNameFor(dslName: String): Pair<String, Boolean>? {
    when (dslName) {
        "enc_out_norm.weight" -> return "encoder.layer_norm.weight" to false
        "enc_out_norm.bias" -> return "encoder.layer_norm.bias" to false
    }
    val m = ENC_LAYER.matchEntire(dslName) ?: return null
    val layer = m.groupValues[1]
    return when (m.groupValues[2]) {
        "attn_norm.weight" -> "encoder.layers.$layer.self_attn_layer_norm.weight" to false
        "attn_norm.bias" -> "encoder.layers.$layer.self_attn_layer_norm.bias" to false
        "attn.q_proj.weight" -> "encoder.layers.$layer.self_attn.q_proj.weight" to true
        "attn.k_proj.weight" -> "encoder.layers.$layer.self_attn.k_proj.weight" to true
        "attn.v_proj.weight" -> "encoder.layers.$layer.self_attn.v_proj.weight" to true
        "attn.o_proj.weight" -> "encoder.layers.$layer.self_attn.o_proj.weight" to true
        "ffn_norm.weight" -> "encoder.layers.$layer.final_layer_norm.weight" to false
        "ffn_norm.bias" -> "encoder.layers.$layer.final_layer_norm.bias" to false
        "ffn_up.weight" -> "encoder.layers.$layer.fc1.weight" to true
        "ffn_up.bias" -> "encoder.layers.$layer.fc1.bias" to false
        "ffn_down.weight" -> "encoder.layers.$layer.fc2.weight" to true
        "ffn_down.bias" -> "encoder.layers.$layer.fc2.bias" to false
        else -> null
    }
}

/** Collect params (weights + biases) of [m] and submodules in deterministic walk order. */
private fun <T : DType, V> walkParams(
    m: Module<T, V>,
    out: MutableList<ModuleParameter<*, *>> = mutableListOf(),
): List<ModuleParameter<*, *>> {
    out.addAll(m.params)
    for (child in m.modules) walkParams(child, out)
    return out
}

/**
 * Overwrite every parameter of [model] with real weights from [src], mapped by name. Returns the
 * number of parameters baked. Throws if a DSL param has no mapping or its tensor is missing (fail
 * fast — a partial bake would silently produce a wrong model).
 */
internal fun <T : DType, V> bakeWeights(
    model: Module<T, V>,
    src: WeightSource,
    dtypeClass: KClass<T>,
    ctx: ExecutionContext,
): Int {
    var baked = 0
    for (p in walkParams(model)) {
        val (hfName, transpose) = hfNameFor(p.name)
            ?: error("no HF mapping for DSL param '${p.name}' (need transformers 0.34.1 layer-qualified names)")

        val shape = p.value.shape.dimensions
        var data = src.weight(hfName)
            ?: error("checkpoint missing tensor '$hfName' for DSL param '${p.name}'")
        if (transpose && shape.size == 2) data = transpose2d(data, rows = shape[1], cols = shape[0])
        require(data.size == shape.fold(1) { a, b -> a * b }) {
            "tensor '$hfName' size ${data.size} != DSL param '${p.name}' shape ${shape.toList()}"
        }

        @Suppress("UNCHECKED_CAST")
        val tensor = ctx.fromData(
            DenseFloatArrayTensorData<T>(Shape(*shape), data) as TensorData<T, V>,
            dtypeClass,
        )
        @Suppress("UNCHECKED_CAST")
        (p as ModuleParameter<T, V>).value = tensor
        baked++
    }
    return baked
}

/** Transpose a row-major `[rows, cols]` f32 matrix to `[cols, rows]`. */
private fun transpose2d(src: FloatArray, rows: Int, cols: Int): FloatArray {
    val dst = FloatArray(src.size)
    for (r in 0 until rows) for (c in 0 until cols) dst[c * rows + r] = src[r * cols + c]
    return dst
}
