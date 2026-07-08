package voicecc.export

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Bake real moonshine-tiny DECODER weights into `moonshineDecoder()` so the traced graphs emit
 * weight CONSTANTS (runnable vmfbs). Companion to the encoder's [hfNameFor]/[bakeWeights]; reuses
 * this package's [WeightSource]/[DirBinWeightSource].
 *
 * Source = the HF safetensors checkpoint converted by `scripts/convert_moonshine_weights.py` (raw
 * `[out,in]` layout, `model.` prefix stripped). The DSL projects through `linearProject = x @ Wᵀ`
 * with `W` in `[out,in]` (identical to HF `nn.Linear`), so raw HF weights map WITHOUT transpose —
 * unlike the encoder's ONNX source (`[in,out]`, transpose=true). `lm_head` is TIED to
 * `decoder.embed_tokens`; Moonshine LayerNorms are weight-only (bias → zeros).
 *
 * Validated in `skainet-transformers` (MoonshineDecoderBakeTest): 112 params, shapes match, the
 * baked decoder transcribes "One, two, three." on CPU. Kept here (jvmMain) so the demo's export
 * pipeline can bake without a published artifact; consolidate into the moonshine module on release.
 */
private val DEC_LAYER = Regex("""^dec\.(\d+)\.(.+)$""")

internal data class DecMap(val hfName: String?, val transpose: Boolean)

/** Preprocessor (audio frontend) param → HF encoder conv/groupnorm tensor (conv weights [out,in,k], no transpose). */
internal fun preprocessorHfNameFor(dslName: String): DecMap? = when (dslName) {
    "conv1.weight" -> DecMap("encoder.conv1.weight", false)
    "conv2.weight" -> DecMap("encoder.conv2.weight", false)
    "conv2.bias" -> DecMap("encoder.conv2.bias", false)
    "conv3.weight" -> DecMap("encoder.conv3.weight", false)
    "conv3.bias" -> DecMap("encoder.conv3.bias", false)
    "groupnorm.weight" -> DecMap("encoder.groupnorm.weight", false)
    "groupnorm.bias" -> DecMap("encoder.groupnorm.bias", false)
    else -> null
}

/** DSL decoder param name → (HF tensor name | null=zeros, transpose?). */
internal fun decoderHfNameFor(dslName: String): DecMap? {
    when (dslName) {
        "dec_out_norm.weight" -> return DecMap("decoder.norm.weight", false)
        "dec_out_norm.bias" -> return DecMap(null, false)
        "lm_head.weight" -> return DecMap("decoder.embed_tokens.weight", false) // tied
        "lm_head.bias" -> return DecMap(null, false)
    }
    val m = DEC_LAYER.matchEntire(dslName) ?: return null
    val l = m.groupValues[1]
    return when (m.groupValues[2]) {
        "self_attn_norm.weight" -> DecMap("decoder.layers.$l.input_layernorm.weight", false)
        "self_attn_norm.bias" -> DecMap(null, false)
        "cross_attn_norm.weight" -> DecMap("decoder.layers.$l.post_attention_layernorm.weight", false)
        "cross_attn_norm.bias" -> DecMap(null, false)
        "mlp_norm.weight" -> DecMap("decoder.layers.$l.final_layernorm.weight", false)
        "mlp_norm.bias" -> DecMap(null, false)
        "self_attn.q_proj.weight" -> DecMap("decoder.layers.$l.self_attn.q_proj.weight", false)
        "self_attn.k_proj.weight" -> DecMap("decoder.layers.$l.self_attn.k_proj.weight", false)
        "self_attn.v_proj.weight" -> DecMap("decoder.layers.$l.self_attn.v_proj.weight", false)
        "self_attn.o_proj.weight" -> DecMap("decoder.layers.$l.self_attn.o_proj.weight", false)
        "cross_attn.q_proj.weight" -> DecMap("decoder.layers.$l.encoder_attn.q_proj.weight", false)
        "cross_attn.k_proj.weight" -> DecMap("decoder.layers.$l.encoder_attn.k_proj.weight", false)
        "cross_attn.v_proj.weight" -> DecMap("decoder.layers.$l.encoder_attn.v_proj.weight", false)
        "cross_attn.o_proj.weight" -> DecMap("decoder.layers.$l.encoder_attn.o_proj.weight", false)
        "mlp_fc1.weight" -> DecMap("decoder.layers.$l.mlp.fc1.weight", false)
        "mlp_fc1.bias" -> DecMap("decoder.layers.$l.mlp.fc1.bias", false)
        "mlp_fc2.weight" -> DecMap("decoder.layers.$l.mlp.fc2.weight", false)
        "mlp_fc2.bias" -> DecMap("decoder.layers.$l.mlp.fc2.bias", false)
        else -> null
    }
}

private fun <T : DType, V> walkDecParams(
    m: Module<T, V>,
    out: MutableList<ModuleParameter<*, *>> = mutableListOf(),
): List<ModuleParameter<*, *>> {
    out.addAll(m.params)
    for (child in m.modules) walkDecParams(child, out)
    return out
}

/** Overwrite every decoder param with real weights (fail-fast on any gap). Returns the count baked. */
internal fun <T : DType, V> bakeDecoderWeights(
    model: Module<T, V>,
    src: WeightSource,
    dtypeClass: KClass<T>,
    ctx: ExecutionContext,
): Int = bakeMoonshineWeights(model, src, ::decoderHfNameFor, dtypeClass, ctx)

/** Overwrite every param of [model] using [mapper] (decoder or preprocessor). Fail-fast on any gap. */
internal fun <T : DType, V> bakeMoonshineWeights(
    model: Module<T, V>,
    src: WeightSource,
    mapper: (String) -> DecMap?,
    dtypeClass: KClass<T>,
    ctx: ExecutionContext,
): Int {
    var baked = 0
    for (p in walkDecParams(model)) {
        val map = mapper(p.name) ?: error("no HF mapping for DSL param '${p.name}'")
        val shape = p.value.shape.dimensions
        val n = shape.fold(1) { a, b -> a * b }
        var data = if (map.hfName == null) FloatArray(n)
        else (src.weight(map.hfName) ?: error("checkpoint missing '${map.hfName}' for '${p.name}'"))
        if (map.transpose && shape.size == 2) data = decTranspose2d(data, rows = shape[1], cols = shape[0])
        require(data.size == n) { "tensor '${map.hfName}' size ${data.size} != '${p.name}' ${shape.toList()}" }
        @Suppress("UNCHECKED_CAST")
        (p as ModuleParameter<T, V>).value =
            ctx.fromData(DenseFloatArrayTensorData<T>(Shape(*shape), data) as TensorData<T, V>, dtypeClass)
        baked++
    }
    return baked
}

private fun decTranspose2d(src: FloatArray, rows: Int, cols: Int): FloatArray {
    val dst = FloatArray(src.size)
    for (r in 0 until rows) for (c in 0 until cols) dst[c * rows + r] = src[r * cols + c]
    return dst
}
