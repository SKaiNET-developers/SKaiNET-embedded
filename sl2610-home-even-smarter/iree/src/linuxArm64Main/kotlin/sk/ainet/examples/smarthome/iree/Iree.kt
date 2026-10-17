package sk.ainet.examples.smarthome.iree

import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.allocArrayOf
import kotlinx.cinterop.convert
import kotlinx.cinterop.cstr
import kotlinx.cinterop.get
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.set
import kotlinx.cinterop.toKString
import kotlinx.cinterop.usePinned
import sk.ainet.examples.smarthome.iree.native.HES_BF16
import sk.ainet.examples.smarthome.iree.native.HES_F16
import sk.ainet.examples.smarthome.iree.native.HES_F32
import sk.ainet.examples.smarthome.iree.native.HES_I32
import sk.ainet.examples.smarthome.iree.native.HES_I64
import sk.ainet.examples.smarthome.iree.native.HES_I8
import sk.ainet.examples.smarthome.iree.native.hes_call
import cnames.structs.hes_kv
import sk.ainet.examples.smarthome.iree.native.hes_kv_chunk
import sk.ainet.examples.smarthome.iree.native.hes_kv_create
import sk.ainet.examples.smarthome.iree.native.hes_kv_position
import sk.ainet.examples.smarthome.iree.native.hes_kv_prefill
import sk.ainet.examples.smarthome.iree.native.hes_kv_release
import sk.ainet.examples.smarthome.iree.native.hes_kv_release_prefill
import sk.ainet.examples.smarthome.iree.native.hes_kv_restore
import cnames.structs.hes_kv_snapshot
import sk.ainet.examples.smarthome.iree.native.hes_kv_snapshot_release
import sk.ainet.examples.smarthome.iree.native.hes_kv_snapshot_take
import sk.ainet.examples.smarthome.iree.native.hes_kv_spec
import sk.ainet.examples.smarthome.iree.native.hes_kv_step
import sk.ainet.examples.smarthome.iree.native.hes_last_error
import cnames.structs.hes_ms
import sk.ainet.examples.smarthome.iree.native.hes_ms_create
import sk.ainet.examples.smarthome.iree.native.hes_ms_dim
import sk.ainet.examples.smarthome.iree.native.hes_ms_feed
import sk.ainet.examples.smarthome.iree.native.hes_ms_finish
import sk.ainet.examples.smarthome.iree.native.hes_ms_release
import sk.ainet.examples.smarthome.iree.native.hes_ms_reset
import sk.ainet.examples.smarthome.iree.native.hes_ms_spec
import sk.ainet.examples.smarthome.iree.native.hes_ms_stats
import sk.ainet.examples.smarthome.iree.native.hes_ms_vocab
import cnames.structs.hes_runtime
import sk.ainet.examples.smarthome.iree.native.hes_runtime_create
import sk.ainet.examples.smarthome.iree.native.hes_runtime_drivers
import sk.ainet.examples.smarthome.iree.native.hes_runtime_release
import cnames.structs.hes_session
import sk.ainet.examples.smarthome.iree.native.hes_session_create
import sk.ainet.examples.smarthome.iree.native.hes_session_release
import cnames.structs.hes_view
import sk.ainet.examples.smarthome.iree.native.hes_view_bytes
import sk.ainet.examples.smarthome.iree.native.hes_view_dim
import sk.ainet.examples.smarthome.iree.native.hes_view_element_type
import sk.ainet.examples.smarthome.iree.native.hes_view_from_host
import sk.ainet.examples.smarthome.iree.native.hes_view_rank
import sk.ainet.examples.smarthome.iree.native.hes_view_release
import sk.ainet.examples.smarthome.iree.native.hes_view_to_host

/** A failure reported by the native runtime; [message] is the formatted IREE status. */
class IreeException(message: String) : IllegalStateException(message)

@OptIn(ExperimentalForeignApi::class)
private fun failure(what: String): Nothing = throw IreeException("$what: ${hes_last_error()?.toKString().orEmpty()}")

/** Element types the app exchanges with graphs. */
@OptIn(ExperimentalForeignApi::class)
enum class ElementType(val code: Int, val bytes: Int) {
    F32(HES_F32.toInt(), 4), F16(HES_F16.toInt(), 2), BF16(HES_BF16.toInt(), 2),
    I32(HES_I32.toInt(), 4), I64(HES_I64.toInt(), 8), I8(HES_I8.toInt(), 1);

    companion object {
        fun of(code: Int): ElementType = entries.firstOrNull { it.code == code } ?: throw IreeException("unsupported element type 0x${code.toString(16)}")
    }
}

/** A dense row-major tensor in host memory: what goes into and comes out of a [IreeSession.call]. */
class HostTensor(val type: ElementType, val shape: LongArray, val bytes: ByteArray) {
    init {
        val count = shape.fold(1L) { a, d -> a * d }
        require(bytes.size.toLong() == count * type.bytes) { "tensor ${shape.toList()} ${type.name} needs ${count * type.bytes} bytes, got ${bytes.size}" }
    }

    val elementCount: Int get() = bytes.size / type.bytes

    fun toFloats(): FloatArray = when (type) {
        ElementType.F32 -> FloatArray(elementCount) { Float.fromBits(int32(it * 4)) }
        ElementType.BF16 -> FloatArray(elementCount) { Float.fromBits(uint16(it * 2) shl 16) }
        else -> throw IreeException("toFloats: ${type.name} is not f32/bf16")
    }

    fun toInts(): IntArray = when (type) {
        ElementType.I32 -> IntArray(elementCount) { int32(it * 4) }
        else -> throw IreeException("toInts: ${type.name} is not i32")
    }

    private fun uint16(o: Int): Int = (bytes[o].toInt() and 0xff) or ((bytes[o + 1].toInt() and 0xff) shl 8)
    private fun int32(o: Int): Int = uint16(o) or (uint16(o + 2) shl 16)

    companion object {
        fun f32(shape: LongArray, values: FloatArray): HostTensor =
            HostTensor(ElementType.F32, shape, ByteArray(values.size * 4).also { b -> values.forEachIndexed { i, v -> b.putInt(i * 4, v.toRawBits()) } })

        /** f32 values rounded to bf16 (round to nearest even). */
        fun bf16(shape: LongArray, values: FloatArray): HostTensor =
            HostTensor(ElementType.BF16, shape, ByteArray(values.size * 2).also { b ->
                values.forEachIndexed { i, v ->
                    val u = v.toRawBits()
                    val h = (u + 0x7FFF + ((u ushr 16) and 1)) ushr 16
                    b[i * 2] = h.toByte(); b[i * 2 + 1] = (h ushr 8).toByte()
                }
            })

        fun i32(shape: LongArray, values: IntArray): HostTensor =
            HostTensor(ElementType.I32, shape, ByteArray(values.size * 4).also { b -> values.forEachIndexed { i, v -> b.putInt(i * 4, v) } })

        private fun ByteArray.putInt(o: Int, v: Int) {
            this[o] = v.toByte(); this[o + 1] = (v ushr 8).toByte(); this[o + 2] = (v ushr 16).toByte(); this[o + 3] = (v ushr 24).toByte()
        }
    }
}

/**
 * One IREE instance for the process. Devices ("local-task" = CPU, "torq" = NPU, "vulkan" = GPU) are created on first
 * use and shared by every session on them. [flags] are IREE runtime flags, parsed once per process, e.g.
 * `--torq_hw_type=astra_machina`.
 */
@OptIn(ExperimentalForeignApi::class)
class IreeRuntime(flags: List<String> = DEFAULT_FLAGS) : AutoCloseable {
    private var handle: CPointer<hes_runtime>? = memScoped {
        val argv = allocArrayOf(flags.map { it.cstr.ptr })
        hes_runtime_create(argv, flags.size) ?: failure("runtime")
    }

    private val rt: CPointer<hes_runtime> get() = handle ?: throw IreeException("runtime closed")

    /** HAL drivers compiled in, e.g. `[local-sync, local-task, vulkan, torq]`. */
    val drivers: List<String> get() = hes_runtime_drivers(rt)?.toKString().orEmpty().split(' ').filter { it.isNotEmpty() }

    /** Loads [vmfb] on [device]; with [irpa] its parameters are served under [scope]. */
    fun session(device: String, vmfb: String, irpa: String? = null, scope: String = "model"): IreeSession =
        IreeSession(hes_session_create(rt, device, vmfb, irpa, scope) ?: failure("session $vmfb on $device"))

    /** The three-graph KV decoder (see [IreeKv]); `null` prefill paths when the prefix is not prefilled here. */
    fun kv(
        device: String, spec: KvSpec,
        withPast: Graph, chunk: Graph, prefill: Graph?,
    ): IreeKv = memScoped {
        val s = alloc<hes_kv_spec>()
        s.n_layers = spec.nLayers; s.head_dim = spec.headDim; s.n_kv_heads = spec.nKvHeads; s.n_heads = spec.nHeads
        s.hidden_size = spec.hiddenSize; s.vocab_size = spec.vocabSize; s.sliding_window = spec.slidingWindow
        s.global_layer_period = spec.globalLayerPeriod; s.chunk = spec.chunk; s.mask_heads = spec.maskHeads
        s.sliding_rope_base = spec.slidingRopeBase; s.global_rope_base = spec.globalRopeBase
        val k = hes_kv_create(
            rt, device, s.ptr,
            withPast.vmfb, withPast.irpa, withPast.function,
            chunk.vmfb, chunk.irpa, chunk.function,
            prefill?.vmfb, prefill?.irpa, prefill?.function,
        ) ?: failure("kv on $device")
        IreeKv(k, spec)
    }

    /**
     * The Moonshine v2 streaming recognizer (see [IreeMoonshine]) over [files], each stage on its device from
     * [devices]. Creating it runs one warm-up pass through every graph unless [MoonshineSpec.warmUp] is off.
     */
    fun moonshine(files: MoonshineFiles, devices: MoonshineDevices, spec: MoonshineSpec = MoonshineSpec()): IreeMoonshine = memScoped {
        val s = alloc<hes_ms_spec>()
        s.n_layers = spec.nLayers; s.head_dim = spec.headDim; s.rope_pairs = spec.ropePairs; s.dim = spec.dim
        s.enc_dim = spec.encDim; s.lag_budget_ms = spec.lagBudgetMs; s.fast_finish = if (spec.fastFinish) 1 else 0
        s.skip_warmup = if (spec.warmUp) 0 else 1; s.verbose = if (spec.verbose) 1 else 0
        val m = hes_ms_create(
            rt, s.ptr,
            devices.frontend, devices.encoder, devices.adapter, devices.decoder,
            files.frontend, files.encoder, files.adapter, files.prefill, files.step,
            files.params, files.vocab, files.decEmbed,
        ) ?: failure("load")
        IreeMoonshine(m)
    }

    override fun close() {
        handle?.let { hes_runtime_release(it) }
        handle = null
    }

    /** A compiled graph: module, parameter archive, exported function (`null` = the contract's default name). */
    data class Graph(val vmfb: String, val irpa: String, val function: String? = null)

    companion object {
        val DEFAULT_FLAGS: List<String> = listOf("--torq_hw_type=astra_machina")
    }
}

/** One loaded module on one device. Not thread-safe: callers serialize calls per session. */
@OptIn(ExperimentalForeignApi::class)
class IreeSession internal constructor(private var handle: CPointer<hes_session>?) : AutoCloseable {
    private val s: CPointer<hes_session> get() = handle ?: throw IreeException("session closed")

    /** Calls [function] (e.g. `module.main`) with host tensors and returns its outputs copied back to the host. */
    fun call(function: String, inputs: List<HostTensor>, maxOutputs: Int = 64): List<HostTensor> = memScoped {
        val ins = allocArray<CPointerVar<hes_view>>(inputs.size.coerceAtLeast(1))
        val outs = allocArray<CPointerVar<hes_view>>(maxOutputs)
        var produced = 0
        try {
            inputs.forEachIndexed { i, t ->
                val dims = allocArray<kotlinx.cinterop.LongVar>(t.shape.size.coerceAtLeast(1))
                t.shape.forEachIndexed { d, v -> dims[d] = v }
                ins[i] = t.bytes.usePinned { pinned ->
                    hes_view_from_host(s, t.type.code, t.shape.size, dims, pinned.addressOf(0), t.bytes.size.convert())
                } ?: failure("input $i of $function")
            }
            produced = hes_call(s, function, ins, inputs.size, outs, maxOutputs)
            if (produced < 0) failure("call $function")
            List(produced) { i -> toHost(outs[i]!!) }
        } finally {
            for (i in inputs.indices) ins[i]?.let { hes_view_release(it) }
            for (i in 0 until produced) outs[i]?.let { hes_view_release(it) }
        }
    }

    private fun toHost(v: CPointer<hes_view>): HostTensor {
        val type = ElementType.of(hes_view_element_type(v))
        val shape = LongArray(hes_view_rank(v)) { hes_view_dim(v, it) }
        val bytes = ByteArray(hes_view_bytes(v).toInt())
        if (bytes.isNotEmpty()) {
            val rc = bytes.usePinned { hes_view_to_host(s, v, it.addressOf(0), bytes.size.convert()) }
            if (rc != 0) failure("read output")
        }
        return HostTensor(type, shape, bytes)
    }

    override fun close() {
        handle?.let { hes_session_release(it) }
        handle = null
    }
}

/** Architecture constants of a KV-cache contract (mirrors the cartridge manifest; see SKaiNET-transformers IreeKvSpec). */
data class KvSpec(
    val nLayers: Int, val headDim: Int, val nKvHeads: Int, val nHeads: Int, val hiddenSize: Int, val vocabSize: Int,
    val slidingWindow: Int, val globalLayerPeriod: Int, val chunk: Int,
    val slidingRopeBase: Float, val globalRopeBase: Float, val maskHeads: Int = 0,
) {
    companion object {
        /** FunctionGemma-270M, contract functiongemma-kv-v1. */
        fun functionGemma270m(chunk: Int = 32, vocabSize: Int = 262_144): KvSpec = KvSpec(
            nLayers = 18, headDim = 256, nKvHeads = 1, nHeads = 4, hiddenSize = 640, vocabSize = vocabSize,
            slidingWindow = 512, globalLayerPeriod = 6, chunk = chunk, slidingRopeBase = 10_000f, globalRopeBase = 1_000_000f,
        )
    }
}

/**
 * Stateful KV-cache decoder: prefill a constant prefix once, snapshot it, per utterance restore + one chunk call,
 * then one step per generated token. Every call returns the next token id. The cache stays on the device.
 */
@OptIn(ExperimentalForeignApi::class)
class IreeKv internal constructor(private var handle: CPointer<hes_kv>?, val spec: KvSpec) : AutoCloseable {
    private val k: CPointer<hes_kv> get() = handle ?: throw IreeException("kv closed")

    val position: Int get() = hes_kv_position(k)

    fun prefill(tokens: IntArray, n: Int = tokens.size): Int =
        tokens.usePinned { hes_kv_prefill(k, it.addressOf(0), tokens.size, n) }.also { if (it < 0) failure("prefill") }

    fun releasePrefill() = hes_kv_release_prefill(k)

    fun chunk(tokens: IntArray, n: Int = tokens.size): Int =
        tokens.usePinned { hes_kv_chunk(k, it.addressOf(0), n) }.also { if (it < 0) failure("chunk") }

    fun step(token: Int): Int = hes_kv_step(k, token).also { if (it < 0) failure("step") }

    fun snapshot(): Snapshot = Snapshot(hes_kv_snapshot_take(k) ?: failure("snapshot"))

    fun restore(snapshot: Snapshot) {
        if (hes_kv_restore(k, snapshot.ptr) != 0) failure("restore")
    }

    override fun close() {
        handle?.let { hes_kv_release(it) }
        handle = null
    }

    /** A retained cache state (zero-copy). */
    class Snapshot internal constructor(private var handle: CPointer<hes_kv_snapshot>?) : AutoCloseable {
        internal val ptr: CPointer<hes_kv_snapshot> get() = handle ?: throw IreeException("snapshot closed")
        override fun close() {
            handle?.let { hes_kv_snapshot_release(it) }
            handle = null
        }
    }
}

/** The eight files of one Moonshine v2 streaming model (cartridge pack layout, `artifacts/<role>/<lang>/`). */
data class MoonshineFiles(
    val frontend: String, val encoder: String, val adapter: String, val prefill: String, val step: String,
    val params: String, val vocab: String, val decEmbed: String,
)

/** The device of each stage; prefill and step share [decoder]. */
data class MoonshineDevices(val frontend: String, val encoder: String, val adapter: String, val decoder: String) {
    companion object {
        fun all(device: String): MoonshineDevices = MoonshineDevices(device, device, device, device)
    }
}

/** Geometry and policy; zeros mean the tiny model's defaults. */
data class MoonshineSpec(
    val nLayers: Int = 0, val headDim: Int = 0, val ropePairs: Int = 0, val dim: Int = 0, val encDim: Int = 0,
    /** Partial decodes are dropped once the stream is this far behind real time; 0 = runtime default, < 0 = never. */
    val lagBudgetMs: Int = 0,
    val fastFinish: Boolean = false,
    val warmUp: Boolean = true,
    val verbose: Boolean = false,
)

/**
 * Streaming Moonshine v2: feed 16 kHz mono PCM as it arrives, get cumulative partials, one exact final per
 * utterance. One utterance at a time; not thread-safe.
 */
@OptIn(ExperimentalForeignApi::class)
class IreeMoonshine internal constructor(private var handle: CPointer<hes_ms>?) : AutoCloseable {
    private val m: CPointer<hes_ms> get() = handle ?: throw IreeException("moonshine closed")

    /** Decoder width and vocabulary size the model was loaded with. */
    val dim: Int get() = hes_ms_dim(m)
    val vocabSize: Int get() = hes_ms_vocab(m)

    /** Returns the new cumulative partial, or `null` when it did not change. */
    fun feed(pcm: FloatArray): String? {
        if (pcm.isEmpty()) return null
        return pcm.usePinned { hes_ms_feed(m, it.addressOf(0), pcm.size) }?.toKString()
    }

    /** End of utterance: the final transcript ("" when nothing was recognized); resets for the next one. */
    fun finish(): String = hes_ms_finish(m)?.toKString() ?: failure("moonshine finish")

    fun reset() = hes_ms_reset(m)

    /** The last [finish]'s counters (flushMs, decodeMs, windows, hops, tokens, timeline, ...). */
    val stats: Map<String, String>
        get() = hes_ms_stats(m)?.toKString().orEmpty().split(',').mapNotNull { kv ->
            val i = kv.indexOf('=')
            if (i <= 0) null else kv.substring(0, i) to kv.substring(i + 1)
        }.toMap()

    override fun close() {
        handle?.let { hes_ms_release(it) }
        handle = null
    }
}
