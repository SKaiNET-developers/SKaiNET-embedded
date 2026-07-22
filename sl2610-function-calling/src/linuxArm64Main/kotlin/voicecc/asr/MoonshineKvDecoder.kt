package voicecc.asr

import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.time.TimeSource
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readByteArray
import platform.posix.getenv
import sk.ainet.apps.llm.tokenizer.GGUFTokenizer

/**
 * Moonshine-tiny STT on the SL2610 with the **KV-cache 2-graph decode** (perf-program Phase 6) —
 * the seq2seq analogue of [sk.ainet.transformers.gemma.iree.GemmaKvDecoder]. Replaces the re-decode
 * loop ([MoonshineDecoder], one static `[1,SEQ,·]` graph re-run per token) with prefill-once +
 * with_past-loop, processing 1 token/step over a growing self-cache.
 *
 *   wav → preprocessor.vmfb (CPU) → encoder.vmfb (NPU) → memory[1,F,288]bf16
 *   PREFILL   : embeds(START)[1,1,288] + memory → logits + per-layer self-K/V[1,8,1,36] + cross-K/V[1,8,F,36]
 *   WITH_PAST : token-embed[1,1,288] + cos/sin[1,36] + self-K/V[1,8,P,36] + cross-K/V[1,8,F,36]
 *               → logits[1,1,32768] + extended self-K/V[1,8,P+1,36]
 * Cross-K/V are computed ONCE in prefill (fixed to the encoder memory) and re-fed every step; self-K/V
 * grow. RoPE position is a runtime input (INTERLEAVED cos/sin, host-built). Embeds are host-side
 * (tied lm_head). K/V + cos/sin are bf16 (the decoder is traced bf16); logits are f32.
 *
 * ⚠️ BOARD-UNVERIFIED DRAFT (mirrors the Gemma loop; see docs/GEMMA-KV-BOARD-LOOP.md). Confirm on the
 * first board run, all caught by transcription correctness:
 *   1. [kFirstInOutput] — per-block K-vs-V output order (trace-terminal ordering; may be V,K).
 *   2. the vmfb entry function names (re-decode uses "main"; the KV graphs export as
 *      moonshine_decoder_prefill / _with_past — override via env if the compile renamed them).
 * Input arg order (token, cos, sin, then per-layer selfK,selfV,crossK,crossV) is trace-order.
 */
@OptIn(ExperimentalForeignApi::class)
internal class MoonshineKvDecoder(
    modelDir: String = "/home/root/sl2610-examples/models/Synaptics/moonshine-tiny-bf16-torq",
    encoderVmfb: String? = getenv("MOONSHINE_ENCODER_VMFB")?.toKString(),
    private val preprocVmfb: String = "/home/root/moon/preprocessor_cpu.vmfb",
    private val prefillVmfb: String = getenv("MOONSHINE_PREFILL_VMFB")?.toKString() ?: "/home/root/moon/decoder_prefill_cpu.vmfb",
    private val withPastVmfb: String = getenv("MOONSHINE_WITH_PAST_VMFB")?.toKString() ?: "/home/root/moon/decoder_with_past_cpu.vmfb",
    private val prefillFn: String = getenv("MOONSHINE_PREFILL_FN")?.toKString() ?: "moonshine_decoder_prefill",
    private val withPastFn: String = getenv("MOONSHINE_WITH_PAST_FN")?.toKString() ?: "moonshine_decoder_with_past",
    private val work: String = "/home/root/moon/rt",
    torqBin: String = "/home/root/sl2610-voice-cc/.venv/lib/python3.12/site-packages/torq/_runtime_libs/torq-run-module",
    torqLibs: String = "/home/root/sl2610-voice-cc/.venv/lib/python3.12/site-packages/torq/_runtime_libs:" +
        "/home/root/sl2610-voice-cc/.venv/lib/python3.12/site-packages/iree/_runtime_libs",
) {
    private val encVmfb = encoderVmfb ?: "$modelDir/encoder.vmfb"
    private val encDevice = getenv("MOONSHINE_ENCODER_DEVICE")?.toKString() ?: "torq"
    private val decDevice = getenv("MOONSHINE_DECODER_DEVICE")?.toKString() ?: "local-task"
    private val profile = getenv("VOICECC_PROFILE")?.toKString().let { it == "1" || it == "true" }
    private val torq = TorqRunModule(torqBin, torqLibs)
    private val emb = Bf16EmbeddingTable(getenv("MOONSHINE_EMBED")?.toKString() ?: "/home/root/moon/our_embed_tokens.npy", DIM)
    private val tokenizer: GGUFTokenizer = GGUFTokenizer.fromTokenizerJson(
        SystemFileSystem.source(Path("$modelDir/tokenizer.json")).buffered().use { it.readByteArray() }.decodeToString(),
    )

    init { if (encDevice == "torq") TorqRunModule.enableNpuClock() }

    fun transcribe(wav: String): String? {
        SystemFileSystem.createDirectories(Path(work))

        // 1-3) preproc (CPU) → encoder (NPU/CPU) → bf16 memory [1, FRAMES, 288].
        val audio = Wav.loadResampledPadded(wav, INPUT_LEN)
        Bin.writeBytes("$work/wav.bin", Bin.f32Bytes(audio))
        val feat = "$work/feat.bin"
        if (!torq.run(preprocVmfb, "main", "local-task",
                listOf(TorqRunModule.Spec("1x$INPUT_LEN", "f32", "$work/wav.bin")), listOf(feat))) return null
        Bin.writeBytes("$work/enc_in.bin", Bin.f32ToBf16(Bin.readBytes(feat)))
        val encRaw = "$work/enc_out.bin"
        if (!torq.run(encVmfb, "main", encDevice,
                listOf(TorqRunModule.Spec("1x288x$FRAMES", "bf16", "$work/enc_in.bin")), listOf(encRaw))) return null
        val encBytes = Bin.readBytes(encRaw)
        val memory = if (encBytes.size == FRAMES * DIM * 4) {
            Bin.writeBytes("$work/enc_mem.bin", Bin.f32ToBf16(encBytes)); "$work/enc_mem.bin"
        } else encRaw

        val genStart = if (profile) TimeSource.Monotonic.markNow() else null

        // 4) PREFILL over the START token: seed self-K/V (len 1) and the fixed cross-K/V (len FRAMES).
        Bin.writeBytes("$work/tok_embed.bin", emb.row(START))
        val preSelf = kvFiles("pre_self")            // 2*N: selfK/V per layer
        val preCross = kvFiles("pre_cross")          // 2*N: crossK/V per layer
        val preLogits = "$work/pre_logits.bin"
        if (!torq.run(prefillVmfb, prefillFn, decDevice,
                listOf(
                    TorqRunModule.Spec("1x1x$DIM", "bf16", "$work/tok_embed.bin"),
                    TorqRunModule.Spec("1x${FRAMES}x$DIM", "bf16", memory),
                ),
                // outputs: logits, then per-layer self-K/V, then per-layer cross-K/V (confirm order on board).
                listOf(preLogits) + preSelf + preCross)) return null
        var next = argmaxF32(Bin.readBytes(preLogits))
        // self cache grows; cross cache is fixed. Keep both as raw bf16 byte buffers per layer.
        var selfK = Array(N_LAYERS) { Bin.readBytes(kOf(preSelf, it)) }
        var selfV = Array(N_LAYERS) { Bin.readBytes(vOf(preSelf, it)) }
        val crossK = Array(N_LAYERS) { kOf(preCross, it) }   // file paths, reused every step
        val crossV = Array(N_LAYERS) { vOf(preCross, it) }

        // 5) DECODE: one token/step over the growing self-cache.
        val ids = arrayListOf(START)
        var pos = 1
        var step = 0
        while (step < MAX_NEW_TOKENS && next != END) {
            ids.add(next)
            val t0 = if (profile) TimeSource.Monotonic.markNow() else null
            // token embed + interleaved cos/sin at the runtime position (bf16).
            Bin.writeBytes("$work/wp_tok.bin", emb.row(next))
            val (c, s) = interleavedCosSin(pos)
            Bin.writeBytes("$work/wp_cos.bin", Bin.f32ToBf16(Bin.f32Bytes(c)))
            Bin.writeBytes("$work/wp_sin.bin", Bin.f32ToBf16(Bin.f32Bytes(s)))
            for (i in 0 until N_LAYERS) {
                Bin.writeBytes("$work/wp_sk_$i.bin", selfK[i]); Bin.writeBytes("$work/wp_sv_$i.bin", selfV[i])
            }
            val outSelf = kvFiles("wp_self")
            val outLogits = "$work/wp_logits.bin"
            if (!torq.run(withPastVmfb, withPastFn, decDevice, withPastInputs(pos, crossK, crossV), listOf(outLogits) + outSelf)) return null
            next = argmaxF32(Bin.readBytes(outLogits))
            selfK = Array(N_LAYERS) { Bin.readBytes(kOf(outSelf, it)) }
            selfV = Array(N_LAYERS) { Bin.readBytes(vOf(outSelf, it)) }
            if (t0 != null) println("[perf] moonshine-kv step $step: ${t0.elapsedNow().inWholeMilliseconds} ms")
            pos++; step++
        }
        if (genStart != null) {
            val total = genStart.elapsedNow().inWholeMilliseconds
            val n = if (ids.size > 1) ids.size - 1 else 1
            println("[perf] moonshine-kv total: $total ms, ${ids.size - 1} tokens, ${total / n} ms/token")
        }

        val out = ids.drop(1)
        return if (out.isEmpty()) "" else tokenizer.decode(out.toIntArray()).trim()
    }

    /** with_past input specs in trace-order: token embed, cos, sin, then per layer selfK,selfV,crossK,crossV. */
    private fun withPastInputs(pos: Int, crossK: Array<String>, crossV: Array<String>): List<TorqRunModule.Spec> {
        val self = "1x${N_HEADS}x${pos}x$HEAD_DIM"
        val cross = "1x${N_HEADS}x${FRAMES}x$HEAD_DIM"
        val specs = arrayListOf(
            TorqRunModule.Spec("1x1x$DIM", "bf16", "$work/wp_tok.bin"),
            TorqRunModule.Spec("1x$HEAD_DIM", "bf16", "$work/wp_cos.bin"),
            TorqRunModule.Spec("1x$HEAD_DIM", "bf16", "$work/wp_sin.bin"),
        )
        for (i in 0 until N_LAYERS) {
            specs += TorqRunModule.Spec(self, "bf16", "$work/wp_sk_$i.bin")
            specs += TorqRunModule.Spec(self, "bf16", "$work/wp_sv_$i.bin")
            specs += TorqRunModule.Spec(cross, "bf16", crossK[i])
            specs += TorqRunModule.Spec(cross, "bf16", crossV[i])
        }
        return specs
    }

    /** INTERLEAVED sign-baked cos/sin `[headDim]` at [position] (port of RoPE.buildInterleavedCosSin;
     *  Moonshine: partial rotary → rotaryDim 32, freqDenom = rotaryDim). */
    private fun interleavedCosSin(position: Int): Pair<FloatArray, FloatArray> {
        val half = HEAD_DIM / 2
        val c = FloatArray(HEAD_DIM); val s = FloatArray(HEAD_DIM)
        for (i in 0 until half) {
            val rot = i < HALF_ROTARY
            val cv = if (rot) cos(position * (1.0 / ROPE_BASE.toDouble().pow(2.0 * i / ROTARY_DIM))).toFloat() else 1f
            val sv = if (rot) sin(position * (1.0 / ROPE_BASE.toDouble().pow(2.0 * i / ROTARY_DIM))).toFloat() else 0f
            c[2 * i] = cv; c[2 * i + 1] = cv
            s[2 * i] = -sv; s[2 * i + 1] = sv
        }
        return c to s
    }

    /** Argmax over a `[1,1,vocab]` little-endian f32 logits buffer (single decode position). */
    private fun argmaxF32(b: ByteArray): Int {
        var best = 0
        var bestV = Float.NEGATIVE_INFINITY
        for (v in 0 until VOCAB) {
            val o = v * 4
            val bits = (b[o].toInt() and 0xFF) or ((b[o + 1].toInt() and 0xFF) shl 8) or
                ((b[o + 2].toInt() and 0xFF) shl 16) or ((b[o + 3].toInt() and 0xFF) shl 24)
            val f = Float.fromBits(bits)
            if (f > bestV) { bestV = f; best = v }
        }
        return best
    }

    // Output/staging file lists: 2*N per-layer K/V (block order, two per block).
    private fun kvFiles(tag: String): List<String> = (0 until 2 * N_LAYERS).map { "$work/${tag}_$it.bin" }
    private fun kOf(files: List<String>, layer: Int): String = files[2 * layer + if (kFirstInOutput) 0 else 1]
    private fun vOf(files: List<String>, layer: Int): String = files[2 * layer + if (kFirstInOutput) 1 else 0]

    private companion object {
        const val DIM = 288
        const val VOCAB = 32768
        const val N_LAYERS = 6
        const val N_HEADS = 8
        const val HEAD_DIM = 36
        const val ROTARY_DIM = 32          // headDim 36 * partialRotary 0.9 -> 32 (even)
        const val HALF_ROTARY = ROTARY_DIM / 2
        const val ROPE_BASE = 10000f
        const val FRAMES = 207
        const val INPUT_LEN = 80000
        const val START = 1
        const val END = 2
        const val MAX_NEW_TOKENS = 30
        // Per-block K-vs-V output order; false = (V,K) per the return-SSA analysis. Flip if the first
        // board run mis-transcribes (the only thing this controls). See docs/GEMMA-KV-BOARD-LOOP.md.
        const val kFirstInOutput = false
    }
}
