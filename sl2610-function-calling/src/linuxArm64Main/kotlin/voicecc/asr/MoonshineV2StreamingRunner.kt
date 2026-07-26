package voicecc.asr

import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readByteArray
import platform.posix.getenv
import sk.ainet.apps.llm.tokenizer.GGUFTokenizer

/**
 * Moonshine **v2** streaming ASR runtime (plan item P6). Drives the self-compiled, chunk-shaped v2 vmfbs
 * INCREMENTALLY over a rolling feature-frame buffer, instead of the v1 per-utterance fixed clip. See
 * `docs/MOONSHINE-V2-STREAMING.md` for the design; this is the board-side state machine.
 *
 * Full pipeline:
 * ```
 *   audio 20ms frames → [frontend] → [v2 encoder] → [v2 adapter] → [decoder prefill] → [decoder with_past] → tokens
 *      (50 Hz)          conv+state   sliding-window   +learned pos    logits+self/cross    step over self-cache
 *                       @main_graph  @main (DSL)       @main (DSL)     K/V (@main, DSL)     (@main, DSL)
 * ```
 * The **decoder is the SELF-COMPILED DSL** `moonshineV2Decoder` — its KV-cached two-graph export (prefill +
 * with_past, transformers #257), NOT the vendor ONNX decoder_kv/cross_kv. Only the **frontend** is still
 * vendor-ONNX-compiled (`@main_graph`); everything else is DSL-authored + self-compiled (`@main`). The whole
 * pipeline runs on IREE (no onnxruntime / vendor runtime).
 *
 * ## Bounded-window finalization (the streaming contract)
 * The encoder is position-free with a sliding window of [WINDOW] left + [LOOKAHEAD] right context. A frame's
 * output is **provisional** until its [LOOKAHEAD] right-context frames arrive, then it **finalizes** and never
 * recomputes. With a fixed [CHUNK]-frame graph we slide the window by [HOP] = CHUNK − WINDOW − LOOKAHEAD, so
 * each pass finalizes the band `[start+WINDOW, start+WINDOW+HOP)`. [finish] pads the tail and finalizes the rest.
 *
 * ## Decode (DSL prefill + with_past — per-layer K/V, like [MoonshineKvDecoder])
 * **PREFILL** `embeds(BOS)[1,1,DIM] + memory[1,F,DIM]` → per-layer `selfK/V[1,H,1,HD]` + `crossK/V[1,H,F,HD]`
 * + `logits[1,1,VOCAB]` (25 outputs: 4·L KV interleaved per layer, then logits). Then **WITH_PAST** steps
 * autoregressively: `tokenEmbed[1,1,DIM] + cos/sin[1,HD] + per-layer selfK/V[1,H,P,HD] + crossK/V` →
 * per-layer extended `selfK/V[1,H,P+1,HD]` + `logits` (13 outputs: 2·L self-K/V, then logits). Token embedding
 * and RoPE cos/sin are host-side (tied lm_head; interleaved partial-rotary, rotaryDim 32 — validated cos-sim
 * 1.0 vs ONNX). Greedy argmax; stops at [EOS] or [MAX_NEW]. Cross-K/V are computed once in prefill and re-fed.
 *
 * ⚠️ BOARD-UNVERIFIED. Verify on the first board run (all surface as transcription errors):
 *   1. frontend streaming-state threading (conv buffers' trailing dim is dynamic — recomputed from output bytes).
 *   2. per-layer K/V output ORDER (here: interleaved `[sK,sV,cK,cV]`/layer for prefill, `[sK,sV]`/layer for
 *      with_past, logits last — matching the compiled MLIR signatures) and the RoPE position convention.
 *   3. prefill/with_past over the GROWING finalized memory — decoded once per [transcribe]; windowing is a follow-up.
 */
@OptIn(ExperimentalForeignApi::class)
internal class MoonshineV2StreamingRunner(
    // DSL-authored, self-compiled (entry @main):
    private val encoderVmfb: String =
        getenv("MOONSHINE_V2_ENCODER_VMFB")?.toKString() ?: "/home/root/moon/moonshine-v2-encoder-cpu.vmfb",
    private val adapterVmfb: String =
        getenv("MOONSHINE_V2_ADAPTER_VMFB")?.toKString() ?: "/home/root/moon/moonshine-v2-adapter-cpu.vmfb",
    // self-compiled DSL decoder — KV-cached two-graph export (entry @main):
    private val prefillVmfb: String =
        getenv("MOONSHINE_V2_PREFILL_VMFB")?.toKString() ?: "/home/root/moon/moonshine-v2-dec-prefill-cpu.vmfb",
    private val withPastVmfb: String =
        getenv("MOONSHINE_V2_WITHPAST_VMFB")?.toKString() ?: "/home/root/moon/moonshine-v2-dec-withpast-cpu.vmfb",
    // token embedding table (dec_embed.weight [VOCAB,DIM] f32, from bake_moonshine_v2_decoder.py) — host-side lookup:
    private val embedPath: String =
        getenv("MOONSHINE_V2_EMBED")?.toKString() ?: "/home/root/moon/dec_embed.weight.bin",
    // vendor-ONNX-compiled frontend (entry @main_graph) — the one remaining non-DSL graph:
    private val frontendVmfb: String =
        getenv("MOONSHINE_V2_FRONTEND_VMFB")?.toKString() ?: "/home/root/moon/moonshine-v2-frontend-cpu.vmfb",
    private val tokenizerJson: String? = getenv("MOONSHINE_V2_TOKENIZER")?.toKString(),
    private val device: String = getenv("MOONSHINE_V2_DEVICE")?.toKString() ?: "local-task",
    private val work: String = getenv("MOONSHINE_V2_WORK")?.toKString() ?: "/home/root/moon/v2rt",
    torqBin: String = "/home/root/sl2610-voice-cc/.venv/lib/python3.12/site-packages/torq/_runtime_libs/torq-run-module",
    torqLibs: String = "/home/root/sl2610-voice-cc/.venv/lib/python3.12/site-packages/torq/_runtime_libs:" +
        "/home/root/sl2610-voice-cc/.venv/lib/python3.12/site-packages/iree/_runtime_libs",
) {
    private val torq = TorqRunModule(torqBin, torqLibs)
    private val tokenizer: GGUFTokenizer? = tokenizerJson?.let {
        GGUFTokenizer.fromTokenizerJson(
            SystemFileSystem.source(Path(it)).buffered().use { s -> s.readByteArray() }.decodeToString(),
        )
    }
    // Host-side token-embedding lookup: dec_embed.weight [VOCAB,DIM] f32; row(t) → [1,1,DIM] embed bytes.
    private val embedTable: ByteArray by lazy { Bin.readBytes(embedPath) }
    private fun embedRow(token: Int): ByteArray {
        val rowBytes = DIM * 4
        return embedTable.copyOfRange(token * rowBytes, token * rowBytes + rowBytes)
    }

    // Rolling encoder state.
    private val featBuf = ArrayList<Float>()
    private var producedFrames = 0
    private var finalizedFrames = 0
    private var chunkStart = 0

    /** Finalized adapted-memory rows (DIM each), in frame order — the decoder's cross-attention input. */
    val finalizedMemory = ArrayList<Float>()

    // Frontend streaming state (threaded across [feedAudio] calls). Trailing conv dims are dynamic.
    private var conv1Trail = 4
    private var conv2Trail = 4

    // Frontend state files (persisted across feedAudio calls).
    private val fSampleBuf = "$work/state_sb.bin"
    private val fSampleLen = "$work/state_sl.bin"
    private val fConv1 = "$work/state_c1.bin"
    private val fConv2 = "$work/state_c2.bin"
    private val fFrameCount = "$work/state_fc.bin"

    init {
        SystemFileSystem.createDirectories(Path(work))
        // Seed the frontend state buffers with zeros (first chunk has no carry-over).
        Bin.writeBytes(fSampleBuf, ByteArray(SAMPLE_BUF * 4))
        Bin.writeBytes(fSampleLen, Bin.i64Bytes(intArrayOf(0)))
        Bin.writeBytes(fConv1, ByteArray(DIM * conv1Trail * 4))
        Bin.writeBytes(fConv2, ByteArray(FFN_CH * conv2Trail * 4))
        Bin.writeBytes(fFrameCount, Bin.i64Bytes(intArrayOf(0)))
    }

    // ---- SEAM 1: frontend (audio → feature frames + streaming conv state) ----

    /** Feed newly-arrived 16 kHz audio samples; runs the conv frontend (threading its streaming state) and
     *  pushes the produced feature frames through the encoder/adapter pipeline. */
    fun feedAudio(samples: FloatArray) {
        Bin.writeBytes("$work/f_audio.bin", Bin.f32Bytes(samples))
        val out = listOf(
            "$work/f_feat.bin", "$work/f_sb.bin", "$work/f_sl.bin", "$work/f_c1.bin", "$work/f_c2.bin", "$work/f_fc.bin",
        )
        val inputs = listOf(
            TorqRunModule.Spec("1x${samples.size}", "f32", "$work/f_audio.bin"),
            TorqRunModule.Spec("1x$SAMPLE_BUF", "f32", fSampleBuf),
            TorqRunModule.Spec("1", "i64", fSampleLen),
            TorqRunModule.Spec("1x${DIM}x$conv1Trail", "f32", fConv1),
            TorqRunModule.Spec("1x${FFN_CH}x$conv2Trail", "f32", fConv2),
            TorqRunModule.Spec("1", "i64", fFrameCount),
        )
        if (!torq.run(frontendVmfb, ONNX_FN, device, inputs, out)) { println("[v2] frontend failed"); return }

        // Thread state forward (conv trailing dims are dynamic → recompute from output byte size).
        Bin.writeBytes(fSampleBuf, Bin.readBytes(out[1]))
        Bin.writeBytes(fSampleLen, Bin.readBytes(out[2]))
        val c1 = Bin.readBytes(out[3]); Bin.writeBytes(fConv1, c1); conv1Trail = c1.size / (DIM * 4)
        val c2 = Bin.readBytes(out[4]); Bin.writeBytes(fConv2, c2); conv2Trail = c2.size / (FFN_CH * 4)
        Bin.writeBytes(fFrameCount, Bin.readBytes(out[5]))

        val feat = Bin.readF32(out[0])           // [featLen * DIM]
        if (feat.isNotEmpty()) feed(feat)
    }

    // ---- encoder + adapter pipeline (feature frames → finalized adapted memory) ----

    /** Feed FEATURE frames directly (`n * DIM` f32) — the frontend output, or a pre-extracted feature stream. */
    fun feed(frames: FloatArray) {
        require(frames.size % DIM == 0) { "frames must be a whole number of $DIM-wide feature rows" }
        for (v in frames) featBuf.add(v)
        producedFrames += frames.size / DIM
        drain(end = false)
    }

    /** Flush the tail (pad the final window) and finalize everything remaining. Returns total finalized frames. */
    fun finish(): Int {
        drain(end = true)
        return finalizedFrames
    }

    private fun drain(end: Boolean) {
        while (producedFrames - chunkStart >= CHUNK) {
            processWindow(chunkStart, end = false)
            chunkStart += HOP
        }
        if (end && finalizedFrames < producedFrames) {
            processWindow(start = maxOf(0, producedFrames - CHUNK), end = true)
        }
    }

    private fun processWindow(start: Int, end: Boolean) {
        val chunk = FloatArray(CHUNK * DIM)
        val avail = (producedFrames - start).coerceAtMost(CHUNK)
        for (i in 0 until avail * DIM) chunk[i] = featBuf[start * DIM + i]

        val chunkFile = "$work/enc_in.bin"; Bin.writeBytes(chunkFile, Bin.f32Bytes(chunk))
        val memFile = "$work/enc_out.bin"
        if (!torq.run(encoderVmfb, ENTRY_FN, device,
                listOf(TorqRunModule.Spec("1x${CHUNK}x$DIM", "f32", chunkFile)), listOf(memFile))) {
            println("[v2] encoder failed at frame $start"); return
        }

        val positions = IntArray(CHUNK) { start + it }
        val posFile = "$work/adp_pos.bin"; Bin.writeBytes(posFile, Bin.i32Bytes(positions))
        val adaFile = "$work/adp_out.bin"
        if (!torq.run(adapterVmfb, ENTRY_FN, device,
                listOf(
                    TorqRunModule.Spec("1x$CHUNK", "i32", posFile),
                    TorqRunModule.Spec("1x${CHUNK}x$DIM", "f32", memFile),
                ), listOf(adaFile))) {
            println("[v2] adapter failed at frame $start"); return
        }
        val adapted = Bin.readF32(adaFile)

        val newFinal = if (end) producedFrames else minOf(start + CHUNK - LOOKAHEAD, producedFrames)
        for (f in finalizedFrames until newFinal) {
            val local = f - start
            if (local < 0 || local >= CHUNK) continue
            val base = local * DIM
            for (d in 0 until DIM) finalizedMemory.add(adapted[base + d])
        }
        if (newFinal > finalizedFrames) finalizedFrames = newFinal
    }

    // ---- SEAM 2: decode (DSL prefill once + with_past autoregressive loop, per-layer K/V) ----

    /** Decode the finalized memory to token ids. Runs the DSL decoder prefill once (seeds self + cross K/V),
     *  then greedy with_past steps over the growing per-layer self-cache. */
    fun decodeTokens(maxNew: Int = MAX_NEW): List<Int> {
        val f = finalizedFrames
        if (f == 0) return emptyList()
        val memFile = "$work/dec_mem.bin"
        Bin.writeBytes(memFile, Bin.f32Bytes(FloatArray(f * DIM) { finalizedMemory[it] }))

        // PREFILL over the START token: embeds[1,1,DIM] + memory → per-layer [selfK,selfV,crossK,crossV] + logits.
        Bin.writeBytes("$work/pre_emb.bin", embedRow(BOS))
        val preFiles = (0 until N_LAYERS).flatMap { l ->
            listOf("$work/pre_sk_$l.bin", "$work/pre_sv_$l.bin", "$work/pre_ck_$l.bin", "$work/pre_cv_$l.bin")
        } + "$work/pre_logits.bin"
        if (!torq.run(prefillVmfb, ENTRY_FN, device,
                listOf(
                    TorqRunModule.Spec("1x1x$DIM", "f32", "$work/pre_emb.bin"),
                    TorqRunModule.Spec("1x${f}x$DIM", "f32", memFile),
                ), preFiles)) {
            println("[v2] decoder prefill failed"); return emptyList()
        }
        // self K/V grow (start at the prefill's len-1 cache); cross K/V are fixed (re-fed every step).
        val selfK = Array(N_LAYERS) { Bin.readBytes("$work/pre_sk_$it.bin") }
        val selfV = Array(N_LAYERS) { Bin.readBytes("$work/pre_sv_$it.bin") }
        val crossK = Array(N_LAYERS) { "$work/pre_ck_$it.bin" }   // paths, reused each step
        val crossV = Array(N_LAYERS) { "$work/pre_cv_$it.bin" }
        var next = Bin.argmaxF32Row(Bin.readBytes("$work/pre_logits.bin"), row = 0, cols = VOCAB)

        val ids = ArrayList<Int>()
        var pastLen = 1   // the BOS self-cache from prefill
        var step = 0
        while (step < maxNew && next != EOS) {
            ids.add(next)
            Bin.writeBytes("$work/wp_emb.bin", embedRow(next))
            val (c, s) = interleavedCosSin(pastLen)   // RoPE at the current decode position
            Bin.writeBytes("$work/wp_cos.bin", Bin.f32Bytes(c))
            Bin.writeBytes("$work/wp_sin.bin", Bin.f32Bytes(s))
            for (l in 0 until N_LAYERS) {
                Bin.writeBytes("$work/wp_sk_$l.bin", selfK[l]); Bin.writeBytes("$work/wp_sv_$l.bin", selfV[l])
            }
            // inputs: token, cos, sin, then per-layer [selfK, selfV, crossK, crossV].
            val selfShape = "1x${N_HEADS}x${pastLen}x$HEAD_DIM"
            val crossShape = "1x${N_HEADS}x${f}x$HEAD_DIM"
            val inputs = arrayListOf(
                TorqRunModule.Spec("1x1x$DIM", "f32", "$work/wp_emb.bin"),
                TorqRunModule.Spec("1x$HEAD_DIM", "f32", "$work/wp_cos.bin"),
                TorqRunModule.Spec("1x$HEAD_DIM", "f32", "$work/wp_sin.bin"),
            )
            for (l in 0 until N_LAYERS) {
                inputs += TorqRunModule.Spec(selfShape, "f32", "$work/wp_sk_$l.bin")
                inputs += TorqRunModule.Spec(selfShape, "f32", "$work/wp_sv_$l.bin")
                inputs += TorqRunModule.Spec(crossShape, "f32", crossK[l])
                inputs += TorqRunModule.Spec(crossShape, "f32", crossV[l])
            }
            // outputs: per-layer [newSelfK, newSelfV], then logits.
            val outs = (0 until N_LAYERS).flatMap { listOf("$work/wp_nsk_$it.bin", "$work/wp_nsv_$it.bin") } +
                "$work/wp_logits.bin"
            if (!torq.run(withPastVmfb, ENTRY_FN, device, inputs, outs)) {
                println("[v2] decoder with_past step $step failed"); break
            }
            next = Bin.argmaxF32Row(Bin.readBytes("$work/wp_logits.bin"), row = 0, cols = VOCAB)
            for (l in 0 until N_LAYERS) {
                selfK[l] = Bin.readBytes("$work/wp_nsk_$l.bin"); selfV[l] = Bin.readBytes("$work/wp_nsv_$l.bin")
            }
            pastLen++; step++
        }
        return ids
    }

    /** INTERLEAVED sign-baked RoPE cos/sin `[HEAD_DIM]` at [position] — partial rotary (rotaryDim [ROTARY_DIM],
     *  the trailing head dims pass through), matching the DSL decoder's RoPE (validated cos-sim 1.0 vs ONNX). */
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

    /** End-to-end: decode the finalized memory and detokenize (raw ids if no tokenizer is configured). */
    fun transcribe(maxNew: Int = MAX_NEW): String {
        val ids = decodeTokens(maxNew)
        if (ids.isEmpty()) return ""
        return tokenizer?.decode(ids.toIntArray())?.trim() ?: ids.joinToString(" ")
    }

    private companion object {
        // Matches the compiled v2 tiny-streaming vmfbs + MoonshineV2Config + streaming_config.json.
        const val DIM = 320
        const val FFN_CH = 640      // c1 — conv2_buffer channel count (frontend state)
        const val SAMPLE_BUF = 79   // frontend sample_buffer width
        const val CHUNK = 64
        const val WINDOW = 16
        const val LOOKAHEAD = 4
        const val HOP = CHUNK - WINDOW - LOOKAHEAD
        const val N_LAYERS = 6
        const val N_HEADS = 8
        const val HEAD_DIM = 40
        const val ROTARY_DIM = 32          // partialRotaryFactor 0.8 * headDim 40 (rotary.inv_freq has 16 entries)
        const val HALF_ROTARY = ROTARY_DIM / 2
        const val ROPE_BASE = 10000f
        const val VOCAB = 32768
        const val BOS = 1
        const val EOS = 2
        const val MAX_NEW = 64
        const val ENTRY_FN = "main"        // DSL-authored + self-compiled (encoder, adapter, decoder prefill/with_past)
        const val ONNX_FN = "main_graph"   // vendor-ONNX-compiled frontend (the one remaining non-DSL graph)
    }
}
