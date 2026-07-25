package voicecc.asr

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
 * Full pipeline (all five graphs now compiled to CPU vmfbs — `scripts/compile-moonshine-v2*.sh`):
 * ```
 *   audio 20ms frames → [frontend] → [v2 encoder] → [v2 adapter] → [cross_kv] → [decoder_kv] → tokens
 *      (50 Hz)          conv+state   sliding-window   +learned pos    per-layer      prefill/step
 *                       @main_graph  @main (DSL)       @main (DSL)     cross K/V      over self-cache
 * ```
 * Encoder + adapter are DSL-authored + self-compiled (`@main`); frontend/cross_kv/decoder_kv are compiled from
 * the vendor float ONNX (`@main_graph`) — the whole pipeline runs on IREE (no onnxruntime / vendor runtime).
 *
 * ## Bounded-window finalization (the streaming contract)
 * The encoder is position-free with a sliding window of [WINDOW] left + [LOOKAHEAD] right context. A frame's
 * output is **provisional** until its [LOOKAHEAD] right-context frames arrive, then it **finalizes** and never
 * recomputes. With a fixed [CHUNK]-frame graph we slide the window by [HOP] = CHUNK − WINDOW − LOOKAHEAD, so
 * each pass finalizes the band `[start+WINDOW, start+WINDOW+HOP)`. [finish] pads the tail and finalizes the rest.
 *
 * ## Decode (cross_kv + decoder_kv)
 * [cross_kv] runs ONCE over the finalized adapted memory → per-layer cross K/V `[6,1,8,F,40]`. Then [decoder_kv]
 * steps autoregressively: `token[1,1] i64 + self-K/V[6,1,8,P,40] + cross-K/V` → `logits[1,1,32768]` + grown
 * self-K/V. Greedy argmax; stops at [EOS] or [MAX_NEW]. (This is the v2 seq2seq analogue of [MoonshineKvDecoder];
 * the 6 layers are stacked in the leading dim, so no per-layer file split.)
 *
 * ⚠️ BOARD-UNVERIFIED. Verify on the first board run (all surface as transcription errors):
 *   1. frontend streaming-state threading (the conv buffers' trailing dim is dynamic — recomputed from output
 *      byte size here) and the empty (length-0) initial self-cache for decoder_kv's first step.
 *   2. input arg order + dtypes vs the compiled vmfbs (token is i64; caches f32; adapter positions-then-memory).
 *   3. cross_kv over the GROWING finalized memory — recomputed per [transcribe]; windowing it is a follow-up.
 */
@OptIn(ExperimentalForeignApi::class)
internal class MoonshineV2StreamingRunner(
    // DSL-authored, self-compiled (entry @main):
    private val encoderVmfb: String =
        getenv("MOONSHINE_V2_ENCODER_VMFB")?.toKString() ?: "/home/root/moon/moonshine-v2-encoder-cpu.vmfb",
    private val adapterVmfb: String =
        getenv("MOONSHINE_V2_ADAPTER_VMFB")?.toKString() ?: "/home/root/moon/moonshine-v2-adapter-cpu.vmfb",
    // vendor-ONNX-compiled (entry @main_graph):
    private val frontendVmfb: String =
        getenv("MOONSHINE_V2_FRONTEND_VMFB")?.toKString() ?: "/home/root/moon/moonshine-v2-frontend-cpu.vmfb",
    private val crossKvVmfb: String =
        getenv("MOONSHINE_V2_CROSS_KV_VMFB")?.toKString() ?: "/home/root/moon/moonshine-v2-cross_kv-cpu.vmfb",
    private val decoderVmfb: String =
        getenv("MOONSHINE_V2_DECODER_VMFB")?.toKString() ?: "/home/root/moon/moonshine-v2-decoder_kv-cpu.vmfb",
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

    // ---- SEAM 2: decode (cross_kv once + decoder_kv autoregressive loop) ----

    /** Decode the finalized memory to token ids. Runs cross_kv once, then greedy decoder_kv steps. */
    fun decodeTokens(maxNew: Int = MAX_NEW): List<Int> {
        val f = finalizedFrames
        if (f == 0) return emptyList()

        // memory [1, F, DIM] → cross_kv → per-layer cross K/V [6,1,8,F,40] (one tensor each, layers stacked).
        val memFile = "$work/dec_mem.bin"
        Bin.writeBytes(memFile, Bin.f32Bytes(FloatArray(f * DIM) { finalizedMemory[it] }))
        val kCross = "$work/dec_kcross.bin"; val vCross = "$work/dec_vcross.bin"
        if (!torq.run(crossKvVmfb, ONNX_FN, device,
                listOf(TorqRunModule.Spec("1x${f}x$DIM", "f32", memFile)), listOf(kCross, vCross))) {
            println("[v2] cross_kv failed"); return emptyList()
        }
        val crossShape = "${N_LAYERS}x1x${N_HEADS}x${f}x$HEAD_DIM"

        // autoregressive greedy decode over a growing self-cache (starts empty at length 0).
        val ids = ArrayList<Int>()
        val kSelf = "$work/dec_kself.bin"; val vSelf = "$work/dec_vself.bin"
        Bin.writeBytes(kSelf, ByteArray(0)); Bin.writeBytes(vSelf, ByteArray(0))
        var pastLen = 0
        var token = BOS
        var step = 0
        while (step < maxNew) {
            Bin.writeBytes("$work/dec_tok.bin", Bin.i64Bytes(intArrayOf(token)))
            val selfShape = "${N_LAYERS}x1x${N_HEADS}x${pastLen}x$HEAD_DIM"
            val inputs = listOf(
                TorqRunModule.Spec("1x1", "i64", "$work/dec_tok.bin"),
                TorqRunModule.Spec(selfShape, "f32", kSelf),
                TorqRunModule.Spec(selfShape, "f32", vSelf),
                TorqRunModule.Spec(crossShape, "f32", kCross),
                TorqRunModule.Spec(crossShape, "f32", vCross),
            )
            // outputs: logits, grown self-K/V, then the cross-K/V passthrough (ignored — reuse the inputs).
            val outs = listOf("$work/dec_logits.bin", "$work/dec_oksf.bin", "$work/dec_ovsf.bin",
                "$work/dec_ock.bin", "$work/dec_ocv.bin")
            if (!torq.run(decoderVmfb, ONNX_FN, device, inputs, outs)) { println("[v2] decoder step $step failed"); break }

            val next = Bin.argmaxF32Row(Bin.readBytes("$work/dec_logits.bin"), row = 0, cols = VOCAB)
            if (next == EOS) break
            ids.add(next)
            Bin.writeBytes(kSelf, Bin.readBytes("$work/dec_oksf.bin"))
            Bin.writeBytes(vSelf, Bin.readBytes("$work/dec_ovsf.bin"))
            pastLen += 1
            token = next
            step++
        }
        return ids
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
        const val VOCAB = 32768
        const val BOS = 1
        const val EOS = 2
        const val MAX_NEW = 64
        const val ENTRY_FN = "main"        // DSL-authored encoder/adapter
        const val ONNX_FN = "main_graph"   // vendor-ONNX-compiled frontend/cross_kv/decoder_kv
    }
}
