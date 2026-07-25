package voicecc.asr

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import platform.posix.getenv

/**
 * Moonshine **v2** streaming ASR runtime (plan item P6). Drives the self-compiled, chunk-shaped v2 vmfbs
 * INCREMENTALLY over a rolling feature-frame buffer, instead of the v1 per-utterance fixed clip. See
 * `docs/MOONSHINE-V2-STREAMING.md` for the design; this is the board-side state machine.
 *
 * Pipeline (all pieces DSL-authored + self-compiled — `scripts/compile-moonshine-v2.sh`):
 * ```
 *   audio 20ms frames → [frontend] → [v2 encoder] → [v2 adapter] → [v2 KV decoder] → provisional/final tokens
 *      (50 Hz)          conv+state   sliding-window   +learned pos-embed   prefill + with_past
 * ```
 * The **encoder** and **adapter** vmfbs exist and are verified (fixed [CHUNK] window, CPU llvm-cpu). Two
 * pieces are explicit SEAMS pending their own compiles (marked below): the **frontend** (audio→features,
 * carries streaming conv state — `frontend_state_shapes`) and the **v2 KV decoder** (`decoder_kv` — reuse
 * the [MoonshineKvDecoder] shape once compiled). So this scaffold takes FEATURE frames as input and produces
 * the **finalized adapted memory** (the decoder's cross-attention input); wiring the decode loop + emitting
 * text is the next step.
 *
 * ## Bounded-window finalization (the streaming contract)
 * The encoder is position-free with a sliding window of [WINDOW] left + [LOOKAHEAD] right context. A frame's
 * output is **provisional** until its [LOOKAHEAD] right-context frames arrive, then it **finalizes** and never
 * recomputes. With a fixed [CHUNK]-frame graph we slide the window by [HOP] = CHUNK − WINDOW − LOOKAHEAD, so
 * each pass finalizes the band `[start+WINDOW, start+WINDOW+HOP)` (WINDOW left + LOOKAHEAD right, all inside
 * the CHUNK). The first pass (`start==0`) finalizes from 0 (the edge layers handle the missing left context);
 * [finish] pads the tail and finalizes the remainder.
 *
 * ⚠️ BOARD-UNVERIFIED SCAFFOLD. Verify on the first board run (all surface as transcription errors):
 *   1. encoder/adapter input **arg order + shape** vs the compiled vmfbs (`1x{CHUNK}x{DIM}` f32; adapter is
 *      positions `1x{CHUNK}` i32 THEN memory — matches `moonshine-v2-adapter.mlir`).
 *   2. first/last-chunk edge handling (no left context at 0; tail padding at [finish]).
 *   3. the frontend streaming-state threading + the v2 decoder cross-memory windowing (the two seams).
 */
@OptIn(ExperimentalForeignApi::class)
internal class MoonshineV2StreamingRunner(
    private val encoderVmfb: String =
        getenv("MOONSHINE_V2_ENCODER_VMFB")?.toKString() ?: "/home/root/moon/moonshine-v2-encoder-cpu.vmfb",
    private val adapterVmfb: String =
        getenv("MOONSHINE_V2_ADAPTER_VMFB")?.toKString() ?: "/home/root/moon/moonshine-v2-adapter-cpu.vmfb",
    private val device: String = getenv("MOONSHINE_V2_DEVICE")?.toKString() ?: "local-task",
    private val work: String = getenv("MOONSHINE_V2_WORK")?.toKString() ?: "/home/root/moon/v2rt",
    torqBin: String = "/home/root/sl2610-voice-cc/.venv/lib/python3.12/site-packages/torq/_runtime_libs/torq-run-module",
    torqLibs: String = "/home/root/sl2610-voice-cc/.venv/lib/python3.12/site-packages/torq/_runtime_libs:" +
        "/home/root/sl2610-voice-cc/.venv/lib/python3.12/site-packages/iree/_runtime_libs",
) {
    private val torq = TorqRunModule(torqBin, torqLibs)

    // Rolling state. featBuf holds all feature frames seen so far (flat, DIM per frame); dropping finalized
    // frames to bound memory is an optimization TODO (a demo utterance fits comfortably).
    private val featBuf = ArrayList<Float>()
    private var producedFrames = 0        // feature frames appended so far
    private var finalizedFrames = 0       // frames finalized + adapted (never recomputed)
    private var chunkStart = 0            // absolute frame index where the next encoder window begins

    /** Finalized adapted-memory rows (DIM each), in frame order — the decoder's cross-attention input. */
    val finalizedMemory = ArrayList<Float>()

    init { SystemFileSystem.createDirectories(Path(work)) }

    /** Feed newly-arrived FEATURE frames (`n * DIM` f32, row-major). Runs as many full chunks as are ready. */
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
        // Run every full CHUNK window that is ready.
        while (producedFrames - chunkStart >= CHUNK) {
            processWindow(chunkStart, end = false)
            chunkStart += HOP
        }
        // Final flush: encode a last CHUNK (zero-padded) covering the not-yet-finalized tail.
        if (end && finalizedFrames < producedFrames) {
            processWindow(start = maxOf(0, producedFrames - CHUNK), end = true)
        }
    }

    /** Encode + adapt the CHUNK window at [start]; finalize the frames whose context is now complete. */
    private fun processWindow(start: Int, end: Boolean) {
        // 1) slice CHUNK feature frames [start, start+CHUNK), zero-padding past the buffer (final flush).
        val chunk = FloatArray(CHUNK * DIM)
        val avail = (producedFrames - start).coerceAtMost(CHUNK)
        for (i in 0 until avail * DIM) chunk[i] = featBuf[start * DIM + i]

        // 2) encoder: [1, CHUNK, DIM] f32 -> memory [1, CHUNK, DIM] f32 (position-free, windowed attention).
        val chunkFile = "$work/enc_in.bin"; Bin.writeBytes(chunkFile, Bin.f32Bytes(chunk))
        val memFile = "$work/enc_out.bin"
        if (!torq.run(encoderVmfb, ENTRY_FN, device,
                listOf(TorqRunModule.Spec("1x${CHUNK}x$DIM", "f32", chunkFile)), listOf(memFile))) {
            println("[v2] encoder failed at frame $start"); return
        }

        // 3) adapter: (positions [1,CHUNK] i32, memory [1,CHUNK,DIM] f32) -> adapted [1,CHUNK,DIM] f32.
        //    Positions are ABSOLUTE frame indices (the learned pos-embed table is indexed by them).
        val positions = IntArray(CHUNK) { start + it }
        val posFile = "$work/adp_pos.bin"; Bin.writeBytes(posFile, Bin.i32Bytes(positions))
        val adaFile = "$work/adp_out.bin"
        if (!torq.run(adapterVmfb, ENTRY_FN, device,
                listOf(
                    TorqRunModule.Spec("1x$CHUNK", "i32", posFile),      // arg0: positions
                    TorqRunModule.Spec("1x${CHUNK}x$DIM", "f32", memFile), // arg1: encoder memory
                ), listOf(adaFile))) {
            println("[v2] adapter failed at frame $start"); return
        }
        val adapted = Bin.readF32(adaFile)   // [CHUNK, DIM] row-major

        // 4) finalize the band whose right-context is complete. For end, finalize the whole remaining tail.
        val newFinal = if (end) producedFrames else minOf(start + CHUNK - LOOKAHEAD, producedFrames)
        for (f in finalizedFrames until newFinal) {
            val local = f - start                        // row within THIS window's adapted output
            if (local < 0 || local >= CHUNK) continue    // safety (shouldn't happen with the HOP arithmetic)
            val base = local * DIM
            for (d in 0 until DIM) finalizedMemory.add(adapted[base + d])
        }
        if (newFinal > finalizedFrames) {
            // SEAM: hand the newly-finalized adapted memory to the v2 KV decoder (decoder_kv vmfb, pending
            // compile). Prefill on the first finalized chunk, then with_past steps; re-feed/window the cross
            // K/V as `finalizedMemory` grows — mirrors MoonshineKvDecoder over v2 dims. Emit provisional text
            // from the provisional (not-yet-finalized) frames for live captions; final text on a VAD boundary.
            finalizedFrames = newFinal
        }
    }

    private companion object {
        // Matches the compiled v2 tiny-streaming vmfbs + MoonshineV2Config.
        const val DIM = 320
        const val CHUNK = 64        // encoder/adapter graph frame count (ENC_FRAMES in compile-moonshine-v2.sh)
        const val WINDOW = 16       // sliding-window left context (the paper's "16")
        const val LOOKAHEAD = 4     // bounded right context on the edge layers (the "(16,4)" layers)
        const val HOP = CHUNK - WINDOW - LOOKAHEAD  // 44 finalized frames per chunk
        const val ENTRY_FN = "main" // both vmfbs export @main (renamed in the compile script)
    }
}
