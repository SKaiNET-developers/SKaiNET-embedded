package voicecc.asr

import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readByteArray
import sk.ainet.apps.llm.tokenizer.GGUFTokenizer

/**
 * Moonshine-tiny speech-to-text on the SL2610, entirely without Python. Mirrors
 * the vendor `utils/moonshine/runner.py` encoder-decoder loop, but drives the
 * prebuilt Synaptics vmfbs through the native [TorqRunModule] subprocess with
 * raw-binary tensor files:
 *
 *   wav → preprocessor.vmfb (CPU) → f32 features → cast bf16 → encoder.vmfb (NPU)
 *       → decoder.vmfb (first step: logits + self/cross KV cache)
 *       → decoder_with_past.vmfb (autoregressive, threading the self KV cache)
 *       → argmax → tokenizer.json decode.
 *
 * The preprocessor is an ONNX graph compiled to an aarch64 llvm-cpu vmfb with the
 * Torq-fork iree-compile (stock IREE 3.11 emits a "Ch" bytecode feature the board
 * runtime rejects). Shapes/dtypes are fixed by the model reflection:
 *   preproc  [1,80000]f32 → [1,288,207]f32
 *   encoder  [1,288,207]bf16 → [1,207,288]bf16
 *   decoder  ([1,1,288]bf16, [1,207,288]bf16) → [1,1,32768]bf16 + 6×(self_k,self_v,cross_k,cross_v)
 *   dec_past ([1,1,288]bf16, [1,1]i32, 6×4 cache) → [1,1,32768]bf16 + 6×2 self cache
 */
internal class MoonshineDecoder(
    modelDir: String = "/home/root/sl2610-examples/models/Synaptics/moonshine-tiny-bf16-torq",
    private val preprocVmfb: String = "/home/root/moon/preprocessor_cpu.vmfb",
    private val work: String = "/home/root/moon/rt",
    torqBin: String = "/home/root/sl2610-voice-cc/.venv/lib/python3.12/site-packages/torq/_runtime_libs/torq-run-module",
    torqLibs: String = "/home/root/sl2610-voice-cc/.venv/lib/python3.12/site-packages/torq/_runtime_libs:" +
        "/home/root/sl2610-voice-cc/.venv/lib/python3.12/site-packages/iree/_runtime_libs",
) {
    private val encVmfb = "$modelDir/encoder.vmfb"
    private val decVmfb = "$modelDir/decoder.vmfb"
    private val decPastVmfb = "$modelDir/decoder_with_past.vmfb"
    private val torq = TorqRunModule(torqBin, torqLibs)
    private val emb = Bf16EmbeddingTable("$modelDir/decoder_token_embeddings.npy", DIM)
    private val tokenizer: GGUFTokenizer = GGUFTokenizer.fromTokenizerJson(
        SystemFileSystem.source(Path("$modelDir/tokenizer.json")).buffered().use { it.readByteArray() }.decodeToString(),
    )

    fun transcribe(wav: String): String? {
        SystemFileSystem.createDirectories(Path(work))

        // 1) wav → 16k mono, pad/truncate to 80000 → raw f32
        val audio = Wav.loadResampledPadded(wav, INPUT_LEN)
        Bin.writeBytes("$work/wav.bin", Bin.f32Bytes(audio))

        // 2) preprocessor on CPU → f32 features → cast to bf16 for the encoder
        val feat = "$work/feat.bin"
        if (!torq.run(preprocVmfb, "main", "local-task",
                listOf(TorqRunModule.Spec("1x$INPUT_LEN", "f32", "$work/wav.bin")), listOf(feat))
        ) return null
        Bin.writeBytes("$work/enc_in.bin", Bin.f32ToBf16(Bin.readBytes(feat)))

        // 3) encoder on the NPU
        val encOut = "$work/enc_out.bin"
        if (!torq.run(encVmfb, "main", "torq",
                listOf(TorqRunModule.Spec("1x288x207", "bf16", "$work/enc_in.bin")), listOf(encOut))
        ) return null

        // 4) first decoder step: START token embedding + encoder output → logits + full cache
        Bin.writeBytes("$work/tok.bin", emb.row(START))
        val logits = "$work/logits.bin"
        // decoder outputs (in order): logits, then per layer [self_k, self_v, cross_k, cross_v]
        val crossFile = Array(2 * N_LAYERS) { "$work/cross_$it.bin" }
        val decSelf0 = Array(2 * N_LAYERS) { "$work/dself_$it.bin" }
        val decOutputs = ArrayList<String>(1 + 4 * N_LAYERS).apply {
            add(logits)
            for (l in 0 until N_LAYERS) {
                add(decSelf0[2 * l]); add(decSelf0[2 * l + 1])
                add(crossFile[2 * l]); add(crossFile[2 * l + 1])
            }
        }
        if (!torq.run(decVmfb, "main", "torq",
                listOf(
                    TorqRunModule.Spec("1x1x$DIM", "bf16", "$work/tok.bin"),
                    TorqRunModule.Spec("1x207x$DIM", "bf16", encOut),
                ), decOutputs)
        ) return null

        var next = Bin.argmaxBf16(Bin.readBytes(logits), VOCAB)
        val tokens = arrayListOf(START, next)

        // 5) seed the self KV cache: pad each [1,8,1,36] to [1,8,30,36] at position 0
        var selfCur = Array(2 * N_LAYERS) { "$work/selfA_$it.bin" }
        var selfNext = Array(2 * N_LAYERS) { "$work/selfB_$it.bin" }
        for (k in 0 until 2 * N_LAYERS) Bin.writeBytes(selfCur[k], padSelfToMax(Bin.readBytes(decSelf0[k])))

        // 6) autoregressive decode with decoder_with_past (max INPUT_LEN/16000*6 = 30 positions)
        val maxTokens = MAX_POS
        var i = 0
        while (i < maxTokens - 1 && next != END) {
            Bin.writeBytes("$work/tok.bin", emb.row(next))
            Bin.writeBytes("$work/seq.bin", Bin.i32Scalar(i + 1))
            val inputs = ArrayList<TorqRunModule.Spec>(2 + 4 * N_LAYERS).apply {
                add(TorqRunModule.Spec("1x1x$DIM", "bf16", "$work/tok.bin"))
                add(TorqRunModule.Spec("1x1", "i32", "$work/seq.bin"))
                for (l in 0 until N_LAYERS) {
                    add(TorqRunModule.Spec("1x8x${MAX_POS}x36", "bf16", selfCur[2 * l]))
                    add(TorqRunModule.Spec("1x8x${MAX_POS}x36", "bf16", selfCur[2 * l + 1]))
                    add(TorqRunModule.Spec("1x8x207x36", "bf16", crossFile[2 * l]))
                    add(TorqRunModule.Spec("1x8x207x36", "bf16", crossFile[2 * l + 1]))
                }
            }
            val outputs = ArrayList<String>(1 + 2 * N_LAYERS).apply {
                add(logits)
                for (k in 0 until 2 * N_LAYERS) add(selfNext[k])
            }
            if (!torq.run(decPastVmfb, "main", "torq", inputs, outputs)) return null
            val tmp = selfCur; selfCur = selfNext; selfNext = tmp // ping-pong
            next = Bin.argmaxBf16(Bin.readBytes(logits), VOCAB)
            tokens.add(next)
            i++
        }

        // 7) detokenize: drop START, stop at END
        val ids = tokens.drop(1).takeWhile { it != END }
        if (ids.isEmpty()) return ""
        return tokenizer.decode(ids.toIntArray()).trim()
    }

    /** Zero-pad a `[1,8,1,36]` bf16 self-cache to `[1,8,30,36]`, placing it at position 0. */
    private fun padSelfToMax(src: ByteArray): ByteArray {
        val headBytes = HEAD_DIM * 2                 // 36 bf16 = 72 bytes (one position, one head)
        val dstHeadStride = MAX_POS * HEAD_DIM * 2   // 30*36 bf16 = 2160 bytes per head
        val dst = ByteArray(8 * dstHeadStride)
        for (h in 0 until 8) src.copyInto(dst, destinationOffset = h * dstHeadStride,
            startIndex = h * headBytes, endIndex = h * headBytes + headBytes)
        return dst
    }

    private companion object {
        const val N_LAYERS = 6
        const val DIM = 288
        const val HEAD_DIM = 36
        const val VOCAB = 32768
        const val INPUT_LEN = 80000    // preprocessor fixed input (5 s @ 16 kHz)
        const val MAX_POS = 30         // self-cache seq dim = 80000/16000*6
        const val START = 1
        const val END = 2
    }
}
