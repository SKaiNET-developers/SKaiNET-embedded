package voicecc.asr

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readByteArray
import platform.posix.getenv
import sk.ainet.apps.llm.tokenizer.GGUFTokenizer

@OptIn(ExperimentalForeignApi::class)
private fun envOrNull(name: String): String? = getenv(name)?.toKString()

/**
 * Moonshine-tiny speech-to-text on the SL2610, entirely without Python and with a
 * fully self-compiled decoder (no vendor Moonshine decoder binaries):
 *
 *   wav → preprocessor.vmfb (CPU) → f32 features → cast bf16 → encoder.vmfb (NPU)
 *       → OUR re-decode decoder.vmfb (CPU): greedy loop, one static graph
 *       → argmax → tokenizer.json decode.
 *
 * The decoder is our DSL-authored `moonshineDecoder()` compiled as a single
 * fixed-max-sequence graph (`inputs_embeds [1,SEQ,dim] + memory → logits [1,SEQ,vocab]`).
 * Each step feeds the token prefix padded to SEQ and reads the last real position's
 * logits; the decoder's causal self-attention masks the padded future tokens, so one
 * static vmfb decodes the whole transcript — no KV-cache threading, position scalar,
 * or with_past graph. (KV-cached `decoder_with_past` is a later latency optimization.)
 *
 * Shapes/dtypes:
 *   preproc  [1,80000]f32 → [1,288,207]f32
 *   encoder  [1,288,207]bf16 → [1,207,288]bf16
 *   decoder  ([1,SEQ,288]bf16, [1,207,288]bf16) → [1,SEQ,32768]f32
 *
 * The token embeddings are OURS (tied to the decoder's lm_head; they differ from the
 * vendor's export), bundled as a bf16 npy — set MOONSHINE_EMBED to override.
 */
internal class MoonshineDecoder(
    modelDir: String = "/home/root/sl2610-examples/models/Synaptics/moonshine-tiny-bf16-torq",
    // Encoder vmfb. Defaults to the vendor prebuilt; point at OUR self-compiled encoder
    // (./gradlew moonshineEncoderMlir -> iree-compile-torq-docker.sh) by setting
    // MOONSHINE_ENCODER_VMFB, e.g. /home/root/moon/encoder-selfcompiled.vmfb.
    encoderVmfb: String? = envOrNull("MOONSHINE_ENCODER_VMFB"),
    private val preprocVmfb: String = "/home/root/moon/preprocessor_cpu.vmfb",
    private val work: String = "/home/root/moon/rt",
    torqBin: String = "/home/root/sl2610-voice-cc/.venv/lib/python3.12/site-packages/torq/_runtime_libs/torq-run-module",
    torqLibs: String = "/home/root/sl2610-voice-cc/.venv/lib/python3.12/site-packages/torq/_runtime_libs:" +
        "/home/root/sl2610-voice-cc/.venv/lib/python3.12/site-packages/iree/_runtime_libs",
) {
    // Self-compiled encoder if provided (env/ctor), else the vendor prebuilt.
    private val encVmfb = encoderVmfb ?: "$modelDir/encoder.vmfb"
    // Encoder device: vendor NPU vmfb runs on "torq"; OUR llvm-cpu encoder vmfb needs "local-task".
    private val encDevice = envOrNull("MOONSHINE_ENCODER_DEVICE") ?: "torq"
    // OUR self-compiled re-decode decoder (single graph). Env override; default the deployed vmfb.
    private val decVmfb = envOrNull("MOONSHINE_DECODER_VMFB") ?: "/home/root/moon/decoder_redecode_cpu.vmfb"
    private val decDevice = envOrNull("MOONSHINE_DECODER_DEVICE") ?: "local-task"
    private val torq = TorqRunModule(torqBin, torqLibs)
    // OUR token embeddings (tied to the decoder lm_head), bundled bf16 npy.
    private val emb = Bf16EmbeddingTable(
        envOrNull("MOONSHINE_EMBED") ?: "/home/root/moon/our_embed_tokens.npy", DIM,
    )
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

        // 3) encoder (vendor prebuilt on the NPU `torq`, or OUR llvm-cpu vmfb on `local-task`)
        val encRaw = "$work/enc_out.bin"
        if (!torq.run(encVmfb, "main", encDevice,
                listOf(TorqRunModule.Spec("1x288x207", "bf16", "$work/enc_in.bin")), listOf(encRaw))
        ) return null
        // The decoder wants bf16 memory. The vendor NPU encoder already emits bf16; OUR llvm-cpu
        // encoder emits f32 (its final LayerNorm edge) — widen to bf16 by output size.
        val encBytes = Bin.readBytes(encRaw)
        val encOut = if (encBytes.size == 207 * DIM * 4) {
            Bin.writeBytes("$work/enc_mem.bin", Bin.f32ToBf16(encBytes)); "$work/enc_mem.bin"
        } else {
            encRaw
        }

        // 4) autoregressive re-decode: one static graph, causal self-attn masks padded future tokens.
        val logits = "$work/logits.bin"
        val ids = arrayListOf(START)
        while (ids.size < SEQ) {
            Bin.writeBytes("$work/dec_in.bin", paddedEmbeds(ids))
            if (!torq.run(decVmfb, "main", decDevice,
                    listOf(
                        TorqRunModule.Spec("1x${SEQ}x$DIM", "bf16", "$work/dec_in.bin"),
                        TorqRunModule.Spec("1x207x$DIM", "bf16", encOut),
                    ), listOf(logits))
            ) return null
            val next = argmaxF32At(Bin.readBytes(logits), pos = ids.size - 1, vocab = VOCAB)
            if (next == END) break
            ids.add(next)
        }

        // 5) detokenize: drop START
        val out = ids.drop(1)
        if (out.isEmpty()) return ""
        return tokenizer.decode(out.toIntArray()).trim()
    }

    /** `[1, SEQ, DIM]` bf16 embeds: rows `0..ids.size-1` = emb(ids[i]), the rest zero. */
    private fun paddedEmbeds(ids: List<Int>): ByteArray {
        val rowBytes = DIM * 2
        val buf = ByteArray(SEQ * rowBytes)
        for (i in ids.indices) emb.row(ids[i]).copyInto(buf, destinationOffset = i * rowBytes)
        return buf
    }

    /** Argmax over `vocab` at sequence position [pos] in a `[SEQ, vocab]` little-endian f32 buffer. */
    private fun argmaxF32At(b: ByteArray, pos: Int, vocab: Int): Int {
        var best = 0
        var bestV = Float.NEGATIVE_INFINITY
        val base = pos * vocab * 4
        for (v in 0 until vocab) {
            val o = base + v * 4
            val bits = (b[o].toInt() and 0xFF) or ((b[o + 1].toInt() and 0xFF) shl 8) or
                ((b[o + 2].toInt() and 0xFF) shl 16) or ((b[o + 3].toInt() and 0xFF) shl 24)
            val f = Float.fromBits(bits)
            if (f > bestV) { bestV = f; best = v }
        }
        return best
    }

    private companion object {
        const val DIM = 288
        const val VOCAB = 32768
        const val INPUT_LEN = 80000    // preprocessor fixed input (5 s @ 16 kHz)
        const val SEQ = 32             // re-decode fixed max sequence (matches the compiled vmfb)
        const val START = 1
        const val END = 2
    }
}
