package voicecc.asr

import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readByteArray
import kotlinx.io.write

/**
 * Raw little-endian tensor I/O for driving torq-run-module with
 * `SHAPExDTYPE=@file.bin` inputs and `--output=@file.bin` outputs. Every
 * Moonshine tensor's shape+dtype is known from the vmfb reflection, so we thread
 * tensors as headerless raw bytes — the board runtime's numpy writer does not
 * support bf16, but raw `.bin` round-trips any dtype.
 *
 * bf16 is stored as 2 LE bytes (the high 16 bits of the f32); f32 as 4 LE bytes;
 * i32 as 4 LE bytes.
 */
internal object Bin {
    fun readBytes(path: String): ByteArray =
        SystemFileSystem.source(Path(path)).buffered().use { it.readByteArray() }

    fun writeBytes(path: String, bytes: ByteArray) {
        SystemFileSystem.sink(Path(path)).buffered().use { it.write(bytes) }
    }

    // --- scalar codecs (truncating bf16, matching the vendor bf16 weights) ---
    fun bf16ToFloat(bits: Int): Float = Float.fromBits((bits and 0xFFFF) shl 16)
    fun floatToBf16(v: Float): Int = (v.toRawBits() ushr 16) and 0xFFFF

    fun getBf16(b: ByteArray, elemIdx: Int): Float {
        val o = elemIdx * 2
        val bits = (b[o].toInt() and 0xFF) or ((b[o + 1].toInt() and 0xFF) shl 8)
        return bf16ToFloat(bits)
    }

    fun getF32(b: ByteArray, elemIdx: Int): Float {
        val o = elemIdx * 4
        val bits = (b[o].toInt() and 0xFF) or
            ((b[o + 1].toInt() and 0xFF) shl 8) or
            ((b[o + 2].toInt() and 0xFF) shl 16) or
            ((b[o + 3].toInt() and 0xFF) shl 24)
        return Float.fromBits(bits)
    }

    /** Convert a raw f32 buffer to a raw bf16 buffer (halves the byte count). */
    fun f32ToBf16(f32: ByteArray): ByteArray {
        val n = f32.size / 4
        val out = ByteArray(n * 2)
        for (i in 0 until n) {
            val bits = floatToBf16(getF32(f32, i))
            out[i * 2] = (bits and 0xFF).toByte()
            out[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
        }
        return out
    }

    /** A float array as raw LE f32 bytes. */
    fun f32Bytes(a: FloatArray): ByteArray {
        val out = ByteArray(a.size * 4)
        for (i in a.indices) {
            val b = a[i].toRawBits()
            out[i * 4] = (b and 0xFF).toByte()
            out[i * 4 + 1] = ((b ushr 8) and 0xFF).toByte()
            out[i * 4 + 2] = ((b ushr 16) and 0xFF).toByte()
            out[i * 4 + 3] = ((b ushr 24) and 0xFF).toByte()
        }
        return out
    }

    /** A `[1,1]` int32 tensor as raw LE bytes. */
    fun i32Scalar(v: Int): ByteArray = byteArrayOf(
        (v and 0xFF).toByte(),
        ((v ushr 8) and 0xFF).toByte(),
        ((v ushr 16) and 0xFF).toByte(),
        ((v ushr 24) and 0xFF).toByte(),
    )

    /** Argmax over a raw bf16 logits buffer of [count] elements. */
    fun argmaxBf16(b: ByteArray, count: Int): Int {
        var best = -1
        var bestV = Float.NEGATIVE_INFINITY
        for (i in 0 until count) {
            val v = getBf16(b, i)
            if (v > bestV) { bestV = v; best = i }
        }
        return best
    }
}

/**
 * bf16 token-embedding table read from `decoder_token_embeddings.npy`
 * (`|V2` raw bf16, shape `(vocab, dim)`). Parses the npy header once to find the
 * data offset, then serves rows as raw bf16 bytes for the decoder token input.
 */
internal class Bf16EmbeddingTable(npyPath: String, private val dim: Int) {
    private val raw: ByteArray = Bin.readBytes(npyPath)
    private val dataOffset: Int = parseNpyDataOffset(raw)
    private val rowBytes = dim * 2

    /** Row [tokenId] as raw bf16 bytes, ready to write as the `[1,1,dim]` decoder input. */
    fun row(tokenId: Int): ByteArray {
        val start = dataOffset + tokenId * rowBytes
        return raw.copyOfRange(start, start + rowBytes)
    }

    private companion object {
        // npy v1: b"\x93NUMPY" \x01 \x00 <u16 headerLen> <header ascii...> ; data
        // follows immediately. v2 uses a u32 headerLen. Total header is padded to
        // a multiple of 64 (data offset). We only need the offset, not the descr.
        fun parseNpyDataOffset(b: ByteArray): Int {
            require(b.size >= 10 && b[0].toInt() and 0xFF == 0x93) { "not an npy file" }
            val major = b[6].toInt()
            return if (major >= 2) {
                val hl = (b[8].toInt() and 0xFF) or ((b[9].toInt() and 0xFF) shl 8) or
                    ((b[10].toInt() and 0xFF) shl 16) or ((b[11].toInt() and 0xFF) shl 24)
                12 + hl
            } else {
                val hl = (b[8].toInt() and 0xFF) or ((b[9].toInt() and 0xFF) shl 8)
                10 + hl
            }
        }
    }
}
