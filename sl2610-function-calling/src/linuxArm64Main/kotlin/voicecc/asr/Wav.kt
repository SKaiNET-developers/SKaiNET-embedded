package voicecc.asr

/**
 * Minimal PCM WAV loader + 16 kHz resampler (replaces the Python
 * `load_wav_16k`). Reads the RIFF/fmt/data chunks, converts samples to mono
 * float in [-1, 1], linearly resamples to 16 kHz, and pads/truncates to the
 * preprocessor's fixed `[1, targetLen]` input.
 */
internal object Wav {
    private const val TARGET_RATE = 16000

    /** Load [path], resample to 16 kHz mono, and pad/truncate to [targetLen] samples. */
    fun loadResampledPadded(path: String, targetLen: Int): FloatArray {
        val (samples, rate) = load(path)
        val at16k = if (rate == TARGET_RATE) samples else resampleLinear(samples, rate, TARGET_RATE)
        return FloatArray(targetLen) { if (it < at16k.size) at16k[it] else 0f }
    }

    /** Returns mono float samples in [-1,1] and the source sample rate. */
    private fun load(path: String): Pair<FloatArray, Int> {
        val b = Bin.readBytes(path)
        fun u16(o: Int) = (b[o].toInt() and 0xFF) or ((b[o + 1].toInt() and 0xFF) shl 8)
        fun u32(o: Int) = (b[o].toInt() and 0xFF) or ((b[o + 1].toInt() and 0xFF) shl 8) or
            ((b[o + 2].toInt() and 0xFF) shl 16) or ((b[o + 3].toInt() and 0xFF) shl 24)
        require(b.size > 44 && b[0].toInt() == 'R'.code && b[1].toInt() == 'I'.code) { "not a RIFF/WAV file" }

        var channels = 1
        var rate = TARGET_RATE
        var bits = 16
        var dataOff = -1
        var dataLen = 0
        var p = 12 // past "RIFF<size>WAVE"
        while (p + 8 <= b.size) {
            val id = "" + b[p].toInt().toChar() + b[p + 1].toInt().toChar() +
                b[p + 2].toInt().toChar() + b[p + 3].toInt().toChar()
            val sz = u32(p + 4)
            val body = p + 8
            when (id) {
                "fmt " -> { channels = u16(body + 2); rate = u32(body + 4); bits = u16(body + 14) }
                "data" -> { dataOff = body; dataLen = sz }
            }
            if (dataOff >= 0) break
            p = body + sz + (sz and 1) // chunks are word-aligned
        }
        require(dataOff >= 0) { "no data chunk" }
        if (dataOff + dataLen > b.size) dataLen = b.size - dataOff

        val bytesPerSample = bits / 8
        val frameStride = bytesPerSample * channels
        val nFrames = dataLen / frameStride
        val out = FloatArray(nFrames)
        val norm = when (bits) { 8 -> 128f; 16 -> 32768f; 32 -> 2.147483648E9f; else -> error("unsupported bits=$bits") }
        for (i in 0 until nFrames) {
            var acc = 0f
            for (c in 0 until channels) {
                val o = dataOff + i * frameStride + c * bytesPerSample
                val s = when (bits) {
                    8 -> (b[o].toInt() and 0xFF) - 128
                    16 -> ((b[o].toInt() and 0xFF) or (b[o + 1].toInt() shl 8))
                    32 -> ((b[o].toInt() and 0xFF) or ((b[o + 1].toInt() and 0xFF) shl 8) or
                        ((b[o + 2].toInt() and 0xFF) shl 16) or (b[o + 3].toInt() shl 24))
                    else -> 0
                }
                acc += s / norm
            }
            out[i] = acc / channels
        }
        return out to rate
    }

    private fun resampleLinear(src: FloatArray, from: Int, to: Int): FloatArray {
        if (src.isEmpty()) return src
        val ratio = to.toDouble() / from
        val n = (src.size * ratio).toInt()
        val out = FloatArray(n)
        for (i in 0 until n) {
            val srcPos = i / ratio
            val i0 = srcPos.toInt()
            val frac = (srcPos - i0).toFloat()
            val a = src[i0]
            val bb = if (i0 + 1 < src.size) src[i0 + 1] else a
            out[i] = a + (bb - a) * frac
        }
        return out
    }
}
