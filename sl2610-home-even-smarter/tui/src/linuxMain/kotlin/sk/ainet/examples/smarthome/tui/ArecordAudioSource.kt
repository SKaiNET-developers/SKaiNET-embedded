package sk.ainet.examples.smarthome.tui

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.convert
import kotlinx.cinterop.usePinned
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.IO
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import platform.posix.F_OK
import platform.posix.SIGTERM
import platform.posix.X_OK
import platform.posix.access
import platform.posix.fgetc
import platform.posix.fread
import platform.posix.kill
import platform.posix.pclose
import platform.posix.popen
import sk.ainet.examples.smarthome.engine.AudioSource

/**
 * Microphone through ALSA's `arecord` (present on the board image): 16 kHz mono S16_LE read from a pipe, emitted as
 * 80 ms float frames. A stop-gap until the libasound cinterop source lands; the AudioSource contract stays the same.
 * [device] is an ALSA PCM name such as `plughw:1,0` (USB mic); `null` = ALSA default.
 */
@OptIn(ExperimentalForeignApi::class)
class ArecordAudioSource(private val device: String?, private val binary: String) : AudioSource {
    override fun frames(): Flow<FloatArray> = flow {
        // The shell prints its pid first, then execs arecord in place, so the pid is arecord's — needed to stop it:
        // pclose() alone would wait forever for a recorder that never ends.
        val dev = device?.let { " -D '$it'" } ?: ""
        val pipe = popen("sh -c 'echo \$\$; exec $binary -q -t raw -f S16_LE -r $sampleRateHz -c 1$dev 2>/dev/null'", "r")
            ?: error("cannot start $binary")
        var pid = 0
        try {
            while (true) {
                val c = fgetc(pipe)
                if (c < 0 || c == '\n'.code) break
                pid = pid * 10 + (c - '0'.code)
            }
            val bytes = ByteArray(FRAME_SAMPLES * 2)
            while (true) {
                val n = bytes.usePinned { fread(it.addressOf(0), 1.convert(), bytes.size.convert(), pipe) }.toInt()
                if (n < 2) break
                emit(FloatArray(n / 2) { i -> ((bytes[2 * i].toInt() and 0xff) or (bytes[2 * i + 1].toInt() shl 8)).toShort() / 32768f })
            }
        } finally {
            if (pid > 0) kill(pid, SIGTERM)
            pclose(pipe)
        }
    }.flowOn(Dispatchers.IO)

    companion object {
        const val FRAME_SAMPLES = 1280

        /** `null` when no `arecord` is installed (typed commands still work). */
        fun ifAvailable(device: String?): ArecordAudioSource? =
            listOf("/usr/bin/arecord", "/bin/arecord").firstOrNull { access(it, F_OK) == 0 && access(it, X_OK) == 0 }
                ?.let { ArecordAudioSource(device, it) }
    }
}
