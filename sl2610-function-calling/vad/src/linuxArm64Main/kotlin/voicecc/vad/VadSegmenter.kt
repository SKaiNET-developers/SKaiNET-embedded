package voicecc.vad

import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.toKString
import platform.posix.fgets
import platform.posix.pclose
import platform.posix.popen

/**
 * Streams speech segments from an audio source via the light Silero-VAD helper
 * (scripts/vad_capture.py through the sample venv — no Moonshine, so low RAM on
 * the 1.9GB board). The helper writes each detected utterance to a wav and emits
 * `SEGMENT\t<wav>`; [segments] invokes [onSegment] with that path per utterance,
 * sequentially, until the helper signals `DONE` / EOF.
 *
 * Deliberately knows nothing about ASR/LLM — the caller decides what to do with
 * each segment — so this module stays reusable.
 */
@OptIn(ExperimentalForeignApi::class)
public class VadSegmenter(
    private val venvPython: String = "/home/root/sl2610-voice-cc/.venv/bin/python",
    private val helper: String = "/home/root/voicecc-kt/scripts/vad_capture.py",
) {
    /**
     * @param source "mic" (live capture) or a wav path (replay through the VAD, for testing)
     * @param device optional ALSA capture device (e.g. a USB mic) for live mode
     * @param once   exit the helper after the first utterance (frees the ~250MB Silero
     *               VAD + the mic). Use on RAM-tight boards so the per-segment work runs
     *               without the VAD resident; the caller re-invokes to keep listening.
     * @param onLog   diagnostics / non-segment lines from the helper
     * @param onSegment called with the wav path of each detected utterance
     */
    public fun segments(
        source: String = "mic",
        device: String? = null,
        once: Boolean = false,
        onLog: (String) -> Unit = {},
        onSegment: (String) -> Unit,
    ) {
        val dev = if (device != null) " --device ${q(device)}" else ""
        val onceArg = if (once) " --once" else ""
        val cmd = "${q(venvPython)} ${q(helper)} --source ${q(source)}$dev$onceArg 2>/dev/null"
        val fp = popen(cmd, "r") ?: run { onLog("cannot start VAD helper"); return }
        memScoped {
            val n = 4096
            val buf = allocArray<ByteVar>(n)
            while (fgets(buf, n, fp) != null) {
                val line = buf.toKString().trim()
                when {
                    line.startsWith("SEGMENT\t") -> onSegment(line.substringAfter('\t'))
                    line == "DONE" -> break
                    line.isNotEmpty() -> onLog(line)
                }
            }
        }
        pclose(fp)
    }

    private fun q(s: String) =
        if (s.all { it.isLetterOrDigit() || it in "-_=/.,:+@" }) s else "'" + s.replace("'", "'\\''") + "'"
}
