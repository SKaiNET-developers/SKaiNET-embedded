package voicecc.asr

import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.toKString
import platform.posix.fgets
import platform.posix.pclose
import platform.posix.popen

/**
 * Board-side Moonshine ASR on the Torq NPU (option 1: subprocess the
 * version-matched torq.runtime). The prebuilt Synaptics vmfbs run on the NPU
 * via the sample venv's torq.runtime (the system iree-run-module's torq HAL is
 * a mismatched newer version and rejects them; recompiling crashes the torq
 * compiler — so we reuse the prebuilt vmfbs through the matched runtime, exactly
 * like the Python sample). scripts/moonshine_npu.py does
 * preprocessor(onnx)->encoder->decoder(+with_past) on the NPU and prints
 * `TRANSCRIPT\t<text>`.
 */
@OptIn(ExperimentalForeignApi::class)
public class MoonshineRunner(
    private val venvPython: String = "/home/root/sl2610-voice-cc/.venv/bin/python",
    private val script: String = "/home/root/voicecc-kt/scripts/moonshine_npu.py",
) {
    /** Transcribe a 16k-or-any-rate wav on the NPU; returns the text (or null on failure). */
    public fun transcribe(wavPath: String): String? {
        val cmd = "${q(venvPython)} ${q(script)} ${q(wavPath)} 2>/dev/null"
        val fp = popen(cmd, "r") ?: return null
        val sb = StringBuilder()
        memScoped {
            val n = 8192
            val buf = allocArray<ByteVar>(n)
            while (fgets(buf, n, fp) != null) sb.append(buf.toKString())
        }
        pclose(fp)
        return sb.toString().lineSequence()
            .firstOrNull { it.startsWith("TRANSCRIPT\t") }
            ?.substringAfter('\t')?.trim()
    }

    private fun q(s: String) =
        if (s.all { it.isLetterOrDigit() || it in "-_=/.,:+@" }) s else "'" + s.replace("'", "'\\''") + "'"
}
