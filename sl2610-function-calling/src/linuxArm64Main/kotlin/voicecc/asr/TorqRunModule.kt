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
 * Subprocess driver for the board's native `torq-run-module` (the Torq fork of
 * `iree-run-module`, a standalone ELF — NOT Python). Runs one compiled vmfb
 * function with raw-binary tensor files: each input is `SHAPExDTYPE=@file.bin`,
 * each output is `--output=@file.bin`. The NPU parts run on `--device=torq`; the
 * ONNX-derived preprocessor runs on `--device=local-task` (CPU) — same binary.
 *
 * This is the ASR analogue of gemma-iree's [sk.ainet.transformers.gemma.iree.IreeRuntime];
 * it exists here because torq-run-module needs the torq HAL flags + raw-bin I/O
 * (the runtime's numpy writer rejects bf16). Extracting a shared runtime is 3.5.
 */
@OptIn(ExperimentalForeignApi::class)
internal class TorqRunModule(
    private val bin: String,
    /** Colon-separated dirs for the torq + iree runtime `.so`s (native libs, not Python). */
    private val libPath: String,
) {
    data class Spec(val shape: String, val dtype: String, val file: String) {
        val inputArg get() = "--input=${shape}x${dtype}=@${file}"
    }

    /**
     * Invoke [function] in [module] on [device]. [inputs] bind raw-bin files in
     * order; [outputs] name the raw-bin files IREE writes (in output order).
     * Returns true on exit 0. stderr is captured and printed on failure.
     */
    fun run(
        module: String,
        function: String,
        device: String,
        inputs: List<Spec>,
        outputs: List<String>,
        torqHw: String? = "astra_machina",
    ): Boolean {
        val args = buildList {
            add("LD_LIBRARY_PATH=$libPath")
            add(bin)
            add("--device=$device")
            if (device == "torq" && torqHw != null) add("--torq_hw_type=$torqHw")
            add("--module=$module")
            add("--function=$function")
            for (s in inputs) add(s.inputArg)
            for (o in outputs) add("--output=@$o")
        }
        val cmd = args.joinToString(" ") { quote(it) } + " 2>&1"
        val fp = popen(cmd, "r") ?: run { println("[torq] popen failed"); return false }
        val sb = StringBuilder()
        memScoped {
            val n = 8192
            val buf = allocArray<ByteVar>(n)
            while (fgets(buf, n, fp) != null) sb.append(buf.toKString())
        }
        val status = pclose(fp)
        val code = if (status == -1) -1 else (status shr 8) and 0xff
        if (code != 0 || sb.contains("ERROR") || sb.contains("UNIMPLEMENTED") || sb.contains("INVALID_ARGUMENT")) {
            println("[torq] $function on $device failed (exit=$code):\n${sb.toString().trim().take(600)}")
            return false
        }
        return true
    }

    private fun quote(s: String): String =
        if (s.all { it.isLetterOrDigit() || it in "-_=/.,:+@" }) s
        else "'" + s.replace("'", "'\\''") + "'"
}
