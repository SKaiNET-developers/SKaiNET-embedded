package voicecc.asr

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString
import platform.posix.getenv

/**
 * Board-side Moonshine ASR on the Torq NPU — now fully Python-free. Delegates to
 * [MoonshineDecoder], which drives the prebuilt Synaptics vmfbs (preprocessor on
 * CPU, encoder/decoder on the NPU) through the native `torq-run-module` binary
 * with raw-binary tensor I/O and decodes with the SKaiNET tokenizer.
 *
 * Replaces the previous `popen` of `scripts/moonshine_npu.py` (venv + onnxruntime
 * + torq.runtime). No Python interpreter is involved; `torq-run-module` and its
 * `.so`s are native ELF artifacts that ship inside the vendor wheel.
 *
 * `MOONSHINE_KV=1` opts into the KV-cache 2-graph decode ([MoonshineKvDecoder], perf-program
 * Phase 6) instead of the default fixed-seq re-decode. Both expose the same `transcribe`.
 */
@OptIn(ExperimentalForeignApi::class)
public class MoonshineRunner {
    private val useKv = getenv("MOONSHINE_KV")?.toKString()?.trim() == "1"
    private val reDecoder = if (!useKv) MoonshineDecoder() else null
    private val kvDecoder = if (useKv) MoonshineKvDecoder() else null

    /** Transcribe a wav on the NPU; returns the text (or null on failure). */
    public fun transcribe(wavPath: String): String? =
        runCatching { (kvDecoder?.transcribe(wavPath)) ?: reDecoder?.transcribe(wavPath) }
            .onFailure { println("[asr] ${it.message}") }
            .getOrNull()
}
