package voicecc.asr

/**
 * Board-side Moonshine ASR on the Torq NPU — now fully Python-free. Delegates to
 * [MoonshineDecoder], which drives the prebuilt Synaptics vmfbs (preprocessor on
 * CPU, encoder/decoder on the NPU) through the native `torq-run-module` binary
 * with raw-binary tensor I/O and decodes with the SKaiNET tokenizer.
 *
 * Replaces the previous `popen` of `scripts/moonshine_npu.py` (venv + onnxruntime
 * + torq.runtime). No Python interpreter is involved; `torq-run-module` and its
 * `.so`s are native ELF artifacts that ship inside the vendor wheel.
 */
public class MoonshineRunner {
    private val decoder = MoonshineDecoder()

    /** Transcribe a wav on the NPU; returns the text (or null on failure). */
    public fun transcribe(wavPath: String): String? =
        runCatching { decoder.transcribe(wavPath) }
            .onFailure { println("[asr] ${it.message}") }
            .getOrNull()
}
