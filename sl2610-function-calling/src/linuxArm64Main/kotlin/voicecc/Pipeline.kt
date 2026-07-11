package voicecc

import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.toKString
import platform.posix.fgets
import platform.posix.getenv
import platform.posix.pclose
import platform.posix.popen
import sk.ainet.transformers.gemma.iree.GemmaDecoder
import sk.ainet.transformers.gemma.iree.GemmaKvDecoder
import voicecc.actions.Intent
import voicecc.actions.defaultRouter
import voicecc.asr.MoonshineRunner

/**
 * End-to-end on the board: wav -> Moonshine ASR (Torq NPU, prebuilt vmfb via
 * matched torq.runtime) -> FunctionGemma greedy decode via the reusable
 * [GemmaDecoder] (OUR DSL->StableHLO->IREE f16 vmfb, CPU) -> <tool_N>(args) ->
 * ActionRouter. The two heavy models: ASR on the NPU, LLM as our own vmfb.
 * Tokenizer memory juggling + the decode loop live upstream in GemmaDecoder
 * (sk.ainet.transformers:skainet-transformers-runtime-gemma-iree).
 */
@OptIn(ExperimentalForeignApi::class)
public fun runPipeline(
    wav: String,
    ireeDir: String = "/home/root/ireetest",
    gguf: String = "/home/root/sl2610-examples/models/functiongemma-physical-ai-v10-Q5_K_M.gguf",
) {
    // 1) ASR on the NPU
    val text = MoonshineRunner().transcribe(wav)
    if (text.isNullOrBlank()) { println("[pipeline] ASR failed"); return }
    println("[1/4 asr]   \"$text\"")

    // 2+3) Octopus-v2 prompt -> greedy decode -> parsed tool calls.
    // GEMMA_KV=1 opts into the KV-cache 2-graph decode (prefill + with_past vmfbs, perf-program Phase 2);
    // default is the shipping fixed-seq re-decode. Both return a GemmaDecoder.Generation.
    val useKv = getenv("GEMMA_KV")?.toKString()?.trim() == "1"
    val g = if (useKv) {
        GemmaKvDecoder(
            prefillVmfb = "$ireeDir/gemma-prefill.vmfb",
            withPastVmfb = "$ireeDir/gemma-with-past.vmfb",
            irpa = "$ireeDir/gemma-gen.irpa",
            gguf = gguf,
        ).generate(text)
    } else {
        GemmaDecoder(
            vmfb = "$ireeDir/gemma-gen.vmfb",
            irpa = "$ireeDir/gemma-gen.irpa",
            gguf = gguf,
        ).generate(text)
    }
    println("[2/4 llm]   ${g.toolCallText}")
    val intents = g.calls.map { Intent(it.tool, it.args) }
    println("[3/4 codec] $intents")

    // 4) dispatch
    val router = defaultRouter()
    for (res in router.dispatchAll(intents)) println("[4/4 act]   $res")
}

/**
 * Live mic -> Silero VAD -> per-utterance full pipeline. Streams `SEGMENT\t<wav>`
 * from the light capture+VAD helper (scripts/vad_capture.py via the venv) and
 * runs [runPipeline] on each segment sequentially (memory-safe on the 1.9GB
 * board: the VAD helper is light; ASR + our LLM vmfb run one at a time).
 *   source = "mic" (live) or a wav path (replay through the VAD, for testing)
 */
@OptIn(ExperimentalForeignApi::class)
public fun runListen(
    source: String = "mic",
    device: String? = null,
    venvPython: String = "/home/root/sl2610-voice-cc/.venv/bin/python",
    helper: String = "/home/root/voicecc-kt/scripts/vad_capture.py",
) {
    val dev = if (device != null) " --device $device" else ""
    val cmd = "$venvPython $helper --source $source$dev 2>/dev/null"
    val fp = popen(cmd, "r") ?: run { println("[listen] cannot start VAD helper"); return }
    println("[listen] source=$source — VAD segments -> ASR(NPU) -> LLM(our vmfb) -> action")
    memScoped {
        val n = 4096
        val buf = allocArray<ByteVar>(n)
        while (fgets(buf, n, fp) != null) {
            val line = buf.toKString().trim()
            when {
                line.startsWith("SEGMENT\t") -> {
                    val wav = line.substringAfter('\t')
                    println("\n[listen] utterance -> $wav")
                    runPipeline(wav)
                }
                line == "DONE" -> { println("[listen] done"); break }
                line.isNotEmpty() -> println("[listen] $line")
            }
        }
    }
    pclose(fp)
}
