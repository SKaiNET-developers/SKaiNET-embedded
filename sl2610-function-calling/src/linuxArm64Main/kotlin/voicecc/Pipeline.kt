package voicecc

import voicecc.actions.Intent
import voicecc.actions.defaultRouter
import voicecc.asr.MoonshineRunner
import sk.ainet.transformers.gemma.iree.GemmaDecoder
import voicecc.vad.VadSegmenter

/**
 * End-to-end on the board: wav -> Moonshine ASR (Torq NPU, prebuilt vmfb via
 * matched torq.runtime) -> FunctionGemma greedy decode (OUR DSL->StableHLO->IREE
 * f16 vmfb, CPU) -> <tool_N>(args) -> ActionRouter. The heavy lifting lives in
 * the extracted gemma-iree runtime ([GemmaDecoder] + codec + IreeRuntime) from
 * SKaiNET-transformers; this just wires ASR -> LLM -> actions and maps the
 * runtime's ToolCall onto the app's [Intent].
 */
public fun runPipeline(
    wav: String,
    ireeDir: String = "/home/root/ireetest",
    gguf: String = "/home/root/sl2610-examples/models/functiongemma-physical-ai-v10-Q5_K_M.gguf",
) {
    // 1) ASR on the NPU
    val text = MoonshineRunner().transcribe(wav)
    if (text.isNullOrBlank()) { println("[pipeline] ASR failed"); return }
    println("[1/4 asr]   \"$text\"")

    // 2+3) FunctionGemma greedy decode (OUR vmfb, CPU) via the reusable :llm module
    val decoder = GemmaDecoder(
        vmfb = "$ireeDir/gemma-gen.vmfb",
        irpa = "$ireeDir/gemma-gen.irpa",
        gguf = gguf,
    )
    val g = decoder.generate(text)
    println("[2/4 llm]   ${g.toolCallText}")

    // 4) map tool calls -> intents -> dispatch
    val intents = g.calls.map { Intent(it.tool, it.args) }
    println("[3/4 codec] $intents")
    val router = defaultRouter()
    for (res in router.dispatchAll(intents)) println("[4/4 act]   $res")
}

/**
 * Live mic / wav -> Silero VAD -> per-utterance full pipeline, via the reusable
 * :vad module ([VadSegmenter]).
 *
 * On the 1.9GB board the ~250MB resident Silero VAD + the 900MB+ LLM gen don't
 * comfortably coexist, so for `mic` we capture ONE utterance (the VAD then exits,
 * freeing its RAM + the mic), run the pipeline with the board to itself, then
 * re-listen — looping until Ctrl-C. For a wav source we stream all segments in a
 * single pass (the replay helper is light and exits on its own).
 *   source = "mic" (live) or a wav path (replay through the VAD, for testing)
 */
public fun runListen(
    source: String = "mic",
    device: String? = null,
) {
    val vad = VadSegmenter()
    val onSeg: (String) -> Unit = { wav ->
        println("\n[listen] utterance -> $wav")
        // A long-running listener must never die on one bad utterance.
        try { runPipeline(wav) } catch (e: Throwable) { println("[listen] pipeline error: ${e.message}") }
    }
    if (source == "mic") {
        println("[listen] mic — VAD(one utterance) -> ASR(NPU) -> LLM(our vmfb) -> action; Ctrl-C to stop")
        while (true) {
            var got = false
            vad.segments(source, device, once = true, onLog = { println("[listen] $it") }) { wav ->
                got = true; onSeg(wav)
            }
            // No utterance before the helper exited => mic unavailable / EOF; stop
            // rather than spin-spawning the helper.
            if (!got) { println("[listen] capture ended (no utterance) — stopping"); break }
        }
    } else {
        println("[listen] source=$source — VAD segments -> ASR(NPU) -> LLM(our vmfb) -> action")
        vad.segments(source, device, onLog = { println("[listen] $it") }, onSegment = onSeg)
        println("[listen] done")
    }
}
