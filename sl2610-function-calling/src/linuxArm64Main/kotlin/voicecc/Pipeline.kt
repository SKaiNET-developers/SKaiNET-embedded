package voicecc

import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.toKString
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import platform.posix.fgets
import platform.posix.pclose
import platform.posix.popen
import sk.ainet.apps.llm.tokenizer.GGUFTokenizer
import voicecc.actions.defaultRouter
import voicecc.asr.MoonshineRunner
import voicecc.llm.CompactCodec
import voicecc.runtime.IreeRuntime

/**
 * End-to-end on the board: wav -> Moonshine ASR (Torq NPU, prebuilt vmfb via
 * matched torq.runtime) -> Octopus-v2 prompt -> FunctionGemma greedy decode
 * (OUR DSL->StableHLO->IREE f16 vmfb, CPU) -> <tool_N>(args) -> CompactCodec ->
 * ActionRouter. The two heavy models: ASR on the NPU, LLM as our own vmfb.
 */
@OptIn(kotlin.native.runtime.NativeRuntimeApi::class)
public fun runPipeline(
    wav: String,
    ireeDir: String = "/home/root/ireetest",
    gguf: String = "/home/root/sl2610-examples/models/functiongemma-physical-ai-v10-Q5_K_M.gguf",
) {
    val seq = 24
    // 1) ASR on the NPU
    val text = MoonshineRunner().transcribe(wav)
    if (text.isNullOrBlank()) { println("[pipeline] ASR failed"); return }
    println("[1/4 asr]   \"$text\"")

    // 2) tokenize the Octopus-v2 prompt (Gemma tokenizer from the gguf). Hold the
    // tokenizer ONLY to encode, then free it — the vocab (262153) resident
    // alongside the 905MB gen subprocess OOMs the 1.9GB board. Reload it after
    // gen just to detokenize.
    val prompt = "<start_of_turn>user\n$text<end_of_turn>\n<start_of_turn>model\n"
    var tok: GGUFTokenizer? = GGUFTokenizer.fromSource(SystemFileSystem.source(Path(gguf)).buffered())
    val eot = tok!!.encode("<end_of_turn>").single()
    val eos = tok!!.eosTokenId
    val ptoks = listOf(tok!!.bosTokenId) + tok!!.encode(prompt).toList()
    tok = null
    kotlin.native.runtime.GC.collect()
    if (ptoks.size > seq - 4) { println("[pipeline] prompt too long (${ptoks.size})"); return }

    // 3) greedy decode with OUR FunctionGemma vmfb (CPU).
    platform.posix.system("sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null")
    val rt = IreeRuntime()
    val buf = IntArray(seq) { if (it < ptoks.size) ptoks[it] else 0 }
    val gen = mutableListOf<Int>()
    var step = 0
    while (ptoks.size + step < seq) {
        val r = rt.invoke("$ireeDir/gemma-gen.vmfb", "gemma",
            listOf("1x24xi32=" + buf.joinToString(",")),
            mapOf("model" to "$ireeDir/gemma-gen.irpa"), "file")
        val arr = IreeRuntime.parseIntResult(r.stdout) ?: break
        val next = arr[ptoks.size - 1 + step]
        if (next == eos) break
        gen.add(next)
        if (next == eot) break
        buf[ptoks.size + step] = next
        step++
    }
    // reload tokenizer (gen subprocess has exited, RAM freed) to detokenize
    val tok2 = GGUFTokenizer.fromSource(SystemFileSystem.source(Path(gguf)).buffered())
    val toolCall = tok2.decode(gen.toIntArray())
    println("[2/4 llm]   $toolCall")

    // 4) decode tool call -> dispatch
    val intents = CompactCodec.parse(toolCall)
    println("[3/4 codec] $intents")
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
