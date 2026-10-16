package sk.ainet.examples.smarthome.tui.nlucheck

import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readString
import sk.ainet.examples.smarthome.cartridges.LinuxEngineFactory
import sk.ainet.examples.smarthome.engine.NluOutcome
import sk.ainet.examples.smarthome.pipeline.GoldenSet
import kotlin.system.exitProcess
import kotlin.time.measureTimedValue

/**
 * Board check for the real NLU: loads a FunctionGemma pack_dir into the in-process runtime, warms it up and runs
 * the golden set (or the given sentences) through the app's NluEngine contract, with timings and peak memory.
 * Usage: hes-nlu-check <pack-dir> [sentence…]
 */
fun main(args: Array<String>) {
    if (args.isEmpty()) { println("usage: hes-nlu-check <pack-dir> [sentence…]"); exitProcess(2) }
    LinuxEngineFactory.cacheDir = Path("/home/root/.homeevensmarter/cache")
    val engine = LinuxEngineFactory.functionGemma(args[0], log = { println("  [engine] $it") })
    val (_, warm) = measureTimedValue { engine.warmUp() }
    println("warm-up ${warm.inWholeMilliseconds} ms ${engine.warmUpTiming}  prefix ${engine.prefixTokens} tokens  rss-peak ${rssPeakMb()} MB")

    val cases = if (args.size > 1) args.drop(1).map { it to null } else GoldenSet.cases.map { it.utterance to it.expectedTool }
    var passed = 0
    val ms = mutableListOf<Long>()
    for ((text, expected) in cases) {
        val (o, t) = measureTimedValue { engine.resolve(text, budgetMs = 60_000) }
        ms += t.inWholeMilliseconds
        val got = (o as? NluOutcome.Call)?.name
        val ok = args.size > 1 || got == expected
        if (ok) passed++
        val what = when (o) {
            is NluOutcome.Call -> "${o.name}(${o.args.entries.joinToString { "${it.key}=${it.value}" }})"
            is NluOutcome.NoCall -> "no call: \"${o.text.take(50)}\""
            is NluOutcome.Failed -> "FAILED: ${o.reason}"
        }
        val timing = when (o) {
            is NluOutcome.Call -> o.timing
            is NluOutcome.NoCall -> o.timing
            is NluOutcome.Failed -> o.timing
        }
        val perTok = timing?.let { if (it.decodeTokens > 0) it.decodeMs / it.decodeTokens else 0 } ?: 0
        println("${if (ok) "OK  " else "MISS"} ${t.inWholeMilliseconds.toString().padStart(6)} ms  \"$text\" → $what" +
            (timing?.let { "  [tok ${it.tokenizeMs} restore ${it.restoreMs} chunk ${it.chunkMs} decode ${it.decodeMs}/${it.decodeTokens}t = $perTok ms/t]" } ?: "") +
            (if (expected != null && !ok) "  (want $expected)" else ""))
    }
    val sorted = ms.sorted()
    println("golden: $passed/${cases.size}  p50 ${sorted[sorted.size / 2]} ms  max ${sorted.last()} ms  rss-peak ${rssPeakMb()} MB")
    engine.close()
}

private fun rssPeakMb(): Long = runCatching {
    SystemFileSystem.source(Path("/proc/self/status")).buffered().use { it.readString() }
        .lineSequence().first { it.startsWith("VmHWM:") }.filter { it.isDigit() }.toLong() / 1024
}.getOrDefault(-1)
