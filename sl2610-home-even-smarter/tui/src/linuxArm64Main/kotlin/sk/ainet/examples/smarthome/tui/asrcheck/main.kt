package sk.ainet.examples.smarthome.tui.asrcheck

import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readByteArray
import kotlinx.io.readString
import platform.posix.usleep
import sk.ainet.examples.smarthome.cartridges.LinuxEngineFactory
import sk.ainet.examples.smarthome.fakes.Wav
import sk.ainet.examples.smarthome.iree.MoonshineSpec
import kotlin.system.exitProcess
import kotlin.time.TimeSource
import kotlin.time.measureTimedValue

/**
 * Board check for the real ASR: loads a Moonshine pack_dir into the in-process runtime and streams 16 kHz mono WAVs
 * through the app's AsrEngine contract in 80 ms frames, paced like a microphone (or as fast as possible with
 * --fast), printing every partial with its time, the final, the latency after the last sample and peak memory.
 * Usage: hes-asr-check [--fast] [--verbose] [--lang en] <pack-dir> <file.wav>…
 */
fun main(args: Array<String>) {
    val flags = args.filter { it.startsWith("--") && it != "--lang" }.toSet()
    val lang = args.indexOf("--lang").takeIf { it >= 0 }?.let { args.getOrNull(it + 1) } ?: "en"
    val positional = args.filterIndexed { i, a -> !a.startsWith("--") && (i == 0 || args[i - 1] != "--lang") }
    if (positional.size < 2) { println("usage: hes-asr-check [--fast] [--verbose] [--lang en] <pack-dir> <file.wav>…"); exitProcess(2) }
    val realTime = "--fast" !in flags

    val (engine, load) = measureTimedValue {
        LinuxEngineFactory.moonshine(positional[0], lang, spec = MoonshineSpec(verbose = "--verbose" in flags))
    }
    val (session, open) = measureTimedValue { engine.open() }
    println("load ${load.inWholeMilliseconds} ms  open+warm-up ${open.inWholeMilliseconds} ms  ${engine.id} [${engine.language}]  rss-peak ${rssPeakMb()} MB")

    val frame = 1280
    for (file in positional.drop(1)) {
        val bytes = SystemFileSystem.source(Path(file)).buffered().use { it.readByteArray() }
        val rate = le32(bytes, 24)
        if (rate != 16_000) { println("SKIP  $file: $rate Hz, needs 16000"); continue }
        val pcm = Wav.decode16BitMono(bytes)
        val start = TimeSource.Monotonic.markNow()
        var fed = 0
        var maxFeedMs = 0L
        while (fed < pcm.size) {
            val n = minOf(frame, pcm.size - fed)
            val t0 = TimeSource.Monotonic.markNow()
            val partial = session.feed(pcm.copyOfRange(fed, fed + n))
            maxFeedMs = maxOf(maxFeedMs, t0.elapsedNow().inWholeMilliseconds)
            fed += n
            if (partial != null) println("   ${start.elapsedNow().inWholeMilliseconds.toString().padStart(6)} ms  partial \"$partial\"")
            if (realTime) {
                val due = fed * 1000L / 16_000 - start.elapsedNow().inWholeMilliseconds
                if (due > 0) usleep((due * 1000).toUInt())
            }
        }
        val audioMs = pcm.size * 1000L / 16_000
        val endOfAudio = start.elapsedNow()
        val (final, finish) = measureTimedValue { session.finish() }
        val total = start.elapsedNow().inWholeMilliseconds
        val stats = engine.cartridge.lastStats
        println("FINAL \"$final\"  ${file.substringAfterLast('/')}  audio $audioMs ms  " +
            "finish ${finish.inWholeMilliseconds} ms (flush ${stats["flushMs"]} decode ${stats["decodeMs"]})  " +
            "after-last-sample ${total - minOf(audioMs, endOfAudio.inWholeMilliseconds).coerceAtLeast(0)} ms  " +
            "rtf ${fmt(total.toDouble() / audioMs)}  max-feed $maxFeedMs ms  " +
            "windows ${stats["windows"]} hops ${stats["hops"]} dropped ${stats["droppedDecodes"]} tokens ${stats["tokens"]} errors ${stats["errors"]}")
        if ("--verbose" in flags) println("      timeline ${stats["timeline"]}")
    }
    println("rss-peak ${rssPeakMb()} MB")
    engine.close()
}

private fun le32(b: ByteArray, i: Int): Int =
    (b[i].toInt() and 0xFF) or ((b[i + 1].toInt() and 0xFF) shl 8) or ((b[i + 2].toInt() and 0xFF) shl 16) or ((b[i + 3].toInt() and 0xFF) shl 24)

private fun fmt(v: Double): String = ((v * 100).toLong() / 100.0).toString()

private fun rssPeakMb(): Long = runCatching {
    SystemFileSystem.source(Path("/proc/self/status")).buffered().use { it.readString() }
        .lineSequence().first { it.startsWith("VmHWM:") }.filter { it.isDigit() }.toLong() / 1024
}.getOrDefault(-1)
