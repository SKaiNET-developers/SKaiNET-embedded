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
 * --fast), printing every partial with its time, the final, the latency after the last sample, the real-time factor
 * (RTF = time spent inside the engine, feed() + finish(), over the audio's duration) and peak memory.
 * Usage: hes-asr-check [--fast] [--verbose] [--peek] [--partials N] [--fast-finish] [--lang en] <pack-dir> <file.wav>…
 * --peek turns the early partial back on, --partials 1 decodes a partial on every hop (the app defaults are off and 2).
 */
fun main(args: Array<String>) {
    val valued = setOf("--lang", "--partials")
    val flags = args.filter { it.startsWith("--") && it !in valued }.toSet()
    fun value(name: String) = args.indexOf(name).takeIf { it >= 0 }?.let { args.getOrNull(it + 1) }
    val lang = value("--lang") ?: "en"
    val positional = args.filterIndexed { i, a -> !a.startsWith("--") && (i == 0 || args[i - 1] !in valued) }
    if (positional.size < 2) { println("usage: hes-asr-check [--fast] [--verbose] [--lang en] <pack-dir> <file.wav>…"); exitProcess(2) }
    val realTime = "--fast" !in flags

    val (engine, load) = measureTimedValue {
        LinuxEngineFactory.moonshine(positional[0], lang, spec = MoonshineSpec(
            verbose = "--verbose" in flags, earlyPeek = "--peek" in flags, partialEvery = value("--partials")?.toInt() ?: MoonshineSpec().partialEvery,
            fastFinish = "--fast-finish" in flags,
        ))
    }
    val (session, open) = measureTimedValue { engine.open() }
    println("load ${load.inWholeMilliseconds} ms  open+warm-up ${open.inWholeMilliseconds} ms  ${engine.id} [${engine.language}]  rss-peak ${rssPeakMb()} MB")

    val frame = 1280
    val rtfs = mutableListOf<Double>()
    for (file in positional.drop(1)) {
        val bytes = SystemFileSystem.source(Path(file)).buffered().use { it.readByteArray() }
        val rate = le32(bytes, 24)
        if (rate != 16_000) { println("SKIP  $file: $rate Hz, needs 16000"); continue }
        val pcm = Wav.decode16BitMono(bytes)
        val start = TimeSource.Monotonic.markNow()
        var fed = 0
        var maxFeedMs = 0L
        var computeUs = 0L  // time inside the engine: feed() + finish(), not the pacing sleeps
        while (fed < pcm.size) {
            val n = minOf(frame, pcm.size - fed)
            val t0 = TimeSource.Monotonic.markNow()
            val partial = session.feed(pcm.copyOfRange(fed, fed + n))
            val spent = t0.elapsedNow()
            computeUs += spent.inWholeMicroseconds
            maxFeedMs = maxOf(maxFeedMs, spent.inWholeMilliseconds)
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
        computeUs += finish.inWholeMicroseconds
        rtfs += computeUs / 1000.0 / audioMs
        val total = start.elapsedNow().inWholeMilliseconds
        val stats = engine.cartridge.lastStats
        println("FINAL \"$final\"  ${file.substringAfterLast('/')}  audio $audioMs ms  " +
            "finish ${finish.inWholeMilliseconds} ms (flush ${stats["flushMs"]} decode ${stats["decodeMs"]})  " +
            "after-last-sample ${total - minOf(audioMs, endOfAudio.inWholeMilliseconds).coerceAtLeast(0)} ms  " +
            "RTF ${fmt(computeUs / 1000.0 / audioMs)} (compute ${computeUs / 1000} ms / audio)  max-feed $maxFeedMs ms  " +
            "windows ${stats["windows"]} hops ${stats["hops"]} dropped ${stats["droppedDecodes"]} tokens ${stats["tokens"]} errors ${stats["errors"]}  " +
            "decoder: ${stats["prefills"]} prefills ${stats["prefillMs"]} ms, ${stats["steps"]} steps ${stats["stepMs"]} ms")
        if ("--verbose" in flags) println("      timeline ${stats["timeline"]}")
    }
    if (rtfs.isNotEmpty()) println("RTF mean ${fmt(rtfs.average())}  min ${fmt(rtfs.min())}  max ${fmt(rtfs.max())}  over ${rtfs.size} files (${if (realTime) "paced like a microphone: partial decodes beyond the lag budget are dropped" else "--fast: every partial decode runs"})")
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
