package sk.ainet.examples.smarthome.tui.appcheck

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readByteArray
import kotlinx.io.readString
import sk.ainet.examples.smarthome.AppEnvironment
import sk.ainet.examples.smarthome.AppViewModel
import sk.ainet.examples.smarthome.placementOf
import sk.ainet.examples.smarthome.cartridges.CartridgeStore
import sk.ainet.examples.smarthome.cartridges.LinuxEngineFactory
import sk.ainet.examples.smarthome.cartridges.PackDevices
import sk.ainet.examples.smarthome.engine.AudioSource
import sk.ainet.examples.smarthome.engine.EngineStatus
import sk.ainet.examples.smarthome.engine.NluOutcome
import sk.ainet.examples.smarthome.fakes.ReplayAudioSource
import sk.ainet.examples.smarthome.pipeline.Endpointer
import sk.ainet.examples.smarthome.tui.BOARD_NLU_BUDGET_MS
import kotlin.system.exitProcess
import kotlin.time.TimeSource

/**
 * Board check of the whole app without the terminal UI: the TUI's own AppViewModel over the installed cartridges
 * (CartridgeStore + LinuxEngineFactory, devices from each pack), the real engines loaded the way the `l` key loads
 * them, then push-to-talk with each WAV replayed as the microphone (endpointer included) and typed commands.
 * Usage: hes-app-check [--home <dir>] [--server <url>] [--asr <id>] [--nlu <id>] [--switch-asr <id>] [--verify] [file.wav…] [-- typed command…]
 * With `--server`, a chosen variant that is not installed is downloaded first, through the Cartridges screen's path.
 * `--switch-asr` runs the WAVs once more after switching the ASR variant the way the Cartridges screen does (the NLU
 * stays loaded).
 */
fun main(args: Array<String>) {
    fun opt(name: String): String? = args.indexOf(name).takeIf { it >= 0 }?.let { args[it + 1] }
    val home = opt("--home") ?: "/home/root/.homeevensmarter"
    val switchAsr = opt("--switch-asr")
    val verify = "--verify" in args
    val dash = args.indexOf("--")
    val wavs = (if (dash >= 0) args.take(dash) else args.toList()).filter { it.endsWith(".wav") }
    val typed = if (dash >= 0) args.drop(dash + 1) else emptyList()
    LinuxEngineFactory.cacheDir = Path(home, "cache")

    val mic = SwitchableReplay()
    val store = CartridgeStore(Path(home, "cartridges"))
    val settings = AppEnvironment.MemorySettings(asrCartridge = opt("--asr") ?: "", nluCartridge = opt("--nlu") ?: "")
    val env = AppEnvironment("linux", store, LinuxEngineFactory, mic, settings, startWithFakes = true, endpointer = Endpointer(), nluBudgetMs = BOARD_NLU_BUDGET_MS)
    val clock = TimeSource.Monotonic.markNow()
    runBlocking {
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        val vm = AppViewModel(env, scope)
        /** Installs [id] from the server index when it is missing; [choose] also makes it the task's variant (as enter does). */
        suspend fun ensure(id: String, choose: Boolean) {
            if (store.byId(id) != null) return
            val variant = vm.cartridges.value.variants().firstOrNull { it.id == id }
                ?: run { println("$id: neither installed nor on the server"); exitProcess(2) }
            val t = clock.elapsedNow()
            if (choose) vm.use(variant) else vm.download(variant.entry ?: return)
            val ev = withTimeout(30 * 60_000) {
                vm.cartridges.first { c -> c.downloads[id].let { it is sk.ainet.examples.smarthome.cartridges.DownloadEvent.Done || it is sk.ainet.examples.smarthome.cartridges.DownloadEvent.Failed } }
            }.downloads.getValue(id)
            println("download $id: $ev in ${(clock.elapsedNow() - t).inWholeMilliseconds} ms")
            if (ev !is sk.ainet.examples.smarthome.cartridges.DownloadEvent.Done) exitProcess(1)
            withTimeout(10_000) { vm.cartridges.first { c -> c.installed.any { it.id == id } } }
        }
        opt("--server")?.let { url ->
            vm.setServerUrl(url)
            vm.fetchIndex()
            val c = withTimeout(60_000) { vm.cartridges.first { it.index != null || it.indexError != null } }
            println("server $url: ${c.index?.cartridges?.joinToString { "${it.id} [${placementOf(it.devices)}]" } ?: c.indexError}")
            listOfNotNull(opt("--asr"), opt("--nlu")).forEach { ensure(it, choose = true) }
            switchAsr?.let { ensure(it, choose = false) }
        }
        val installed = store.installed()
        if (installed.isEmpty()) { println("no cartridges under $home/cartridges"); exitProcess(2) }
        for (c in installed) {
            println("installed ${c.id} [${c.task}] ${c.sizeBytes / 1_000_000} MB, devices ${PackDevices.of(c.descriptor).describe()}, complete=${store.complete(c)}")
            if (verify) {
                val t = clock.elapsedNow()
                val v = store.verify(c)
                println("  verify: ${if (v.ok) "ok, ${v.checked} artifacts" else v.problems}  ${(clock.elapsedNow() - t).inWholeMilliseconds} ms  rss-peak ${rssPeakMb()} MB")
            }
        }

        vm.loadAndWarmUp()
        val engines = withTimeout(15 * 60_000) { vm.engines.first { it.ready || it.asr is EngineStatus.Failed || it.nlu is EngineStatus.Failed } }
        println("engines: asr ${engines.asr} (${engines.asrId}), nlu ${engines.nlu} (${engines.nluId})  rss-peak ${rssPeakMb()} MB")
        if (!engines.ready) { vm.log.value.forEach { println("  log $it") }; exitProcess(1) }

        suspend fun talk() {
            for (wav in wavs) {
                mic.current = SystemFileSystem.source(Path(wav)).buffered().use { it.readByteArray() }
                val before = vm.log.value.size
                while (!vm.canTalk) delay(50)
                vm.startListening()
                val line = withTimeout(5 * 60_000) { vm.log.first { it.size > before && it.drop(before).any { l -> "[asr " in l } } }
                    .drop(before).last { "[asr " in it }
                delay(200)  // the Finished event (totalMs) can land just after the log line
                val r = vm.run.value
                println("${wav.substringAfterLast('/')}: ${line.substringAfter("s  ")}  partial-last \"${r.partial}\"" +
                    (if (r.correction.isNotEmpty()) "  ${r.correction}" else "") + "  total ${r.totalMs} ms")
            }
        }
        talk()
        if (switchAsr != null) {
            val variant = vm.cartridges.value.variants().firstOrNull { it.id == switchAsr }
                ?: run { println("--switch-asr: $switchAsr is not installed"); exitProcess(2) }
            val t = clock.elapsedNow()
            vm.use(variant)
            delay(100)
            val after = withTimeout(5 * 60_000) { vm.engines.first { (it.ready && it.asrId == switchAsr) || it.asr is EngineStatus.Failed } }
            println("switched: asr ${after.asr} (${after.asrId}), nlu ${after.nlu} (${after.nluId}) in ${(clock.elapsedNow() - t).inWholeMilliseconds} ms  rss-peak ${rssPeakMb()} MB")
            if (!after.ready) { vm.log.value.forEach { println("  log $it") }; exitProcess(1) }
            talk()
        }
        for (text in typed) {
            val before = vm.log.value.size
            while (vm.pipelineState.value != sk.ainet.examples.smarthome.pipeline.PipelineState.IDLE) delay(50)
            vm.runText(text)
            val line = withTimeout(5 * 60_000) { vm.log.first { it.size > before && it.drop(before).any { l -> "[asr " in l } } }
                .drop(before).last { "[asr " in it }
            val o = vm.run.value.outcome
            println("typed: ${line.substringAfter("s  ")}" + ((o as? NluOutcome.Call)?.let { "  args ${it.args}" } ?: ""))
        }
        println("home: ${vm.homeState.value}")
        println("rss-peak ${rssPeakMb()} MB")
        scope.cancel()
    }
    exitProcess(0)
}

/** The microphone for this check: replays whichever WAV is current, in real time. */
private class SwitchableReplay : AudioSource {
    var current: ByteArray = ByteArray(0)
    override fun frames(): Flow<FloatArray> = ReplayAudioSource(current, realTime = true).frames()
}

private fun rssPeakMb(): Long = runCatching {
    SystemFileSystem.source(Path("/proc/self/status")).buffered().use { it.readString() }
        .lineSequence().first { it.startsWith("VmHWM:") }.filter { it.isDigit() }.toLong() / 1024
}.getOrDefault(-1)
