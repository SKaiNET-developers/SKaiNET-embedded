package sk.ainet.examples.smarthome.tui

import com.jakewharton.mosaic.runMosaic
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readByteArray
import platform.posix.getenv
import sk.ainet.examples.smarthome.AppEnvironment
import sk.ainet.examples.smarthome.AppViewModel
import sk.ainet.examples.smarthome.cartridges.CartridgeStore
import sk.ainet.examples.smarthome.cartridges.LinuxEngineFactory
import sk.ainet.examples.smarthome.engine.AudioSource
import sk.ainet.examples.smarthome.fakes.ReplayAudioSource
import sk.ainet.examples.smarthome.pipeline.Endpointer
import kotlin.system.exitProcess

/** FunctionGemma on the board's CPU takes 10–33 s per command (LOG 2026-10-16 R3); the phone's 15 s would cut it off. */
const val BOARD_NLU_BUDGET_MS: Long = 60_000

private const val USAGE = """hes-tui — HomeEvenSmarter in the terminal

  --mic <pcm>      ALSA capture device for arecord, e.g. plughw:1,0 (default: ALSA default)
  --no-mic         typed commands only
  --wav <file>     replay a 16 kHz 16-bit mono WAV instead of the microphone (each talk replays it)
  --server <url>   cartridge server (persisted in settings.properties)
  --home <dir>     settings and cartridges (default: ~/.homeevensmarter)
  --load           load the installed cartridges into the real engines at start (as the `l` key does)
"""

private class Options(val mic: String?, val noMic: Boolean, val wav: String?, val server: String?, val home: String?, val load: Boolean) {
    companion object {
        fun parse(args: Array<String>): Options {
            val values = mutableMapOf<String, String>()
            val flags = mutableSetOf<String>()
            var i = 0
            while (i < args.size) {
                when (val a = args[i]) {
                    "--no-mic", "--load" -> flags += a
                    "--mic", "--wav", "--server", "--home" -> { values[a] = args.getOrNull(i + 1) ?: usage("$a needs a value"); i++ }
                    "-h", "--help" -> usage(null)
                    else -> usage("unknown option $a")
                }
                i++
            }
            return Options(values["--mic"], "--no-mic" in flags, values["--wav"], values["--server"], values["--home"], "--load" in flags)
        }

        private fun usage(problem: String?): Nothing {
            problem?.let { println(it) }
            println(USAGE)
            exitProcess(if (problem == null) 0 else 2)
        }
    }
}

@OptIn(ExperimentalForeignApi::class)
fun main(args: Array<String>) {
    val opts = Options.parse(args)
    val home = Path(opts.home ?: "${getenv("HOME")?.toKString() ?: "."}/.homeevensmarter")
    val cartridges = Path(home, "cartridges")
    SystemFileSystem.createDirectories(cartridges)
    val settings = FileSettings(Path(home, "settings.properties"))
    opts.server?.let { settings.serverUrl = it }
    LinuxEngineFactory.cacheDir = Path(home, "cache")  // the NLU's tokenized catalog prefix survives restarts

    val audio: AudioSource? = when {
        opts.wav != null -> ReplayAudioSource(SystemFileSystem.source(Path(opts.wav)).buffered().use { it.readByteArray() })
        opts.noMic -> null
        else -> ArecordAudioSource.ifAvailable(opts.mic)
    }
    val env = AppEnvironment(
        platform = "linux",
        store = CartridgeStore(cartridges),
        engineFactory = LinuxEngineFactory,
        audioSource = audio,
        settings = settings,
        startWithFakes = true,
        endpointer = Endpointer(),
        nluBudgetMs = BOARD_NLU_BUDGET_MS,
    )

    runBlocking {
        val vmScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        val vm = AppViewModel(env, vmScope)
        if (opts.load) vm.loadAndWarmUp()
        lateinit var ui: Job
        ui = launch { runMosaic { TuiApp(vm, micLabel = micLabel(opts, audio), onQuit = { ui.cancel() }) } }
        ui.join()
        vmScope.cancel()
    }
}

private fun micLabel(opts: Options, audio: AudioSource?): String = when {
    opts.wav != null -> "replay ${opts.wav.substringAfterLast('/')}"
    audio == null -> if (opts.noMic) "off" else "none (arecord missing)"
    else -> "arecord ${opts.mic ?: "default"}"
}
