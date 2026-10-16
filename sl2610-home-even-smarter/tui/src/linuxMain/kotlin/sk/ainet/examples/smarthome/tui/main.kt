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

private const val USAGE = """hes-tui — HomeEvenSmarter in the terminal

  --mic <pcm>      ALSA capture device for arecord, e.g. plughw:1,0 (default: ALSA default)
  --no-mic         typed commands only
  --wav <file>     replay a 16 kHz 16-bit mono WAV instead of the microphone (each talk replays it)
  --server <url>   cartridge server (persisted in settings.properties)
  --home <dir>     settings and cartridges (default: ~/.homeevensmarter)
"""

private class Options(val mic: String?, val noMic: Boolean, val wav: String?, val server: String?, val home: String?) {
    companion object {
        fun parse(args: Array<String>): Options {
            val values = mutableMapOf<String, String>()
            val flags = mutableSetOf<String>()
            var i = 0
            while (i < args.size) {
                when (val a = args[i]) {
                    "--no-mic" -> flags += a
                    "--mic", "--wav", "--server", "--home" -> { values[a] = args.getOrNull(i + 1) ?: usage("$a needs a value"); i++ }
                    "-h", "--help" -> usage(null)
                    else -> usage("unknown option $a")
                }
                i++
            }
            return Options(values["--mic"], "--no-mic" in flags, values["--wav"], values["--server"], values["--home"])
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
    )

    runBlocking {
        val vmScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        val vm = AppViewModel(env, vmScope)
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
