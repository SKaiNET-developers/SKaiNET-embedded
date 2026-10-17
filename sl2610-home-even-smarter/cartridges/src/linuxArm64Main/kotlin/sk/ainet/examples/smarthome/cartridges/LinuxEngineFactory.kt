package sk.ainet.examples.smarthome.cartridges

import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.cartridge.asr.moonshine.Language
import sk.ainet.cartridge.asr.moonshine.StreamingAsrCartridge
import sk.ainet.cartridge.asr.moonshine.StreamingAsrSession
import sk.ainet.cartridge.nlu.functiongemma.FilePrefixIdCache
import sk.ainet.cartridge.nlu.functiongemma.FunctionGemmaEngine
import sk.ainet.cartridge.nlu.functiongemma.KvBackend
import sk.ainet.cartridge.nlu.functiongemma.NluResolution
import sk.ainet.cartridge.nlu.functiongemma.NluTokenizer
import sk.ainet.cartridge.nlu.functiongemma.ToolCatalog
import sk.ainet.examples.smarthome.engine.AsrEngine
import sk.ainet.examples.smarthome.engine.AsrSession
import sk.ainet.examples.smarthome.engine.NluEngine
import sk.ainet.examples.smarthome.engine.NluOutcome
import sk.ainet.examples.smarthome.iree.IreeKv
import sk.ainet.examples.smarthome.iree.IreeMoonshine
import sk.ainet.examples.smarthome.iree.MoonshineDevices
import sk.ainet.examples.smarthome.iree.MoonshineFiles
import sk.ainet.examples.smarthome.iree.MoonshineSpec
import sk.ainet.examples.smarthome.iree.IreeRuntime
import sk.ainet.examples.smarthome.iree.KvSpec
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.io.gguf.createRandomAccessSource
import sk.ainet.cartridge.asr.moonshine.PackDir as AsrPackDir
import sk.ainet.cartridge.nlu.functiongemma.PackDir as NluPackDir

/**
 * The board's engines: the blueprint modules' engine code over the in-process IREE runtime (CPU + Torq NPU in one
 * process): FunctionGemma over the KV session, Moonshine over the streaming session. Cartridges for the board are
 * materialized for the blueprints' SL26xx targets, compiled by the Torq-fork compiler that matches the runtime linked here.
 */
object LinuxEngineFactory : EngineFactory {
    override val platform: String = "linux-arm64"
    override val available: Boolean = true

    /** One IREE instance per process; devices are shared by every engine. */
    val runtime: IreeRuntime by lazy { IreeRuntime() }

    /** Where per-cartridge caches live (the tokenized catalog prefix); set by the app before loading engines. */
    var cacheDir: Path = Path("/tmp/hes-cache")

    override fun asr(cartridge: InstalledCartridge, language: String, device: DevicePreference): AsrEngine =
        moonshine(cartridge.dir.toString(), language, device)

    /** The streaming ASR straight from a pack_dir (also used by the board check tool). */
    fun moonshine(
        packDir: String, language: String = "en", device: DevicePreference = DevicePreference.AUTO,
        spec: MoonshineSpec = MoonshineSpec(),
    ): MoonshineAsrEngine {
        val pack = AsrPackDir(packDir)
        val d = PackDevices.of(packDir)
        val devices = MoonshineDevices(d.of("frontend", device), d.of("encoder", device), d.of("adapter", device), d.of("decoder", device))
        return MoonshineAsrEngine(LinuxMoonshineCartridge(pack, language, devices, spec))
    }

    override fun nlu(cartridge: InstalledCartridge, device: DevicePreference): NluEngine =
        functionGemma(cartridge.dir.toString(), device)

    /** The NLU straight from a pack_dir (also used by the board check tool). */
    fun functionGemma(packDir: String, device: DevicePreference = DevicePreference.AUTO, log: (String) -> Unit = {}): FunctionGemmaNluEngine {
        val pack = NluPackDir(packDir)
        val dev = PackDevices.of(packDir).of("decoder", device)  // the three KV graphs share one device
        val cache = Path(cacheDir, "nlu").also { SystemFileSystem.createDirectories(it) }
        val config = FunctionGemmaEngine.Config()
        val engine = FunctionGemmaEngine(
            id = pack.id,
            catalog = ToolCatalog.load(pack.toolCatalog),
            config = config,
            openTokenizer = { GgufTokenizer.open(pack.tokenizerGguf.toString()) },
            openBackend = {
                IreeKvBackend(runtime.kv(
                    dev, KvSpec.functionGemma270m(config.chunk),
                    withPast = IreeRuntime.Graph(pack.graph("with-past").toString(), pack.parameters("with-past").toString()),
                    chunk = IreeRuntime.Graph(pack.graph("prefill-with-past").toString(), pack.parameters("prefill-with-past").toString()),
                    prefill = IreeRuntime.Graph(pack.graph("prefill-at").toString(), pack.parameters("prefill-at").toString()),
                ))
            },
            prefixCache = FilePrefixIdCache(cache),
            cacheSalt = (SystemFileSystem.metadataOrNull(pack.tokenizerGguf)?.size ?: 0L).toString(16),
            log = log,
        )
        return FunctionGemmaNluEngine(engine)
    }
}

/**
 * The blueprint's [StreamingAsrCartridge] contract over the in-process streaming session ([IreeMoonshine]), the
 * board's counterpart of the blueprint's Android `MoonshineAsrCartridge`. The engine (five graphs + one warm-up pass)
 * is created on the first [open] and lives until [close].
 */
class LinuxMoonshineCartridge(
    val pack: AsrPackDir,
    languageTag: String,
    private val devices: MoonshineDevices,
    private val spec: MoonshineSpec,
) : StreamingAsrCartridge {
    override val id: String = pack.id
    override val language: String = Language.select(languageTag, pack.languages)
        ?: throw IllegalArgumentException("pack ${pack.id} holds languages ${pack.languages}, none serves '$languageTag'")

    private var stream: IreeMoonshine? = null
    private var session: Session? = null

    /** The last utterance's counters (see [IreeMoonshine.stats]); empty before the first final. */
    val lastStats: Map<String, String> get() = stream?.stats.orEmpty()

    override fun open(): StreamingAsrSession {
        session?.let { return it }
        val f = pack.model(language)
        val s = LinuxEngineFactory.runtime.moonshine(
            MoonshineFiles(
                f.frontendVmfb.toString(), f.encoderVmfb.toString(), f.adapterVmfb.toString(), f.prefillVmfb.toString(),
                f.stepVmfb.toString(), f.paramsIrpa.toString(), f.vocabBin.toString(), f.decEmbedBin.toString(),
            ),
            devices, spec,
        )
        stream = s
        return Session(s).also { session = it }
    }

    override fun close() {
        session = null
        stream?.close(); stream = null
    }

    private class Session(private val s: IreeMoonshine) : StreamingAsrSession {
        override fun feed(pcm: FloatArray): String? = s.feed(pcm)
        override fun finish(): String = s.finish()
        override fun reset() = s.reset()
        override fun close() = reset()
    }
}

/** The app's [AsrEngine] over the cartridge contract. */
class MoonshineAsrEngine(val cartridge: LinuxMoonshineCartridge) : AsrEngine {
    override val id: String get() = cartridge.id
    override val language: String get() = cartridge.language
    override fun open(): AsrSession = Session(cartridge.open())
    override fun close() = cartridge.close()

    private class Session(private val s: StreamingAsrSession) : AsrSession {
        override fun feed(pcm: FloatArray): String? = s.feed(pcm)
        override fun finish(): String = s.finish()
        override fun reset() = s.reset()
    }
}

/** [KvBackend] over the in-process KV session. */
class IreeKvBackend(private val kv: IreeKv) : KvBackend {
    private class Snap(val inner: IreeKv.Snapshot) : KvBackend.Snapshot {
        override fun close() = inner.close()
    }

    override fun prefill(tokens: IntArray, n: Int): Int = kv.prefill(tokens, n)
    override fun releasePrefill() = kv.releasePrefill()
    override fun snapshot(): KvBackend.Snapshot = Snap(kv.snapshot())
    override fun restore(snapshot: KvBackend.Snapshot) = kv.restore((snapshot as Snap).inner)
    override fun chunk(tokens: IntArray, n: Int): Int = kv.chunk(tokens, n)
    override fun step(token: Int): Int = kv.step(token)
    override fun close() = kv.close()
}

/** The GGUF tokenizer (SKaiNET-transformers), read with pread on the board. */
object GgufTokenizer {
    fun open(path: String): NluTokenizer {
        val source = createRandomAccessSource(path) ?: throw IllegalStateException("cannot open $path")
        val fields = StreamingGGUFReader.open(source).use { it.fields }
        val tokenizer = TokenizerFactory.fromGgufFields(fields)
        return object : NluTokenizer {
            override fun encode(text: String): IntArray = tokenizer.encode(text)
            override fun decode(tokens: IntArray): String = tokenizer.decode(tokens)
        }
    }
}

/** The app's [NluEngine] over the blueprint engine. */
class FunctionGemmaNluEngine(private val engine: FunctionGemmaEngine) : NluEngine {
    override val id: String get() = engine.id
    override val toolNames: Set<String> get() = engine.toolNames
    val warmUpTiming: Map<String, Long> get() = engine.warmUpTiming
    val prefixTokens: Int get() = engine.prefixTokens

    override fun warmUp() = engine.warmUp()
    override fun close() = engine.close()

    override fun resolve(transcript: String, budgetMs: Long): NluOutcome = when (val r = engine.resolve(transcript, budgetMs)) {
        is NluResolution.Call -> NluOutcome.Call(r.name, r.args, r.raw, r.timing.toOutcome())
        is NluResolution.NoCall -> NluOutcome.NoCall(r.text, r.timing.toOutcome())
        is NluResolution.Failed -> NluOutcome.Failed(r.reason, r.cause, r.timing?.toOutcome())
    }

    private fun NluResolution.Timing.toOutcome() = NluOutcome.Timing(tokenizeMs, restoreMs, chunkMs, decodeMs, decodeTokens)
}
