package sk.ainet.examples.smarthome

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.consumeAsFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import sk.ainet.examples.smarthome.actions.HomeActions
import sk.ainet.examples.smarthome.actions.HomeStore
import sk.ainet.examples.smarthome.cartridge.CartridgeIndex
import sk.ainet.examples.smarthome.cartridges.CartridgeDownloader
import sk.ainet.examples.smarthome.cartridges.DevicePreference
import sk.ainet.examples.smarthome.cartridges.DownloadEvent
import sk.ainet.examples.smarthome.cartridges.InstalledCartridge
import sk.ainet.examples.smarthome.engine.AsrEngine
import sk.ainet.examples.smarthome.engine.EngineStatus
import sk.ainet.examples.smarthome.engine.NluEngine
import sk.ainet.examples.smarthome.engine.NluOutcome
import sk.ainet.examples.smarthome.fakes.FakeAsrEngine
import sk.ainet.examples.smarthome.fakes.KeywordNluEngine
import sk.ainet.examples.smarthome.home.DeviceRef
import sk.ainet.examples.smarthome.home.HomeState
import sk.ainet.examples.smarthome.pipeline.GoldenCase
import sk.ainet.examples.smarthome.pipeline.GoldenSet
import sk.ainet.examples.smarthome.pipeline.PipelineEvent
import sk.ainet.examples.smarthome.pipeline.PipelineRun
import sk.ainet.examples.smarthome.pipeline.PipelineState
import sk.ainet.examples.smarthome.pipeline.Stage
import sk.ainet.examples.smarthome.pipeline.VoicePipeline
import kotlin.time.TimeSource

/** Engines as the UI sees them. */
data class EnginesState(
    val fakes: Boolean,
    val asr: EngineStatus = EngineStatus.Absent,
    val nlu: EngineStatus = EngineStatus.Absent,
    val device: DevicePreference = DevicePreference.AUTO,
    val asrId: String? = null,
    val nluId: String? = null,
) {
    val ready: Boolean get() = asr is EngineStatus.Ready && nlu is EngineStatus.Ready
}

data class CartridgesState(
    val installed: List<InstalledCartridge> = emptyList(),
    val index: CartridgeIndex? = null,
    val indexError: String? = null,
    val downloads: Map<String, DownloadEvent> = emptyMap(),
    val verification: Map<String, String> = emptyMap(),
)

/** One stage's status for the chips. */
enum class StageStatus { PENDING, ACTIVE, DONE, FAILED, SKIPPED }

/** The current (or last) run, as the pipeline panel draws it. */
data class RunView(
    val stages: Map<Stage, StageStatus> = Stage.entries.associateWith { StageStatus.PENDING },
    val source: String = "",
    val partial: String = "",
    val transcript: String = "",
    val audioMs: Long = 0,
    val asrMs: Long = 0,
    val outcome: NluOutcome? = null,
    val nluMs: Long = 0,
    val correction: String = "",
    val actionMessage: String = "",
    val actionOk: Boolean? = null,
    val escalation: String = "",
    val totalMs: Long = 0,
    val failure: String? = null,
    val level: Float = 0f,
)

data class GoldenResult(val case: GoldenCase, val run: PipelineRun) {
    val passed: Boolean get() = run.calledTool == case.expectedTool && (case.expectedTool == null || run.action?.ok == true)
}

/**
 * The single owner of app state: engines, cartridges, the pipeline and what the UI shows. Platform entry points
 * create one per process (it survives configuration changes) and pass it into `App`.
 */
class AppViewModel(val env: AppEnvironment, private val scope: CoroutineScope) {
    val homeStore = HomeStore()
    val homeState: StateFlow<HomeState> get() = homeStore.state

    private val _engines = MutableStateFlow(EnginesState(fakes = env.startWithFakes, device = runCatching { DevicePreference.valueOf(env.settings.devicePreference) }.getOrDefault(DevicePreference.AUTO)))
    val engines: StateFlow<EnginesState> = _engines.asStateFlow()

    private val _cartridges = MutableStateFlow(CartridgesState())
    val cartridges: StateFlow<CartridgesState> = _cartridges.asStateFlow()

    private val _run = MutableStateFlow(RunView())
    val run: StateFlow<RunView> = _run.asStateFlow()

    private val _pipelineState = MutableStateFlow(PipelineState.IDLE)
    val pipelineState: StateFlow<PipelineState> = _pipelineState.asStateFlow()

    private val _recentlyChanged = MutableStateFlow<Set<DeviceRef>>(emptySet())
    val recentlyChanged: StateFlow<Set<DeviceRef>> = _recentlyChanged.asStateFlow()

    private val _log = MutableStateFlow<List<String>>(emptyList())
    val log: StateFlow<List<String>> = _log.asStateFlow()

    private val _golden = MutableStateFlow<List<GoldenResult>>(emptyList())
    val golden: StateFlow<List<GoldenResult>> = _golden.asStateFlow()

    private val _serverUrl = MutableStateFlow(env.settings.serverUrl)
    val serverUrl: StateFlow<String> = _serverUrl.asStateFlow()

    private var asr: AsrEngine? = null
    private var nlu: NluEngine? = null
    private var pipeline: VoicePipeline? = null
    private var pipelineJob: Job? = null
    private var frames: Channel<FloatArray>? = null
    private var captureJob: Job? = null
    private var utteranceJob: Job? = null
    private var fakeScriptIndex = 0
    private val downloader = CartridgeDownloader(env.store)
    private val started = TimeSource.Monotonic.markNow()

    init {
        refreshCartridges()
        if (env.startWithFakes) useFakes(true)
    }

    // ---- engines -------------------------------------------------------------------------------------------

    fun useFakes(on: Boolean) {
        closeEngines()
        _engines.update { it.copy(fakes = on, asr = EngineStatus.Absent, nlu = EngineStatus.Absent, asrId = null, nluId = null) }
        if (on) {
            asr = FakeAsrEngine({ GoldenSet.cases[fakeScriptIndex++ % GoldenSet.cases.size].utterance })
            nlu = KeywordNluEngine(simulatedLatencyMs = 400)
            _engines.update { it.copy(asr = EngineStatus.Ready(0, "scripted partials"), nlu = EngineStatus.Ready(0, "keyword rules"), asrId = asr!!.id, nluId = nlu!!.id) }
            rebuildPipeline()
            log("engines: fakes")
        }
    }

    fun setDevice(pref: DevicePreference) {
        env.settings.devicePreference = pref.name
        _engines.update { it.copy(device = pref) }
    }

    /** Loads the installed cartridges into real engines and warms them up. Long: tens of seconds for the NLU. */
    fun loadAndWarmUp() {
        if (!env.engineFactory.available) { log("no runtime on ${env.platform}; staying on fakes"); return }
        val asrPack = env.store.asr(); val nluPack = env.store.nlu()
        if (asrPack == null || nluPack == null) { log("cartridges missing: asr=${asrPack?.id} nlu=${nluPack?.id}"); return }
        closeEngines()
        _engines.update { it.copy(fakes = false, asr = EngineStatus.Loading("opening", 0), nlu = EngineStatus.Loading("waiting", 0), asrId = asrPack.id, nluId = nluPack.id) }
        scope.launch(Dispatchers.Default) {
            val device = _engines.value.device
            val t0 = started.elapsedNow().inWholeMilliseconds
            val ticker = launch { while (true) { delay(250); val e = started.elapsedNow().inWholeMilliseconds - t0; _engines.update { s -> s.copy(asr = (s.asr as? EngineStatus.Loading)?.copy(elapsedMs = e) ?: s.asr, nlu = (s.nlu as? EngineStatus.Loading)?.copy(elapsedMs = e) ?: s.nlu) } } }
            try {
                val a = env.engineFactory.asr(asrPack, "en", device)
                a.open() // loads the five graphs
                asr = a
                val asrMs = started.elapsedNow().inWholeMilliseconds - t0
                _engines.update { it.copy(asr = EngineStatus.Ready(asrMs, "${asrPack.descriptor.target.accelerator ?: "cpu"}"), nlu = EngineStatus.Loading("warm-up: catalog prefill", asrMs)) }
                log("asr ready in $asrMs ms (${asrPack.id})")
                val t1 = started.elapsedNow().inWholeMilliseconds
                val n = env.engineFactory.nlu(nluPack, device)
                n.warmUp()
                nlu = n
                val nluMs = started.elapsedNow().inWholeMilliseconds - t1
                _engines.update { it.copy(nlu = EngineStatus.Ready(nluMs, "${nluPack.descriptor.target.accelerator ?: "cpu"}")) }
                log("nlu ready in $nluMs ms (${nluPack.id})")
                rebuildPipeline()
            } catch (e: Throwable) {
                val reason = e.message ?: e::class.simpleName ?: "failed"
                _engines.update { s -> s.copy(asr = if (asr == null) EngineStatus.Failed(reason) else s.asr, nlu = if (nlu == null) EngineStatus.Failed(reason) else s.nlu) }
                log("engine load failed: $reason")
            } finally { ticker.cancel() }
        }
    }

    private fun rebuildPipeline() {
        val n = nlu ?: return
        pipelineJob?.cancel()
        val p = VoicePipeline(asr, n, HomeActions(homeStore).router(), nluBudgetMs = env.nluBudgetMs)
        pipeline = p
        pipelineJob = scope.launch { p.events.collect(::onEvent) }
        scope.launch { p.state.collect { _pipelineState.value = it } }
    }

    private fun closeEngines() {
        stopListening()
        pipelineJob?.cancel(); pipelineJob = null; pipeline = null
        runCatching { asr?.close() }; runCatching { nlu?.close() }
        asr = null; nlu = null
    }

    // ---- talking -------------------------------------------------------------------------------------------

    val canTalk: Boolean get() = pipeline != null && asr != null && env.audioSource != null && _pipelineState.value == PipelineState.IDLE

    /** Push-to-talk: press. Frames flow from the microphone into the pipeline until [stopListening]. */
    fun startListening() {
        val p = pipeline ?: return
        val source = env.audioSource ?: return
        if (_pipelineState.value != PipelineState.IDLE) return
        val channel = Channel<FloatArray>(capacity = 64)
        frames = channel
        captureJob = scope.launch(Dispatchers.Default) {
            try { source.frames().collect { channel.send(it) } } catch (_: Exception) {} finally { channel.close() }
        }
        utteranceJob = scope.launch {
            val mic = channel.consumeAsFlow()
            val run = p.runUtterance(env.endpointer?.bound(mic) ?: mic, source = "mic")
            log(runLine(run))
        }
    }

    /** Push-to-talk: release. */
    fun stopListening() {
        captureJob?.cancel(); captureJob = null
        frames?.close(); frames = null
    }

    fun runText(text: String) {
        val p = pipeline ?: return
        if (text.isBlank()) return
        scope.launch { log(runLine(p.runText(text))) }
    }

    fun runGoldenSet() {
        val p = pipeline ?: return
        scope.launch {
            _golden.value = emptyList()
            for (case in GoldenSet.cases) {
                val run = p.runText(case.utterance)
                _golden.update { it + GoldenResult(case, run) }
            }
            val results = _golden.value
            log("golden set: ${results.count { it.passed }}/${results.size} passed, nlu p50 ${median(results.map { it.run.nluMs })} ms")
        }
    }

    fun resetHome() { homeStore.reset(); log("home reset") }

    private fun runLine(run: PipelineRun): String = buildString {
        append('"').append(run.transcript).append('"')
        append(" → ").append(run.calledTool ?: (run.outcome as? NluOutcome.NoCall)?.let { "no call" } ?: run.failure ?: "?")
        run.action?.let { append(" → ").append(it.message) }
        append(" [asr ${run.asrMs} ms, nlu ${run.nluMs} ms]")
    }

    private fun onEvent(e: PipelineEvent) {
        _run.update { v ->
            when (e) {
                is PipelineEvent.Started -> RunView(source = e.source).withStage(Stage.LISTENING, StageStatus.ACTIVE)
                is PipelineEvent.Level -> v.copy(level = e.rms)
                is PipelineEvent.Partial -> v.copy(partial = e.text).withStage(Stage.ASR, StageStatus.ACTIVE)
                is PipelineEvent.Transcribed -> v.copy(transcript = e.text, partial = e.text, audioMs = e.audioMs, asrMs = e.asrMs, level = 0f)
                    .withStage(Stage.LISTENING, StageStatus.DONE).withStage(Stage.ASR, if (e.source() == "text") StageStatus.SKIPPED else StageStatus.DONE)
                is PipelineEvent.Resolving -> v.withStage(Stage.NLU, StageStatus.ACTIVE)
                is PipelineEvent.Resolved -> v.copy(outcome = e.outcome, nluMs = e.nluMs).withStage(Stage.NLU, if (e.outcome is NluOutcome.Failed) StageStatus.FAILED else StageStatus.DONE)
                is PipelineEvent.Corrected -> v.copy(correction = "rule: ${e.correction.reason} → ${e.correction.to.name}(${e.correction.to.args.entries.joinToString { "${it.key}=\"${it.value}\"" }})")
                is PipelineEvent.Acted -> {
                    if (e.result.ok) flash(e.result.changed)
                    v.copy(actionMessage = e.result.message, actionOk = e.result.ok).withStage(Stage.ACTION, if (e.result.ok) StageStatus.DONE else StageStatus.FAILED)
                }
                is PipelineEvent.Escalated -> v.copy(escalation = "${e.reason} → ${e.result::class.simpleName}: ${e.result.describe()}")
                    .withStage(Stage.ACTION, if (v.stages[Stage.ACTION] == StageStatus.PENDING) StageStatus.SKIPPED else v.stages.getValue(Stage.ACTION))
                    .withStage(Stage.CLOUD, StageStatus.SKIPPED)
                is PipelineEvent.Failed -> v.copy(failure = e.reason, level = 0f).withStage(e.stage, StageStatus.FAILED)
                is PipelineEvent.Finished -> v.copy(totalMs = e.totalMs, level = 0f)
            }
        }
    }

    private fun PipelineEvent.Transcribed.source(): String = _run.value.source

    private fun sk.ainet.examples.smarthome.hybrid.EscalationResult.describe(): String = when (this) {
        is sk.ainet.examples.smarthome.hybrid.EscalationResult.Handled -> "$message ($source)"
        is sk.ainet.examples.smarthome.hybrid.EscalationResult.NotHandled -> reason
    }

    private fun RunView.withStage(stage: Stage, status: StageStatus) = copy(stages = stages + (stage to status))

    private fun flash(devices: Set<DeviceRef>) {
        if (devices.isEmpty()) return
        _recentlyChanged.update { it + devices }
        scope.launch { delay(900); _recentlyChanged.update { it - devices } }
    }

    // ---- cartridges ----------------------------------------------------------------------------------------

    fun refreshCartridges() {
        scope.launch(Dispatchers.Default) {
            val installed = env.store.installed()
            _cartridges.update { it.copy(installed = installed) }
        }
    }

    fun setServerUrl(url: String) { _serverUrl.value = url; env.settings.serverUrl = url }

    fun fetchIndex() {
        scope.launch {
            _cartridges.update { it.copy(indexError = null) }
            try {
                val index = withContext(Dispatchers.Default) { downloader.index(_serverUrl.value) }
                _cartridges.update { it.copy(index = index) }
                log("server lists ${index.cartridges.size} cartridges")
            } catch (e: Exception) {
                _cartridges.update { it.copy(indexError = e.message ?: "cannot reach server") }
            }
        }
    }

    fun download(entry: CartridgeIndex.Entry) {
        scope.launch(Dispatchers.Default) {
            downloader.download(_serverUrl.value, entry).collect { ev ->
                _cartridges.update { it.copy(downloads = it.downloads + (entry.id to ev)) }
                if (ev is DownloadEvent.Done) { log("downloaded ${entry.id} (${ev.bytes / 1_000_000} MB)"); refreshCartridges() }
                if (ev is DownloadEvent.Failed) log("download failed ${entry.id}: ${ev.reason}")
            }
        }
    }

    fun verify(cartridge: InstalledCartridge) {
        scope.launch(Dispatchers.Default) {
            _cartridges.update { it.copy(verification = it.verification + (cartridge.id to "verifying…")) }
            val v = env.store.verify(cartridge)
            val text = if (v.ok) "verified ${v.checked} artifacts" else v.problems.joinToString("; ")
            _cartridges.update { it.copy(verification = it.verification + (cartridge.id to text)) }
            log("${cartridge.id}: $text")
        }
    }

    fun delete(cartridge: InstalledCartridge) {
        scope.launch(Dispatchers.Default) { env.store.delete(cartridge.id); refreshCartridges(); log("deleted ${cartridge.id}") }
    }

    private fun log(line: String) {
        val t = started.elapsedNow().inWholeMilliseconds
        _log.update { (it + "${(t / 1000).toString().padStart(4)}s  $line").takeLast(200) }
    }

    private fun median(values: List<Long>): Long = if (values.isEmpty()) 0 else values.sorted()[values.size / 2]
}
