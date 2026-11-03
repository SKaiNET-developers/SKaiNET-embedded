package sk.ainet.examples.smarthome.tui

import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import com.jakewharton.mosaic.LocalTerminalState
import com.jakewharton.mosaic.layout.KeyEvent
import com.jakewharton.mosaic.layout.onKeyEvent
import com.jakewharton.mosaic.layout.padding
import com.jakewharton.mosaic.modifier.Modifier
import com.jakewharton.mosaic.ui.Color
import com.jakewharton.mosaic.ui.Column
import com.jakewharton.mosaic.ui.Row
import com.jakewharton.mosaic.ui.Text
import com.jakewharton.mosaic.ui.TextStyle
import sk.ainet.examples.smarthome.AppViewModel
import sk.ainet.examples.smarthome.CartridgesState
import sk.ainet.examples.smarthome.EnginesState
import sk.ainet.examples.smarthome.RunView
import sk.ainet.examples.smarthome.StageStatus
import sk.ainet.examples.smarthome.cartridges.DownloadEvent
import sk.ainet.examples.smarthome.engine.EngineStatus
import sk.ainet.examples.smarthome.engine.NluOutcome
import sk.ainet.examples.smarthome.home.DeviceRef
import sk.ainet.examples.smarthome.home.Door
import sk.ainet.examples.smarthome.home.HomeReducer
import sk.ainet.examples.smarthome.home.HomeState
import sk.ainet.examples.smarthome.home.Room
import sk.ainet.examples.smarthome.pipeline.PipelineState

private enum class Screen { HOME, CARTRIDGES }

private val Dim = Color(128, 128, 128)
private val Accent = Color(95, 215, 255)
private val Ok = Color(95, 215, 95)
private val Warn = Color(255, 215, 95)
private val Bad = Color(255, 95, 95)

/**
 * The whole terminal UI. Terminals report key presses but no releases, so talking is a toggle: space starts listening,
 * space again (or the endpointer, on silence) ends the utterance.
 */
@Composable
fun TuiApp(vm: AppViewModel, micLabel: String, onQuit: () -> Unit) {
    val home by vm.homeState.collectAsState()
    val run by vm.run.collectAsState()
    val pipelineState by vm.pipelineState.collectAsState()
    val engines by vm.engines.collectAsState()
    val cartridges by vm.cartridges.collectAsState()
    val changed by vm.recentlyChanged.collectAsState()
    val log by vm.log.collectAsState()
    val golden by vm.golden.collectAsState()
    val server by vm.serverUrl.collectAsState()
    var screen by remember { mutableStateOf(Screen.HOME) }
    var typing by remember { mutableStateOf<String?>(null) }
    val rows = LocalTerminalState.current.size.rows

    fun onKey(e: KeyEvent): Boolean {
        val input = typing
        if (input != null) {
            when {
                e.key == "Enter" -> { vm.runText(input); typing = null }
                e.key == "Escape" -> typing = null
                e.key == "Backspace" -> typing = input.dropLast(1)
                e.key.length == 1 && !e.ctrl -> typing = input + e.key
                else -> return false
            }
            return true
        }
        when (e.key) {
            " " -> if (pipelineState == PipelineState.LISTENING) vm.stopListening() else if (vm.canTalk) vm.startListening()
            "t", ":" -> typing = ""
            "g" -> vm.runGoldenSet()
            "r" -> vm.resetHome()
            "f" -> vm.useFakes(!engines.fakes)
            "l" -> vm.loadAndWarmUp()
            "i" -> vm.fetchIndex()
            "d" -> cartridges.index?.cartridges?.forEach(vm::download)
            "v" -> cartridges.installed.forEach(vm::verify)
            "Tab" -> screen = if (screen == Screen.HOME) Screen.CARTRIDGES else Screen.HOME
            "q" -> onQuit()
            else -> return false
        }
        return true
    }

    Column(modifier = Modifier.onKeyEvent(::onKey)) {
        Header(engines, pipelineState, micLabel, server)
        when (screen) {
            Screen.HOME -> {
                HomePanel(home, changed)
                PipelinePanel(run, pipelineState)
                if (golden.isNotEmpty()) {
                    val passed = golden.count { it.passed }
                    Line("golden  $passed/${golden.size} passed" + golden.filterNot { it.passed }.joinToString("") { "  ✗ \"${it.case.utterance}\" → ${it.run.calledTool ?: "no call"}" },
                        color = if (passed == golden.size) Ok else Warn)
                }
            }
            Screen.CARTRIDGES -> CartridgesPanel(cartridges, engines)
        }
        val logRows = (rows - if (screen == Screen.HOME) 24 else 16).coerceIn(3, 12)
        Line("─ log " + "─".repeat(40), color = Dim)
        log.takeLast(logRows).forEach { Line(it, color = Dim) }
        Footer(typing, screen)
    }
}

@Composable
private fun Header(engines: EnginesState, state: PipelineState, mic: String, server: String) {
    Row {
        Text("HomeEvenSmarter", color = Accent, textStyle = TextStyle.Bold)
        Text(" · $state · engines ${if (engines.fakes) "fakes" else "cartridges"} · device ${engines.device}")
    }
    Line("asr ${engines.asr.short()} · nlu ${engines.nlu.short()}", color = Dim)
    Line("mic $mic · server $server", color = Dim)
}

/**
 * One line of text clipped to the terminal width. A line that wraps breaks Mosaic's redraw (it moves the cursor up by
 * logical lines), leaving stale copies on screen, so nothing may be wider than the terminal.
 */
@Composable
private fun Line(value: String, color: Color = Color.Unspecified, textStyle: TextStyle = TextStyle.Unspecified) {
    val columns = LocalTerminalState.current.size.columns - 2
    Text(if (columns > 1 && value.length > columns) value.take(columns - 1) + "…" else value, color = color, textStyle = textStyle)
}

@Composable
private fun HomePanel(home: HomeState, changed: Set<DeviceRef>) {
    Line("─ home " + "─".repeat(40), color = Dim)
    for (room in Room.entries) {
        val r = home.room(room)
        Row(modifier = Modifier.padding(left = 1)) {
            Text(room.label.padEnd(13))
            Text(("light " + if (r.light.on) "on ${r.light.brightness}%" else "off").padEnd(15),
                color = if (DeviceRef.LightOf(room) in changed) Warn else if (r.light.on) Ok else Color.Unspecified)
            Text(HomeReducer.formatTemperature(r.thermostat.targetCelsius).padEnd(9),
                color = if (DeviceRef.ThermostatOf(room) in changed) Warn else Color.Unspecified)
            Text("blinds ${r.blinds.label.lowercase()}", color = if (DeviceRef.BlindsOf(room) in changed) Warn else Color.Unspecified)
        }
    }
    Row(modifier = Modifier.padding(left = 1)) {
        for (door in Door.entries) {
            val locked = home.locks[door] == true
            Text("${door.label} ${if (locked) "locked" else "unlocked"}   ",
                color = if (DeviceRef.LockOf(door) in changed) Warn else if (locked) Ok else Color.Unspecified)
        }
        Text("scene ${home.activeScene?.label ?: "–"}")
    }
}

@Composable
private fun PipelinePanel(run: RunView, state: PipelineState) {
    Line("─ pipeline " + "─".repeat(36), color = Dim)
    Row(modifier = Modifier.padding(left = 1)) {
        run.stages.forEach { (stage, status) ->
            val (color, style) = when (status) {
                StageStatus.PENDING -> Dim to TextStyle.Unspecified
                StageStatus.ACTIVE -> Warn to TextStyle.Bold
                StageStatus.DONE -> Ok to TextStyle.Unspecified
                StageStatus.FAILED -> Bad to TextStyle.Bold
                StageStatus.SKIPPED -> Dim to TextStyle.Strikethrough
            }
            Text("[${stage.name}]", color = color, textStyle = style)
            Text(" ")
        }
        if (state == PipelineState.LISTENING) Text(" level " + meter(run.level), color = Warn)
    }
    Column(modifier = Modifier.padding(left = 1)) {
        val heard = run.transcript.ifEmpty { run.partial }
        Line("heard   " + if (heard.isEmpty()) "–" else "\"$heard\"" + if (run.audioMs > 0) "  (${run.audioMs} ms audio, asr ${run.asrMs} ms)" else "")
        when (val o = run.outcome) {
            is NluOutcome.Call -> Line("call    ${o.name}(${o.args.entries.joinToString { "${it.key}=\"${it.value}\"" }})  nlu ${run.nluMs} ms " +
                "[tok ${o.timing.tokenizeMs} · restore ${o.timing.restoreMs} · chunk ${o.timing.chunkMs} · decode ${o.timing.decodeMs}/${o.timing.decodeTokens}t]", color = Accent)
            is NluOutcome.NoCall -> Line("call    none: \"${o.text.take(60)}\"  nlu ${run.nluMs} ms", color = Warn)
            is NluOutcome.Failed -> Line("call    failed: ${o.reason}", color = Bad)
            null -> Line("call    –", color = Dim)
        }
        if (run.correction.isNotEmpty()) Line("fix     ${run.correction}", color = Warn)
        Line("action  " + run.actionMessage.ifEmpty { "–" }, color = when (run.actionOk) { true -> Ok; false -> Bad; null -> Dim })
        if (run.escalation.isNotEmpty()) Line("cloud   ${run.escalation}", color = Dim)
        run.failure?.let { Line("failed  $it", color = Bad) }
        if (run.totalMs > 0) Line("total   ${run.totalMs} ms", color = Dim)
    }
}

@Composable
private fun CartridgesPanel(c: CartridgesState, engines: EnginesState) {
    Line("─ cartridges " + "─".repeat(34), color = Dim)
    Column(modifier = Modifier.padding(left = 1)) {
        Text("installed", textStyle = TextStyle.Bold)
        if (c.installed.isEmpty()) Text("  none", color = Dim)
        c.installed.forEach { Line("  ${it.id}  ${it.sizeBytes / 1_000_000} MB  ${c.verification[it.id] ?: ""}") }
        Text("server index", textStyle = TextStyle.Bold)
        c.indexError?.let { Line("  $it", color = Bad) }
        val index = c.index
        if (index == null && c.indexError == null) Text("  not fetched (press i)", color = Dim)
        index?.cartridges?.forEach { e ->
            val dl = when (val d = c.downloads[e.id]) {
                is DownloadEvent.Progress -> "${(d.fraction * 100).toInt()}% ${d.file}"
                is DownloadEvent.Verifying -> "verifying ${d.file}"
                is DownloadEvent.Done -> "done"
                is DownloadEvent.Failed -> "failed: ${d.reason}"
                null -> ""
            }
            Line("  ${e.id}  ${e.accelerator ?: "cpu"}  ${e.sizeBytes / 1_000_000} MB  $dl")
        }
        Line("engines: asr ${engines.asr.short()} · nlu ${engines.nlu.short()}", color = Dim)
    }
}

@Composable
private fun Footer(typing: String?, screen: Screen) {
    if (typing != null) {
        Row { Text("> ", color = Accent, textStyle = TextStyle.Bold); Text(typing); Text("█", color = Dim) }
        Line("enter run · esc cancel", color = Dim)
    } else {
        val keys = when (screen) {
            Screen.HOME -> "space talk · t type · g golden set · r reset · f fakes · tab cartridges · q quit"
            Screen.CARTRIDGES -> "i fetch index · d download · v verify · l load & warm up · tab home · q quit"
        }
        Line(keys, color = Dim)
    }
}

private fun EngineStatus.short(): String = when (this) {
    EngineStatus.Absent -> "absent"
    is EngineStatus.Missing -> "missing"
    is EngineStatus.Loading -> "$phase ${elapsedMs / 1000}s"
    is EngineStatus.Ready -> "ready" + if (detail.isNotEmpty()) " ($detail)" else ""
    is EngineStatus.Failed -> "failed: $reason"
}

private fun meter(rms: Float): String {
    val n = (rms / 0.2f * 20).toInt().coerceIn(0, 20)
    return "█".repeat(n) + "·".repeat(20 - n)
}
