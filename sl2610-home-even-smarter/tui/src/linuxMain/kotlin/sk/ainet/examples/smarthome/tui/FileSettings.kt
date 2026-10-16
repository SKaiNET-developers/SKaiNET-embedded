package sk.ainet.examples.smarthome.tui

import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readString
import kotlinx.io.writeString
import sk.ainet.examples.smarthome.AppEnvironment

/** `key=value` lines in `~/.homeevensmarter/settings.properties`. */
class FileSettings(private val file: Path) : AppEnvironment.Settings {
    private val props: MutableMap<String, String> = load()

    private fun load(): MutableMap<String, String> {
        if (!SystemFileSystem.exists(file)) return mutableMapOf()
        val text = SystemFileSystem.source(file).buffered().use { it.readString() }
        return text.lineSequence()
            .map { it.trim() }
            .filter { it.isNotEmpty() && !it.startsWith("#") && '=' in it }
            .associateTo(mutableMapOf()) { it.substringBefore('=').trim() to it.substringAfter('=').trim().replace("\\:", ":") }
    }

    private fun save() {
        SystemFileSystem.sink(file).buffered().use { out -> props.forEach { (k, v) -> out.writeString("$k=$v\n") } }
    }

    override var serverUrl: String
        get() = props["serverUrl"] ?: "http://localhost:8080"
        set(value) { props["serverUrl"] = value; save() }

    override var devicePreference: String
        get() = props["device"] ?: "AUTO"
        set(value) { props["device"] = value; save() }
}
