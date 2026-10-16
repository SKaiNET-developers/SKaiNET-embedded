package sk.ainet.examples.smarthome.cartridges

import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readByteArray
import kotlinx.io.readString
import sk.ainet.examples.smarthome.cartridge.CartridgeDescriptor
import sk.ainet.examples.smarthome.cartridge.CartridgeJson
import sk.ainet.examples.smarthome.cartridge.CartridgeKinds
import sk.ainet.examples.smarthome.cartridge.CartridgeManifest
import sk.ainet.examples.smarthome.cartridge.ManifestVerifier
import sk.ainet.examples.smarthome.cartridge.Verification

/** A materialized cartridge found on disk. */
data class InstalledCartridge(val dir: Path, val descriptor: CartridgeDescriptor, val manifest: CartridgeManifest) {
    val id: String get() = descriptor.id
    val task: String get() = descriptor.task
    val sizeBytes: Long get() = manifest.totalBytes
    val usesAccelerator: Boolean get() = descriptor.usesAccelerator
    fun file(relative: String): Path = Path(dir, relative)
}

/**
 * The app's local cartridge directory: `<root>/<cartridge id>/{descriptor.json,manifest.json,artifacts/…}` —
 * exactly the pack_dir layout a materialization produces, so a pack can also be pushed with adb as it is.
 */
class CartridgeStore(val root: Path) {
    private val fs = SystemFileSystem

    fun installed(): List<InstalledCartridge> {
        if (!fs.exists(root)) return emptyList()
        return fs.list(root).sortedBy { it.name }.mapNotNull { dir -> read(dir) }
    }

    fun read(dir: Path): InstalledCartridge? {
        val d = Path(dir, "descriptor.json"); val m = Path(dir, "manifest.json")
        if (!fs.exists(d) || !fs.exists(m)) return null
        return runCatching {
            InstalledCartridge(dir, CartridgeJson.descriptor(readText(d)), CartridgeJson.manifest(readText(m)))
        }.getOrNull()
    }

    fun byId(id: String): InstalledCartridge? = read(Path(root, id))

    fun asr(): InstalledCartridge? = installed().firstOrNull { it.task == CartridgeKinds.TASK_ASR }
    fun nlu(): InstalledCartridge? = installed().firstOrNull { it.task == CartridgeKinds.TASK_NLU }

    /** Full digest check of every artifact against the manifest (reads gigabytes for the NLU pack — call off the main thread). */
    fun verify(cartridge: InstalledCartridge): Verification =
        ManifestVerifier.verify(cartridge.manifest) { rel -> val p = cartridge.file(rel); if (fs.exists(p)) readBytes(p) else null }

    /** Cheap check: every artifact present with the manifest's size. */
    fun complete(cartridge: InstalledCartridge): Boolean = cartridge.manifest.artifacts.all { a ->
        val p = cartridge.file(a.path); fs.exists(p) && fs.metadataOrNull(p)?.size == a.size
    }

    fun delete(id: String) { deleteRecursively(Path(root, id)) }

    fun deleteRecursively(path: Path) {
        if (!fs.exists(path)) return
        if (fs.metadataOrNull(path)?.isDirectory == true) fs.list(path).forEach { deleteRecursively(it) }
        fs.delete(path, mustExist = false)
    }

    private fun readText(p: Path): String = fs.source(p).buffered().use { it.readString() }
    private fun readBytes(p: Path): ByteArray = fs.source(p).buffered().use { it.readByteArray() }
}
