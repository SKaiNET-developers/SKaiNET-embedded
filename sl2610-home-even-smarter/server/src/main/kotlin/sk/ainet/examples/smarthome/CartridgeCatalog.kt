package sk.ainet.examples.smarthome

import sk.ainet.examples.smarthome.cartridge.CartridgeIndex
import sk.ainet.examples.smarthome.cartridge.CartridgeJson
import java.io.File

/**
 * Scans a directory of materialized pack_dirs (`<root>/<cartridge id>/{descriptor.json,manifest.json,artifacts/…}`)
 * into the index the app downloads. Digests come from each manifest — the server never hashes multi-gigabyte
 * archives itself; the board verifies what it received against the signed manifest.
 */
class CartridgeCatalog(val root: File) {

    fun scan(): CartridgeIndex {
        val packs = root.listFiles { f -> f.isDirectory && File(f, "descriptor.json").isFile && File(f, "manifest.json").isFile }
            ?.sortedBy { it.name } ?: emptyList()
        return CartridgeIndex(packs.mapNotNull { pack -> runCatching { entry(pack) }.getOrNull() })
    }

    private fun entry(pack: File): CartridgeIndex.Entry {
        val descriptor = CartridgeJson.descriptor(File(pack, "descriptor.json").readText())
        val manifest = CartridgeJson.manifest(File(pack, "manifest.json").readText())
        val files = listOf("descriptor.json", "manifest.json").map { CartridgeIndex.Entry.File(it, File(pack, it).length()) } +
            manifest.artifacts.map { CartridgeIndex.Entry.File(it.path, it.size, it.digest.removePrefix("sha256:")) }
        return CartridgeIndex.Entry(
            id = pack.name,
            version = descriptor.version,
            family = descriptor.family,
            task = descriptor.task,
            abi = descriptor.target.abi,
            accelerator = descriptor.target.accelerator,
            languages = descriptor.languages,
            sizeBytes = files.sumOf { it.sizeBytes },
            files = files,
        )
    }
}
