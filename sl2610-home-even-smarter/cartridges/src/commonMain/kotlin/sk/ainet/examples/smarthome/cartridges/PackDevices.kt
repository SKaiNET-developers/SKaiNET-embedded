package sk.ainet.examples.smarthome.cartridges

import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readString
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive
import sk.ainet.examples.smarthome.cartridge.CartridgeDescriptor
import sk.ainet.examples.smarthome.cartridge.CartridgeJson

/**
 * Where each graph (or stage) of a cartridge runs: the pack's `descriptor.attributes.devices`, e.g.
 * `{"frontend": "local-task", "encoder": "torq", "decoder": "local-task"}`, with IREE device names (`local-task` =
 * CPU, `torq` = NPU, `vulkan` = GPU). A stage the pack does not name runs on the CPU. The user's [DevicePreference]
 * overrides the pack: CPU puts everything on the CPU, GPU everything on Vulkan.
 */
class PackDevices(private val declared: Map<String, String>) {
    fun of(stage: String, preference: DevicePreference = DevicePreference.AUTO): String = when (preference) {
        DevicePreference.AUTO -> declared[stage] ?: CPU
        DevicePreference.CPU -> CPU
        DevicePreference.GPU -> GPU
    }

    /** What the pack declares, for logs and the engine panel. */
    fun describe(): String = if (declared.isEmpty()) "all $CPU" else declared.entries.joinToString { "${it.key}=${it.value}" }

    companion object {
        const val CPU = "local-task"
        const val GPU = "vulkan"

        fun of(descriptor: CartridgeDescriptor): PackDevices = PackDevices(
            (descriptor.attributes?.get("devices") as? JsonObject)?.mapValues { it.value.jsonPrimitive.content } ?: emptyMap(),
        )

        /** From `<packDir>/descriptor.json`; none declared when it is missing or unreadable. */
        fun of(packDir: String): PackDevices = runCatching {
            of(CartridgeJson.descriptor(SystemFileSystem.source(Path(packDir, "descriptor.json")).buffered().use { it.readString() }))
        }.getOrDefault(PackDevices(emptyMap()))
    }
}
