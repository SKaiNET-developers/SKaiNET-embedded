package sk.ainet.examples.smarthome.cartridge

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive

/** The subset of a cartridge `descriptor.json` the app reads. Open vocabulary elsewhere is ignored. */
@Serializable
data class CartridgeDescriptor(
    val id: String,
    val version: String,
    val family: String,
    val task: String,
    val modality: String = "",
    val target: Target = Target(),
    val attributes: JsonObject? = null,
) {
    @Serializable
    data class Target(val hardware: String = "", val abi: String = "", val accelerator: String? = null)

    val languages: List<String> get() = (attributes?.get("languages") as? JsonArray)?.map { it.jsonPrimitive.content } ?: emptyList()
    val usesAccelerator: Boolean get() = target.accelerator != null
}

/** The subset of `manifest.json` (schema_version 2) the app needs to fetch and verify a cartridge. */
@Serializable
data class CartridgeManifest(
    @SerialName("schema_version") val schemaVersion: Int = 2,
    val id: String,
    val version: String,
    @SerialName("digest_alg") val digestAlg: String = "sha256",
    val artifacts: List<Artifact>,
) {
    @Serializable
    data class Artifact(val role: String, val path: String, val digest: String, val size: Long)

    val totalBytes: Long get() = artifacts.sumOf { it.size }
}

/** What the companion server publishes at `/cartridges/index.json`. */
@Serializable
data class CartridgeIndex(val cartridges: List<Entry>) {
    @Serializable
    data class Entry(
        val id: String,
        val version: String,
        val family: String,
        val task: String,
        val abi: String,
        val accelerator: String? = null,
        val languages: List<String> = emptyList(),
        @SerialName("size_bytes") val sizeBytes: Long,
        /** Paths relative to the pack_dir, always including descriptor.json and manifest.json. */
        val files: List<File>,
    ) {
        @Serializable
        data class File(val path: String, @SerialName("size_bytes") val sizeBytes: Long, val sha256: String? = null)
    }
}

object CartridgeJson {
    val json: Json = Json { ignoreUnknownKeys = true; isLenient = true; encodeDefaults = true; prettyPrint = true }
    fun descriptor(text: String): CartridgeDescriptor = json.decodeFromString(CartridgeDescriptor.serializer(), text)
    fun manifest(text: String): CartridgeManifest = json.decodeFromString(CartridgeManifest.serializer(), text)
    fun index(text: String): CartridgeIndex = json.decodeFromString(CartridgeIndex.serializer(), text)
}

/** Well-known family/task values of the two cartridges this app consumes. */
object CartridgeKinds {
    const val TASK_ASR = "asr"
    const val TASK_NLU = "nlu"
}
