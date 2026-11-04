package sk.ainet.examples.smarthome.cartridges

import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.files.SystemTemporaryDirectory
import kotlinx.io.writeString
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class CartridgeStoreTest {
    // two variants of one blueprint carry the same descriptor id; the store tells them apart by directory
    private fun descriptor(task: String, devices: String) = """{"spec_version":"0.4","id":"asr-moonshine-v2-streaming-iree","version":"0.0.0-dev",
        "family":"moonshine-v2-streaming","modality":"audio","task":"$task","target":{"hardware":"SL26xx","abi":"linux-aarch64"},
        "attributes":{"devices":$devices},"license":"MIT"}"""
    private val manifest = """{"schema_version":2,"id":"x","version":"0.0.0-dev","digest_alg":"sha256","artifacts":[]}"""

    private fun store(): CartridgeStore {
        val tmp = SystemTemporaryDirectory.takeIf { it.isAbsolute } ?: Path("/tmp")
        val root = Path(tmp, "hes-store-${Random.nextLong()}")
        fun pack(id: String, task: String, devices: String) {
            val dir = Path(root, id).also { SystemFileSystem.createDirectories(it) }
            SystemFileSystem.sink(Path(dir, "descriptor.json")).buffered().use { it.writeString(descriptor(task, devices)) }
            SystemFileSystem.sink(Path(dir, "manifest.json")).buffered().use { it.writeString(manifest) }
        }
        pack("asr-a-cpu", "asr", """{"encoder":"local-task"}""")
        pack("asr-b-npu", "asr", """{"encoder":"torq"}""")
        pack("nlu-c", "nlu", """{"decoder":"local-task"}""")
        return CartridgeStore(root)
    }

    @Test
    fun `installed cartridges are keyed by directory, not by the shared descriptor id`() {
        assertEquals(listOf("asr-a-cpu", "asr-b-npu", "nlu-c"), store().installed().map { it.id })
    }

    @Test
    fun `the preferred variant of a task is picked when installed, otherwise the first one`() {
        val s = store()
        assertEquals("asr-b-npu", s.asr("asr-b-npu")?.id)
        assertEquals("asr-a-cpu", s.asr("asr-gone")?.id)
        assertEquals("asr-a-cpu", s.asr()?.id)
        assertEquals("nlu-c", s.nlu("asr-b-npu")?.id, "a preference of another task does not apply")
        assertNull(CartridgeStore(Path("/nonexistent-${Random.nextLong()}")).asr("asr-b-npu"))
    }
}
