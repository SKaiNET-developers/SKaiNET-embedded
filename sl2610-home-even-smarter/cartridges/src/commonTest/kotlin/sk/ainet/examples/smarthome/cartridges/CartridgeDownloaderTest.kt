package sk.ainet.examples.smarthome.cartridges

import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.client.engine.mock.respondError
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.files.SystemTemporaryDirectory
import sk.ainet.examples.smarthome.cartridge.CartridgeIndex
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class CartridgeDownloaderTest {
    private val descriptor = """{"spec_version":"0.4","id":"asr-test","version":"0.1.0","family":"moonshine-v2-streaming","modality":"audio","task":"asr",
        "target":{"hardware":"IREE-arm64-v8a","abi":"arm64-v8a"},"attributes":{"languages":["en"]},"license":"MIT"}"""
    private val manifest = """{"schema_version":2,"kind":"cartridge-manifest","id":"asr-test","version":"0.1.0","digest_alg":"sha256",
        "artifacts":[{"role":"model","path":"artifacts/model/a.vmfb","digest":"sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","size":3,"license":"MIT","derived_from":[]}],"signatures":[]}"""

    private fun entry() = CartridgeIndex.Entry(
        id = "asr-test", version = "0.1.0", family = "moonshine-v2-streaming", task = "asr", abi = "arm64-v8a",
        sizeBytes = 3L + descriptor.length + manifest.length,
        files = listOf(
            CartridgeIndex.Entry.File("descriptor.json", descriptor.length.toLong()),
            CartridgeIndex.Entry.File("manifest.json", manifest.length.toLong()),
            CartridgeIndex.Entry.File("artifacts/model/a.vmfb", 3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        ),
    )

    // On Kotlin/Native an empty TMPDIR yields a relative temp directory (the module dir); fall back to /tmp then.
    private fun tempRoot(): Path {
        val tmp = SystemTemporaryDirectory.takeIf { it.isAbsolute } ?: Path("/tmp")
        return Path(tmp, "hes-store-${Random.nextLong()}").also { SystemFileSystem.createDirectories(it) }
    }

    @Test
    fun `downloads all files and verifies digests then the store sees the cartridge`() = runTest {
        val root = tempRoot()
        val engine = MockEngine { request ->
            when (request.url.encodedPath) {
                "/cartridges/asr-test/descriptor.json" -> respond(descriptor)
                "/cartridges/asr-test/manifest.json" -> respond(manifest)
                "/cartridges/asr-test/artifacts/model/a.vmfb" -> respond("abc")
                else -> respondError(HttpStatusCode.NotFound)
            }
        }
        val store = CartridgeStore(root)
        val events = CartridgeDownloader(store, engine).download("http://laptop:8080", entry()).toList()
        assertIs<DownloadEvent.Done>(events.last())
        assertTrue(events.filterIsInstance<DownloadEvent.Progress>().isNotEmpty())
        val installed = assertNotNull(store.asr())
        assertEquals("asr-test", installed.id)
        assertTrue(store.complete(installed))
        assertTrue(store.verify(installed).ok)
    }

    @Test
    fun `a digest mismatch fails and leaves no installed cartridge`() = runTest {
        val root = tempRoot()
        val engine = MockEngine { request ->
            when (request.url.encodedPath) {
                "/cartridges/asr-test/artifacts/model/a.vmfb" -> respond("abd")
                "/cartridges/asr-test/descriptor.json" -> respond(descriptor)
                "/cartridges/asr-test/manifest.json" -> respond(manifest)
                else -> respondError(HttpStatusCode.NotFound)
            }
        }
        val store = CartridgeStore(root)
        val events = CartridgeDownloader(store, engine).download("http://laptop:8080/", entry()).toList()
        val failed = assertIs<DownloadEvent.Failed>(events.last())
        assertTrue("sha256" in failed.reason)
        assertEquals(emptyList(), store.installed())
    }

    @Test
    fun `url helpers`() {
        assertEquals("http://h:1/cartridges/index.json", CartridgeDownloader.indexUrl("http://h:1"))
        assertEquals("http://h:1/cartridges/index.json", CartridgeDownloader.indexUrl("http://h:1/cartridges/index.json"))
        assertEquals("http://h:1/cartridges/x/artifacts/a.vmfb", CartridgeDownloader.fileUrl("http://h:1/", "x", "artifacts/a.vmfb"))
    }
}
