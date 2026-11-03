package sk.ainet.examples.smarthome

import io.ktor.client.request.get
import io.ktor.client.statement.bodyAsText
import io.ktor.http.HttpStatusCode
import io.ktor.server.testing.testApplication
import sk.ainet.examples.smarthome.cartridge.CartridgeJson
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ApplicationTest {
    private fun fakePacks(): File {
        val root = File(System.getProperty("java.io.tmpdir"), "hes-packs-${System.nanoTime()}").apply { mkdirs() }
        val pack = File(root, "asr-test-cpu-arm64").apply { mkdirs() }
        File(pack, "descriptor.json").writeText(
            """{"spec_version":"0.4","id":"asr-test-cpu-arm64","version":"0.1.0","family":"moonshine-v2-streaming","modality":"audio","task":"asr",
               "target":{"hardware":"IREE-arm64-v8a","abi":"arm64-v8a"},"attributes":{"languages":["en"]},"license":"MIT"}""",
        )
        File(pack, "artifacts/model").mkdirs()
        File(pack, "artifacts/model/a.vmfb").writeText("abc")
        File(pack, "manifest.json").writeText(
            """{"schema_version":2,"kind":"cartridge-manifest","id":"asr-test-cpu-arm64","version":"0.1.0","digest_alg":"sha256",
               "artifacts":[{"role":"model","path":"artifacts/model/a.vmfb","digest":"sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","size":3,"license":"MIT","derived_from":[]}],
               "signatures":[]}""",
        )
        return root
    }

    @Test
    fun `index lists packs with their files and the files are served`() = testApplication {
        val root = fakePacks()
        application { module(CartridgeCatalog(root)) }
        val index = CartridgeJson.index(client.get("/cartridges/index.json").bodyAsText())
        val entry = index.cartridges.single()
        assertEquals("asr-test-cpu-arm64", entry.id)
        assertEquals("asr", entry.task)
        assertEquals(listOf("en"), entry.languages)
        assertEquals(setOf("descriptor.json", "manifest.json", "artifacts/model/a.vmfb"), entry.files.map { it.path }.toSet())
        assertEquals("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", entry.files.first { it.path.endsWith("vmfb") }.sha256)

        val file = client.get("/cartridges/asr-test-cpu-arm64/artifacts/model/a.vmfb")
        assertEquals(HttpStatusCode.OK, file.status)
        assertEquals("abc", file.bodyAsText())
        assertTrue(client.get("/").bodyAsText().startsWith("HomeEvenSmarter cartridge server"))

        val tail = client.get("/cartridges/asr-test-cpu-arm64/artifacts/model/a.vmfb") { headers.append("Range", "bytes=1-") }
        assertEquals(HttpStatusCode.PartialContent, tail.status)
        assertEquals("bc", tail.bodyAsText())
    }
}
