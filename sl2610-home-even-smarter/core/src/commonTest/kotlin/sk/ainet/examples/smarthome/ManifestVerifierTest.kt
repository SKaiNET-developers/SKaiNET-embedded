package sk.ainet.examples.smarthome

import sk.ainet.examples.smarthome.cartridge.CartridgeJson
import sk.ainet.examples.smarthome.cartridge.CartridgeManifest
import sk.ainet.examples.smarthome.cartridge.ManifestVerifier
import sk.ainet.examples.smarthome.cartridge.Sha256
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ManifestVerifierTest {
    @Test
    fun `sha256 known answers`() {
        assertEquals("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Sha256.hex(ByteArray(0)))
        assertEquals("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", Sha256.hex("abc".encodeToByteArray()))
        assertEquals(
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            Sha256.hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".encodeToByteArray()),
        )
        val million = ByteArray(1_000_000) { 'a'.code.toByte() }
        assertEquals("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", Sha256.hex(million))
        // streaming across block boundaries gives the same digest
        val s = Sha256(); for (i in 0 until 1_000_000 step 1000) s.update(million, i, 1000)
        assertEquals("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", Sha256.toHex(s.digest()))
    }

    @Test
    fun `manifest parsing and verification`() {
        val manifest = CartridgeJson.manifest(
            """{"schema_version":2,"kind":"cartridge-manifest","id":"x","version":"0.1.0","digest_alg":"sha256",
               "artifacts":[{"role":"model","path":"artifacts/model/a.vmfb","digest":"sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","size":3,"license":"MIT","derived_from":[]}],
               "signatures":[]}""",
        )
        assertEquals(3, manifest.totalBytes)
        val ok = ManifestVerifier.verify(manifest) { "abc".encodeToByteArray() }
        assertTrue(ok.ok); assertEquals(1, ok.checked)
        val wrong = ManifestVerifier.verify(manifest) { "abd".encodeToByteArray() }
        assertFalse(wrong.ok); assertTrue(wrong.problems.single().contains("sha256"))
        val missing = ManifestVerifier.verify(manifest) { null }
        assertEquals("artifacts/model/a.vmfb: missing", missing.problems.single())
    }

    @Test
    fun `descriptor subset`() {
        val d = CartridgeJson.descriptor(
            """{"spec_version":"0.4","id":"asr-x","version":"0.1.0","family":"moonshine-v2-streaming","modality":"audio","task":"asr",
               "io":{"mode":"streaming"},"target":{"hardware":"IREE-arm64-v8a","abi":"arm64-v8a","accelerator":"Vulkan GPU (valhall4)"},
               "attributes":{"languages":["en"],"endpointing":"none"},"license":"MIT"}""",
        )
        assertEquals("asr", d.task); assertEquals(listOf("en"), d.languages); assertTrue(d.usesAccelerator)
    }
}
