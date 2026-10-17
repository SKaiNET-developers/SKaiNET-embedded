package sk.ainet.examples.smarthome.cartridges

import sk.ainet.examples.smarthome.cartridge.CartridgeJson
import kotlin.test.Test
import kotlin.test.assertEquals

class PackDevicesTest {
    private val descriptor = CartridgeJson.descriptor(
        """{"id":"asr","version":"0","family":"moonshine-v2-streaming","task":"asr",
           "attributes":{"languages":["en"],"devices":{"encoder":"torq","decoder":"local-task"}}}""",
    )

    @Test
    fun `the pack places its stages and unnamed stages run on the CPU`() {
        val d = PackDevices.of(descriptor)
        assertEquals("torq", d.of("encoder"))
        assertEquals("local-task", d.of("decoder"))
        assertEquals("local-task", d.of("frontend"))
        assertEquals("encoder=torq, decoder=local-task", d.describe())
    }

    @Test
    fun `the user's preference overrides the pack`() {
        val d = PackDevices.of(descriptor)
        assertEquals("local-task", d.of("encoder", DevicePreference.CPU))
        assertEquals("vulkan", d.of("decoder", DevicePreference.GPU))
    }

    @Test
    fun `a pack without devices runs on the CPU`() {
        val d = PackDevices.of(CartridgeJson.descriptor("""{"id":"x","version":"0","family":"f","task":"nlu"}"""))
        assertEquals("local-task", d.of("decoder"))
        assertEquals("all local-task", d.describe())
        assertEquals("all local-task", PackDevices.of("/nonexistent").describe())
    }
}
