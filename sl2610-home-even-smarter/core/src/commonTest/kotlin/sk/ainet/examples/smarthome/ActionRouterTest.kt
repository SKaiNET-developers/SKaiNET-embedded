package sk.ainet.examples.smarthome

import sk.ainet.examples.smarthome.actions.ActionRouter
import sk.ainet.examples.smarthome.actions.HomeActions
import sk.ainet.examples.smarthome.actions.HomeStore
import sk.ainet.examples.smarthome.actions.Intent
import sk.ainet.examples.smarthome.home.DeviceRef
import sk.ainet.examples.smarthome.home.Room
import sk.ainet.examples.smarthome.tools.HomeTools
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ActionRouterTest {
    @Test
    fun `home router registers exactly the catalog functions`() {
        val router = HomeActions(HomeStore()).router()
        assertEquals(HomeTools.names.sorted(), router.tools)
    }

    @Test
    fun `dispatch mutates the store and reports what changed`() {
        val store = HomeStore()
        val router = HomeActions(store).router()
        val r = router.dispatch(Intent(HomeTools.SET_LIGHT, mapOf("room" to "kitchen", "state" to "on")))
        assertTrue(r.ok)
        assertTrue(store.current.room(Room.KITCHEN).light.on)
        assertEquals(setOf(DeviceRef.LightOf(Room.KITCHEN)), r.changed)
    }

    @Test
    fun `unknown tool and unmappable arguments fail softly`() {
        val router = HomeActions(HomeStore()).router()
        assertFalse(router.dispatch(Intent("order_pizza")).ok)
        val r = router.dispatch(Intent(HomeTools.SET_BLINDS, mapOf("room" to "garage", "position" to "open")))
        assertFalse(r.ok)
        assertTrue("room" in r.message)
    }

    @Test
    fun `handler exceptions become failed results`() {
        val router = ActionRouter().register("boom") { throw IllegalStateException("kaboom") }
        val r = router.dispatch(Intent("boom"))
        assertFalse(r.ok)
        assertEquals("handler error: kaboom", r.message)
    }
}
