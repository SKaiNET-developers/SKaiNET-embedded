package sk.ainet.examples.smarthome

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import sk.ainet.examples.smarthome.tools.HomeTools
import sk.ainet.examples.smarthome.tools.IntentMapper
import sk.ainet.examples.smarthome.tools.Mapping
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

/** The JSON catalog the NLU cartridge is materialized with is the contract; the Kotlin mirror must not drift. */
class CatalogContractTest {
    private val catalog = Json.parseToJsonElement(File("../cartridges/catalogs/home-tools.v1.json").readText()).jsonObject
    private val functions = catalog.getValue("functions").jsonArray.map { it.jsonObject }

    @Test
    fun `names match HomeTools and stay few`() {
        assertEquals(HomeTools.names, functions.map { it.getValue("name").jsonPrimitive.content }.toSet())
        assertTrue(functions.size <= 6, "a 270M model degrades beyond a handful of functions")
    }

    @Test
    fun `argument keys match and every enum value maps`() {
        for (f in functions) {
            val name = f.getValue("name").jsonPrimitive.content
            val params = f["parameters"]?.jsonObject ?: emptyMap()
            assertEquals(HomeTools.arguments.getValue(name), params.keys, "$name arguments")
            val required = f["required"]?.jsonArray?.map { it.jsonPrimitive.content } ?: emptyList()
            // build one intent per enum value of each parameter and make sure the mapper accepts it
            val sample = params.mapValues { (k, v) -> v.jsonObject["enum"]?.jsonArray?.first()?.jsonPrimitive?.content ?: if (k == "temperature") "21" else "50" }
            assertIs<Mapping.Command>(IntentMapper.map(name, sample.filterKeys { it in required || it != "brightness" }), "$name with $sample")
            for ((k, v) in params) v.jsonObject["enum"]?.jsonArray?.forEach { e ->
                val args = sample + (k to e.jsonPrimitive.content)
                assertIs<Mapping.Command>(IntentMapper.map(name, args), "$name $k=${e.jsonPrimitive.content}")
            }
        }
    }
}
