package sk.ainet.examples.smarthome

import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.http.content.staticFiles
import io.ktor.server.netty.Netty
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.plugins.partialcontent.PartialContent
import io.ktor.server.response.respond
import io.ktor.server.response.respondText
import io.ktor.server.routing.get
import io.ktor.server.routing.routing
import sk.ainet.examples.smarthome.cartridge.CartridgeJson
import java.io.File
import java.net.NetworkInterface

/**
 * The companion server: run it on the development host, point the board at it, and the app pulls the
 * materialized cartridges over WiFi. Serves `GET /cartridges/index.json` plus every file of every pack_dir below
 * the cartridges directory (`-Dcartridges=<dir>` or the CARTRIDGES_DIR environment variable; default
 * `build/cartridges`).
 */
fun main() {
    val dir = File(System.getProperty("cartridges") ?: System.getenv("CARTRIDGES_DIR") ?: "build/cartridges").absoluteFile
    val port = (System.getProperty("port") ?: System.getenv("PORT") ?: "8080").toInt()
    val catalog = CartridgeCatalog(dir)
    println("HomeEvenSmarter cartridge server")
    println("  cartridges: $dir")
    for (c in catalog.scan().cartridges) println("  - ${c.id} ${c.version} (${c.abi}${c.accelerator?.let { ", $it" } ?: ""}, ${c.sizeBytes / 1_000_000} MB)")
    for (url in localUrls(port)) println("  url: $url")
    embeddedServer(Netty, port = port, host = "0.0.0.0") { module(catalog) }.start(wait = true)
}

fun Application.module(catalog: CartridgeCatalog = CartridgeCatalog(File("build/cartridges"))) {
    install(ContentNegotiation) { json(CartridgeJson.json) }
    install(PartialContent) // Range requests, so an interrupted 1.7 GB download resumes instead of restarting
    routing {
        get("/") {
            val index = catalog.scan()
            call.respondText(
                "HomeEvenSmarter cartridge server\n" + index.cartridges.joinToString("") { "${it.id} ${it.version} ${it.sizeBytes} bytes\n" },
                ContentType.Text.Plain,
            )
        }
        get("/cartridges/index.json") { call.respond(catalog.scan()) }
        get("/health") { call.respond(HttpStatusCode.OK, mapOf("ok" to true)) }
        staticFiles("/cartridges", catalog.root)
    }
}

private fun localUrls(port: Int): List<String> = try {
    NetworkInterface.getNetworkInterfaces().toList()
        .filter { it.isUp && !it.isLoopback }
        .flatMap { it.inetAddresses.toList() }
        .filter { it.address.size == 4 }
        .map { "http://${it.hostAddress}:$port/cartridges/index.json" }
} catch (_: Exception) { emptyList() }
