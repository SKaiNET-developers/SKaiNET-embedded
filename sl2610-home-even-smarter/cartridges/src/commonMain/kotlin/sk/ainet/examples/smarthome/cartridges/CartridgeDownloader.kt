package sk.ainet.examples.smarthome.cartridges

import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.engine.HttpClientEngine
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.request.get
import io.ktor.client.request.header
import io.ktor.client.request.prepareGet
import io.ktor.client.statement.bodyAsText
import io.ktor.http.HttpStatusCode
import io.ktor.http.isSuccess
import io.ktor.utils.io.ByteReadChannel
import io.ktor.utils.io.readAvailable
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.write
import sk.ainet.examples.smarthome.cartridge.CartridgeIndex
import sk.ainet.examples.smarthome.cartridge.CartridgeJson
import sk.ainet.examples.smarthome.cartridge.Sha256

/** Progress of one cartridge download, for the cartridge screen. */
sealed interface DownloadEvent {
    data class Progress(val cartridgeId: String, val file: String, val bytesDone: Long, val bytesTotal: Long) : DownloadEvent {
        val fraction: Float get() = if (bytesTotal == 0L) 0f else bytesDone.toFloat() / bytesTotal
    }
    data class Verifying(val cartridgeId: String, val file: String) : DownloadEvent
    data class Done(val cartridgeId: String, val bytes: Long) : DownloadEvent
    data class Failed(val cartridgeId: String, val reason: String) : DownloadEvent
}

/**
 * Pulls a cartridge from the companion server into the [CartridgeStore]: every file listed in the index, resumed by
 * byte range when a partial file exists, digest-checked while streaming when the index carries a SHA-256. The result
 * on disk is a complete pack_dir; `descriptor.json` and `manifest.json` are written last so a half-downloaded
 * cartridge is never mistaken for an installed one.
 */
class CartridgeDownloader(
    private val store: CartridgeStore,
    engine: HttpClientEngine? = null,
) {
    private val client = if (engine != null) HttpClient(engine) { install(HttpTimeout) { requestTimeoutMillis = 30 * 60_000 } }
    else HttpClient(CIO) { install(HttpTimeout) { requestTimeoutMillis = 30 * 60_000; socketTimeoutMillis = 60_000 } }
    private val fs = SystemFileSystem

    suspend fun index(baseUrl: String): CartridgeIndex {
        val response = client.get(indexUrl(baseUrl))
        if (!response.status.isSuccess()) error("server answered ${response.status} for ${indexUrl(baseUrl)}")
        return CartridgeJson.index(response.bodyAsText())
    }

    /** Downloads [entry] from [baseUrl]; emits progress and ends with [DownloadEvent.Done] or [DownloadEvent.Failed]. */
    // channelFlow, not flow: progress is sent from inside Ktor's execute block, which runs on another dispatcher on
    // Kotlin/Native (a plain flow's context invariant rejects that).
    fun download(baseUrl: String, entry: CartridgeIndex.Entry): Flow<DownloadEvent> = channelFlow {
        val dir = Path(store.root, entry.id)
        try {
            fs.createDirectories(dir)
            val total = entry.files.sumOf { it.sizeBytes }
            var done = 0L
            // metadata last, so an interrupted download never looks installed
            val ordered = entry.files.sortedBy { if (it.path == "descriptor.json" || it.path == "manifest.json") 1 else 0 }
            for (f in ordered) {
                val target = Path(dir, f.path)
                target.parent?.let { fs.createDirectories(it) }
                val existing = fs.metadataOrNull(target)?.size ?: 0L
                if (existing == f.sizeBytes && f.sha256 == null) { done += f.sizeBytes; send(DownloadEvent.Progress(entry.id, f.path, done, total)); continue }
                if (existing == f.sizeBytes && f.sha256 != null) {
                    send(DownloadEvent.Verifying(entry.id, f.path))
                    if (digestOf(target) == f.sha256) { done += f.sizeBytes; send(DownloadEvent.Progress(entry.id, f.path, done, total)); continue }
                    fs.delete(target)
                }
                val resumeFrom = if (existing in 1 until f.sizeBytes) existing else { if (existing > 0) fs.delete(target); 0L }
                done += resumeFrom
                fetch(fileUrl(baseUrl, entry.id, f.path), target, resumeFrom) { delta -> done += delta; send(DownloadEvent.Progress(entry.id, f.path, done, total)) }
                val size = fs.metadataOrNull(target)?.size ?: -1
                if (size != f.sizeBytes) throw IllegalStateException("${f.path}: got $size bytes, expected ${f.sizeBytes}")
                if (f.sha256 != null) {
                    send(DownloadEvent.Verifying(entry.id, f.path))
                    val actual = digestOf(target)
                    if (actual != f.sha256) { fs.delete(target); throw IllegalStateException("${f.path}: sha256 mismatch") }
                }
            }
            send(DownloadEvent.Done(entry.id, total))
        } catch (e: Exception) {
            send(DownloadEvent.Failed(entry.id, e.message ?: e::class.simpleName ?: "download failed"))
        }
    }

    private suspend fun fetch(url: String, target: Path, resumeFrom: Long, onBytes: suspend (Long) -> Unit) {
        client.prepareGet(url) { if (resumeFrom > 0) header("Range", "bytes=$resumeFrom-") }.execute { response ->
            if (!response.status.isSuccess()) error("$url: ${response.status}")
            val append = resumeFrom > 0 && response.status == HttpStatusCode.PartialContent
            if (resumeFrom > 0 && !append) fs.delete(target, mustExist = false)
            val sink = fs.sink(target, append = append).buffered()
            try {
                val channel: ByteReadChannel = response.body()
                val buffer = ByteArray(1 shl 16)
                while (true) {
                    val n = channel.readAvailable(buffer, 0, buffer.size)
                    if (n <= 0) break
                    sink.write(buffer, 0, n)
                    onBytes(n.toLong())
                }
            } finally {
                sink.close()
            }
        }
    }

    private fun digestOf(path: Path): String {
        val sha = Sha256()
        fs.source(path).buffered().use { src ->
            val buf = ByteArray(1 shl 16)
            while (true) { val n = src.readAtMostTo(buf, 0, buf.size); if (n <= 0) break; sha.update(buf, 0, n) }
        }
        return Sha256.toHex(sha.digest())
    }

    companion object {
        fun normalizeBase(url: String): String = url.trim().removeSuffix("/").removeSuffix("/cartridges/index.json").removeSuffix("/cartridges")
        fun indexUrl(baseUrl: String): String = "${normalizeBase(baseUrl)}/cartridges/index.json"
        fun fileUrl(baseUrl: String, id: String, path: String): String = "${normalizeBase(baseUrl)}/cartridges/$id/$path"
    }
}
