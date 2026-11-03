package sk.ainet.examples.smarthome.cartridges

import kotlinx.io.files.Path

/** Linux host (x86_64): no IREE runtime here; the TUI runs on the built-in fakes. The board build has the real one. */
object LinuxEngineFactory : EngineFactory by UnsupportedEngineFactory(
    platform = "linux",
    reason = "no runtime on the host build — real engines run on the board (linuxArm64)",
) {
    /** Same surface as the board factory; unused here. */
    var cacheDir: Path = Path("/tmp/hes-cache")
}
