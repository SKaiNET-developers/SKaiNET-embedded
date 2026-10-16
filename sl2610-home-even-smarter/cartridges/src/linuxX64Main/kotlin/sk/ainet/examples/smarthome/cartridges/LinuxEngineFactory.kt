package sk.ainet.examples.smarthome.cartridges

/** Linux host (x86_64): no IREE runtime here; the TUI runs on the built-in fakes. The board build has the real one. */
object LinuxEngineFactory : EngineFactory by UnsupportedEngineFactory(
    platform = "linux",
    reason = "no runtime on the host build — real engines run on the board (linuxArm64)",
)
