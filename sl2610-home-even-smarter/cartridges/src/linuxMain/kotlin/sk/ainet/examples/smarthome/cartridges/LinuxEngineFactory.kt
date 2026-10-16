package sk.ainet.examples.smarthome.cartridges

/** Linux (board and host) runtime: the in-process IREE binding is being built; until then the TUI runs on fakes. */
object LinuxEngineFactory : EngineFactory by UnsupportedEngineFactory(
    platform = "linux",
    reason = "no native IREE runtime binding yet — run the terminal app with the built-in fake engines",
)
