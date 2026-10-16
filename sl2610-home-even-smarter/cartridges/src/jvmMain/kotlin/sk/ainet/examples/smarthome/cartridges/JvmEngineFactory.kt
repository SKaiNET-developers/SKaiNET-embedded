package sk.ainet.examples.smarthome.cartridges

/** Host JVM: used by the tests only; real engines run on the board through the native IREE (Torq) binding. */
object JvmEngineFactory : EngineFactory by UnsupportedEngineFactory(
    platform = "jvm",
    reason = "no JVM runtime binding — real engines run natively on the board",
)
