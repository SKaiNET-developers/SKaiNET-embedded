package voicecc.actions

actual fun readSystemStatus(): Map<String, Any?> {
    val rt = Runtime.getRuntime()
    val usedMb = (rt.totalMemory() - rt.freeMemory()) / (1024 * 1024)
    return mapOf("mem_used_mb" to usedMb, "cpus" to rt.availableProcessors())
}
