package voicecc.actions

// Phase 0: minimal. Phase 4 reads /proc (cpu/mem) + thermal_zone via posix.
actual fun readSystemStatus(): Map<String, Any?> = mapOf("status" to "unknown")
