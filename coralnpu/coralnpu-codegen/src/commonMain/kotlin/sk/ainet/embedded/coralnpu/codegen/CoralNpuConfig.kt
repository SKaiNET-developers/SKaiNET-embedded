package sk.ainet.embedded.coralnpu.codegen

/**
 * Target configuration for Coral NPU C code generation.
 *
 * Encapsulates hardware-specific conventions: section attributes,
 * memory placement, data types, and code style.
 */
data class CoralNpuConfig(
    /** Section attribute for I/O arrays (placed in DTCM). */
    val dataSectionAttr: String = """__attribute__((section(".data")))""",

    /** C float type name. */
    val floatType: String = "float",

    /** Header comment lines. */
    val headerComment: (funcName: String) -> List<String> = { funcName ->
        listOf(
            "// Generated from StableHLO MLIR function @$funcName",
            "// f16 promoted to f32 (Coral NPU has hardware f32, no f16)"
        )
    }
) {
    companion object {
        val DEFAULT = CoralNpuConfig()
    }
}
