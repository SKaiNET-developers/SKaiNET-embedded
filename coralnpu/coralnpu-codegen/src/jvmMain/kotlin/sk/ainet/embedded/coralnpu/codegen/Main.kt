package sk.ainet.embedded.coralnpu.codegen

import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import java.io.File

/**
 * CLI entry point for the Coral NPU C code generator.
 *
 * Usage:
 *   java -jar coralnpu-codegen.jar --graph <model.json> [-o <output.cc>] [--name <funcName>]
 *
 * Accepts a serialized ComputeGraph in JSON format (from skainet-compile-json)
 * and outputs C source code to stdout or a file.
 */
fun main(args: Array<String>) {
    val parsedArgs = parseArgs(args)

    val graphPath = parsedArgs["graph"]
    val outputPath = parsedArgs["o"]
    val modelName = parsedArgs["name"] ?: "main"

    if (graphPath == null) {
        System.err.println("Usage: coralnpu-codegen --graph <model.json> [-o <output.cc>] [--name <funcName>]")
        System.err.println()
        System.err.println("Options:")
        System.err.println("  --graph <path>  Path to ComputeGraph JSON file (required)")
        System.err.println("  -o <path>       Output C file path (default: stdout)")
        System.err.println("  --name <name>   Model/function name for comments (default: main)")
        kotlin.system.exitProcess(1)
    }

    val graphFile = File(graphPath)
    if (!graphFile.exists()) {
        System.err.println("Error: graph file not found: $graphPath")
        kotlin.system.exitProcess(1)
    }

    val jsonContent = graphFile.readText()
    val graph = deserializeGraph(jsonContent)

    val generator = CoralNpuCodeGenerator()
    val cCode = generator.generate(graph, modelName)

    if (outputPath != null) {
        File(outputPath).writeText(cCode)
        System.err.println("Written to $outputPath")
    } else {
        print(cCode)
    }
}

private fun parseArgs(args: Array<String>): Map<String, String?> {
    val result = mutableMapOf<String, String?>()
    var i = 0
    while (i < args.size) {
        when {
            args[i] == "--graph" && i + 1 < args.size -> {
                result["graph"] = args[i + 1]; i += 2
            }
            args[i] == "-o" && i + 1 < args.size -> {
                result["o"] = args[i + 1]; i += 2
            }
            args[i] == "--name" && i + 1 < args.size -> {
                result["name"] = args[i + 1]; i += 2
            }
            else -> i++
        }
    }
    return result
}

/**
 * Minimal JSON deserialization for ComputeGraph.
 *
 * This is a simple parser for the skainet-compile-json format.
 * In production, this would use kotlinx.serialization with the
 * skainet-compile-json module's serializers.
 *
 * TODO: Replace with proper skainet-compile-json dependency when available.
 */
@Suppress("UNCHECKED_CAST")
private fun deserializeGraph(json: String): DefaultComputeGraph {
    // Minimal JSON parsing — in production use kotlinx.serialization
    // For now, this is a placeholder that demonstrates the interface.
    // The real implementation would use:
    //   ComputeGraphSerializer.deserialize(json)
    // from the skainet-compile-json module.
    throw NotImplementedError(
        "JSON deserialization requires skainet-compile-json dependency. " +
        "For now, build ComputeGraph programmatically or add the JSON module."
    )
}
