package voicecc.llm

// Native-resolution probe: verifies the SKaiNET commonMain gemma3 graph + GGUF
// loader resolve and compile for BOTH jvm and linuxArm64 (i.e. the native
// klibs exist on Maven Central). Replaced by the real GemmaRunner next.
import sk.ainet.io.gguf.GGUFReader
import sk.ainet.models.gemma.Gemma4WeightLoader
import sk.ainet.models.gemma.GemmaNetworkLoader

@Suppress("unused")
internal val gemmaProbe = listOf(
    GemmaNetworkLoader::class,
    Gemma4WeightLoader::class,
    GGUFReader::class,
)
