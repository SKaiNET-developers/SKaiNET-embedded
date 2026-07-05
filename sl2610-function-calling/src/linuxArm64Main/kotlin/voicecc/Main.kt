package voicecc

import sk.ainet.transformers.gemma.iree.IreeRuntime

fun main(args: Array<String>) {
    // `voicecc iree-smoke [dir]` — board-side IREE runtime binding smoke test:
    // load a vmfb + .irpa via the on-board iree-run-module and read the result.
    if (args.isNotEmpty() && args[0] == "iree-smoke") {
        val dir = args.getOrElse(1) { "/home/root/ireetest" }
        val rt = IreeRuntime()
        val r = rt.invoke(
            module = "$dir/tiny.vmfb",
            function = "f",
            inputs = listOf("4xf32=1,2,3,4"),
            parameters = mapOf("model" to "$dir/w.irpa"),
        )
        println("[iree-smoke] exit=${r.exitCode} ok=${r.ok}")
        println(r.stdout.trim())
        println("[iree-smoke] parsed=${IreeRuntime.parseFloatResults(r.stdout).map { it.toList() }}")
        return
    }

    // `voicecc gemma [dir]` — run the REAL FunctionGemma vmfb (+1.74GB .irpa,
    // mmap'd) on the board and print the last-position argmax token. Verifies
    // the full DSL->StableHLO->IREE model executes on-device via the binding.
    if (args.isNotEmpty() && args[0] == "gemma") {
        val dir = args.getOrElse(1) { "/home/root/ireetest" }
        // gemma-argmax.vmfb has an in-graph last-position slice + argmax tail, so
        // it returns a single token id (tensor<1xi32>) — a tiny output the board
        // can emit (the full [1,4,262153] logits OOM the old iree-run-module's
        // result formatter). The .irpa is mmap'd (parameterMode=file).
        // f16 weights: .irpa 831MB, ~905MB RSS — fits the 1.9GB board (FP32's
        // 1.74GB OOM'd). In-graph last-position argmax -> single token id.
        val r = IreeRuntime().invoke(
            module = "$dir/gemma-f16.vmfb",
            function = "gemma",
            inputs = listOf("1x4xi32=2,14070,563,506"),
            parameters = mapOf("model" to "$dir/gemma-f16.irpa"),
            parameterMode = "file",
        )
        println("[gemma] exit=${r.exitCode} ok=${r.ok}")
        // result line: "1xi32=3678"
        val tok = r.stdout.lineSequence()
            .firstOrNull { it.substringBefore('=').endsWith("xi32") }
            ?.substringAfter('=')?.trim()?.toIntOrNull()
        if (tok != null) println("[gemma] next-token argmax=$tok  (llama ref=3678)")
        else println("[gemma] no token (FP32 weights OOM the 1.9GB board — needs quantized .irpa)\n${r.stdout.trim().take(300)}")
        return
    }
    // `voicecc gen [dir]` — ON-DEVICE GREEDY DECODE LOOP. Re-runs the seq=24
    // prefill vmfb (per-position argmax, f16 weights) with the token sequence
    // growing each step (causal masking makes padding to 24 safe), reading the
    // argmax at the last real position. Generates the full <tool_N>(args) tool
    // call on the board. Prompt = "turn the light on" (Octopus v2 turn tokens).
    if (args.isNotEmpty() && args[0] == "gen") {
        val dir = args.getOrElse(1) { "/home/root/ireetest" }
        val s = 24
        val eot = 106
        val prompt = intArrayOf(2, 105, 2364, 107, 887, 506, 2214, 580, 106, 107, 105, 4368, 107)
        val rt = IreeRuntime()
        val toks = IntArray(s) { if (it < prompt.size) prompt[it] else 0 }
        val gen = mutableListOf<Int>()
        var step = 0
        while (prompt.size + step < s) {
            val inp = "1x24xi32=" + toks.joinToString(",")
            val r = rt.invoke("$dir/gemma-gen.vmfb", "gemma", listOf(inp), mapOf("model" to "$dir/gemma-gen.irpa"), "file")
            val arr = IreeRuntime.parseIntResult(r.stdout)
            if (!r.ok || arr == null) { println("[gen] step $step FAILED exit=${r.exitCode}"); break }
            val next = arr[prompt.size - 1 + step]
            gen.add(next)
            println("[gen] step=$step next=$next")
            if (next == eot) break
            toks[prompt.size + step] = next
            step++
        }
        println("[gen] generated=$gen")
        println("[gen] llama-ref=[262146, 236769, 3255, 718, 498, 1373, 262152, 106]  (<tool_0>(state=\"on\")<end><end_of_turn>)")
        return
    }
    // `voicecc listen [mic|<wav>] [device]` — live mic -> Silero VAD -> per-utterance pipeline.
    if (args.isNotEmpty() && args[0] == "listen") {
        runListen(source = args.getOrElse(1) { "mic" }, device = args.getOrNull(2))
        return
    }

    // `voicecc pipeline <wav>` — full board pipeline: ASR(NPU) -> LLM(our vmfb,CPU) -> action.
    if (args.isNotEmpty() && args[0] == "pipeline") {
        runPipeline(args.getOrElse(1) { "/home/root/voicecc-kt/test.wav" })
        return
    }

    // `voicecc asr <wav>` — Moonshine ASR on the Torq NPU via the Kotlin binding.
    if (args.isNotEmpty() && args[0] == "asr") {
        val wav = args.getOrElse(1) { "/home/root/voicecc-kt/test.wav" }
        val text = voicecc.asr.MoonshineRunner().transcribe(wav)
        if (text != null) println("[asr] NPU transcript: \"$text\"")
        else println("[asr] transcription failed")
        return
    }
    runApp(args)
}
