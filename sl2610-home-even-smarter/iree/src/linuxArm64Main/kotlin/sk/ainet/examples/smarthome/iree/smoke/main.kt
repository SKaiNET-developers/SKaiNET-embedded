package sk.ainet.examples.smarthome.iree.smoke

import sk.ainet.examples.smarthome.iree.HostTensor
import sk.ainet.examples.smarthome.iree.IreeRuntime
import kotlin.system.exitProcess
import kotlin.time.measureTimedValue

/**
 * Board smoke test for the in-process runtime: one IreeRuntime, a CPU session and an NPU session, called
 * alternately. Graphs (native/iree-torq-runtime/mlir, compiled with the Torq compiler):
 *   cpu.vmfb   main(f32[4], f32[4]) -> (a + b) * a       on local-task
 *   torq.vmfb  main(bf16[1,64], bf16[1,64]) -> a + b     on torq
 * Usage: iree-smoke <cpu.vmfb> <torq.vmfb> [rounds]
 */
fun main(args: Array<String>) {
    if (args.size < 2) { println("usage: iree-smoke <cpu.vmfb> <torq.vmfb> [rounds]"); exitProcess(2) }
    val rounds = args.getOrNull(2)?.toInt() ?: 3
    var ok = true
    IreeRuntime().use { rt ->
        println("drivers: ${rt.drivers}")
        val (cpu, cpuOpen) = measureTimedValue { rt.session("local-task", args[0]) }
        val (npu, npuOpen) = measureTimedValue { rt.session("torq", args[1]) }
        println("open: local-task ${cpuOpen.inWholeMicroseconds / 1000.0} ms, torq ${npuOpen.inWholeMicroseconds / 1000.0} ms")
        cpu.use { npu.use {
            val a = floatArrayOf(1f, 2f, 3f, 4f)
            val b = floatArrayOf(10f, 20f, 30f, 40f)
            val x = FloatArray(64) { it * 0.5f }
            val y = FloatArray(64) { 1f }
            repeat(rounds) { r ->
                val (c, tc) = measureTimedValue { cpu.call("module.main", listOf(HostTensor.f32(longArrayOf(4), a), HostTensor.f32(longArrayOf(4), b))) }
                val cv = c.single().toFloats()
                val cOk = cv.contentEquals(floatArrayOf(11f, 44f, 99f, 176f))
                val (n, tn) = measureTimedValue { npu.call("module.main", listOf(HostTensor.bf16(longArrayOf(1, 64), x), HostTensor.bf16(longArrayOf(1, 64), y))) }
                val nv = n.single().toFloats()
                val nOk = nv.indices.all { kotlin.math.abs(nv[it] - (x[it] + 1f)) <= 0.25f } && nv.any { it != 0f }
                ok = ok && cOk && nOk
                println("round $r: cpu ${cv.toList()} ${if (cOk) "OK" else "WRONG"} (${tc.inWholeMicroseconds / 1000.0} ms) | " +
                    "npu [${nv.take(4).joinToString()} …] ${if (nOk) "OK" else "WRONG"} (${tn.inWholeMicroseconds / 1000.0} ms)")
            }
        } }
    }
    println(if (ok) "IREE SMOKE OK" else "IREE SMOKE FAILED")
    exitProcess(if (ok) 0 else 1)
}
