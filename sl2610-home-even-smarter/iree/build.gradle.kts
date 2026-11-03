// In-process IREE runtime for the board: one native process drives the CPU (local-task) and the Torq NPU (torq)
// through native/iree-torq-runtime (the Torq-fork runtime as a static library plus the hes_iree C shim).
// linuxArm64 only: the runtime archive is built for the board. Build it first: native/iree-torq-runtime/build-dist.sh
plugins {
    alias(libs.plugins.kotlinMultiplatform)
}

// -PireeDist=native/iree-torq-runtime/dist-221 links another runtime build (e.g. the next Torq version) side by side
val dist = rootProject.layout.projectDirectory.dir(providers.gradleProperty("ireeDist").getOrElse("native/iree-torq-runtime/dist"))

kotlin {
    linuxArm64 {
        compilations.getByName("main").cinterops.create("hes_iree") {
            definitionFile.set(project.file("src/nativeInterop/cinterop/hes_iree.def"))
            includeDirs(dist.dir("include"))
            extraOpts("-libraryPath", dist.dir("lib").asFile.absolutePath)
        }
        // Board smoke test: CPU and NPU graphs in one process (iree-smoke <cpu.vmfb> <torq.vmfb>).
        binaries.executable("ireeSmoke") {
            baseName = "iree-smoke"
            entryPoint = "sk.ainet.examples.smarthome.iree.smoke.main"
        }
    }
}

tasks.matching { it.name.startsWith("cinteropHes_iree") }.configureEach {
    val runtimeArchive = dist.file("lib/libiree_torq_runtime.a").asFile
    // The archives are embedded into the cinterop klib: rebuild it when build-dist.sh changes them.
    inputs.files(dist.file("lib/libhes_iree.a"), dist.file("lib/libiree_torq_runtime.a")).optional()
    doFirst {
        check(runtimeArchive.exists()) { "missing $runtimeArchive — run native/iree-torq-runtime/build-dist.sh first" }
    }
}
