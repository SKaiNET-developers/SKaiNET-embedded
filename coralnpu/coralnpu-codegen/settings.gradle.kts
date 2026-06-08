rootProject.name = "coralnpu-codegen"

// For development: source dependency on SKaiNET
val localSkainet = file("../../../SKaiNET")
if (localSkainet.isDirectory) {
    includeBuild(localSkainet) {
        dependencySubstitution {
            substitute(module("sk.ainet.core:skainet-lang-core")).using(project(":skainet-lang:skainet-lang-core"))
            substitute(module("sk.ainet.core:skainet-compile-dag")).using(project(":skainet-compile:skainet-compile-dag"))
            substitute(module("sk.ainet.core:skainet-compile-opt")).using(project(":skainet-compile:skainet-compile-opt"))
        }
    }
}
