rootProject.name = "coralnpu-codegen"

// For development: source dependency on SKaiNET
includeBuild("../../../SKaiNET") {
    dependencySubstitution {
        substitute(module("sk.ainet:skainet-lang-core")).using(project(":skainet-lang:skainet-lang-core"))
        substitute(module("sk.ainet:skainet-compile-dag")).using(project(":skainet-compile:skainet-compile-dag"))
        substitute(module("sk.ainet:skainet-compile-opt")).using(project(":skainet-compile:skainet-compile-opt"))
    }
}
