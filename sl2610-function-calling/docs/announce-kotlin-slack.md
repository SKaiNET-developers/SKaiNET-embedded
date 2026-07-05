# Slack / social blurbs — SL2610 voice demo

Short teasers for a Kotlin Slack / dev channel (the long-form article is
[kotlin-on-the-edge.md](kotlin-on-the-edge.md)). No lib pitch — just the story + a repo link.

## One-liner

> TIL you can write a whole edge-AI stack in Kotlin: an offline voice → LLM → action loop
> on a little NPU dev board, no Python. KMP means host↔board parity, the speech model's
> authored in a Kotlin DSL, the NPU's driven over cinterop. MIT if you want to poke → `<repo link>`

## A few lines with more texture

> 🎙️→💡 Been building a voice command-and-control demo that runs *entirely* on a low-power
> NPU board: you talk, it transcribes, a small function-calling LLM decides which tool to
> call, and it fires — offline, no cloud.
> The fun part for Kotlin folks: it's Kotlin all the way down. Same KMP code runs as a JVM
> reference on your laptop and cross-compiles to a native aarch64 board binary; the ASR model
> is written in a Kotlin DSL and compiled to the NPU; the runtime binding is plain cinterop.
> No Python in the hot path.
> Self-contained + MIT — `./gradlew jvmRun` works with nothing plugged in → `<repo link>`
