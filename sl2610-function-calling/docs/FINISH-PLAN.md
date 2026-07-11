# FINISH PLAN — full "SKaiNET" port of the Synaptics FunctionGemma demo

> **Single source of truth for finishing the project.** `PLAN.md` (at the workspace root) remains the
> detailed, historical phase tracker (what was built and how). This file is the forward-looking plan to
> take the port from "works on my machine, across 5 repos" to **"clone → connect board → run
> demo → finetune on my own commands."**
> Authored 2026-07-11. Companion working copy: `~/.claude/plans/you-are-senior-ml-velvet-candy.md`.

## Goal (the user's north star)

```
git clone <the port>  →  ./bootstrap.sh  →  BOARD=root@<ip> ./gradlew deployBoard  →  speak a command
                                                                                      →  then finetune FunctionGemma on MY commands
```

Everything on the board runs on our own self-compiled vmfbs (no vendor Moonshine binaries), zero
runtime Python, at KV-cache-usable latency — and a documented recipe lets the user retrain
FunctionGemma to understand new commands and re-bake into the demo.

## Where we are (verified 2026-07-11)

The **hard ML/compiler risk is retired.** FunctionGemma self-compiles from the DSL (`compile-gemma.sh`,
Python-free) and is board-verified on CPU; the Moonshine **encoder** is DSL-authored and self-compiled;
the Moonshine **decoder** is DSL-authored and CPU-proven. Engine + models are at a matched **0.35.0**;
IREE conformance is green. What's left is **productization, performance verification, and extensibility
docs** — not research. Full maturity table in `PLAN.md`'s current-state section and in the companion plan.

| Gap | Impact | Phase |
|---|---|---|
| 5 dirty repos, private Torq wheel + hardcoded paths/IP | Not clonable by anyone else | P1 |
| Default ASR still rides vendor vmfbs; VAD still Python | Not "fully SKaiNET" | P3 |
| Gemma re-decode ~6 s/token; KV graphs board-unverified | Not demo-usable | P4 |
| Torq compiler pinned to g165e12a dev snapshot | "Latest" claim not re-validated/documented | P2 |
| `PLAN.md` carries two overlapping phase generations | Docs drift | P0 |
| No documented finetuning path | User's downstream goal unmet | P5 |

## Definition of done (all four — locked with the user 2026-07-11)

1. **Reproducible clone-to-run** — fresh clone + bootstrap + `deployBoard`, no manual path/IP surgery.
2. **Finetuning recipe** — scripted path to retrain FunctionGemma on custom commands and re-bake.
3. **Fully self-compiled, zero Python** — our encoder+decoder as the board default; VAD in Kotlin.
4. **Usable performance** — Gemma (and Moonshine) KV-cache decode board-verified.

## Phases

> **Paths:** this plan lives in the demo's `docs/`. Paths below are relative to the demo root
> `SKaiNET-embedded/sl2610-function-calling/` — so `docs/TOOLCHAIN-PIN.md`, `docs/FINETUNING.md`,
> `docs/BOARD-RUNBOOK.md`, `docs/PERF-LOGBOOK.md` are this file's siblings, and `bootstrap.sh`,
> `demo.env`, `examples/custom-command/` sit one level up (the demo root). `PLAN.md` and
> `docs/torq-debug-notes/…` are loose at the workspace root (`coraldevboard/`), above the repo.

### P0 — Freeze & reconcile *(this session)*
- Reconcile `PLAN.md`: retire the superseded `3.1–3.5` block (kept as a one-line pointer to
  `3SC`/`3DEC`), refresh the stale 2026-07-02 baseline, add a 2026-07-11 status banner and a link here.
- Commit the in-flight work on the `SKaiNET-transformers` / `SKaiNET-embedded` feature branches so the
  baseline is reproducible before restructuring. *(left to the user — see `docs/BOARD-RUNBOOK.md`)*.

### P1 — Reproducible clone-to-run → *done-when #1*
Packaging = **one consolidated `skainet-fc-demo` repo** (chosen with the user), engine + transformers
consumed as published **0.35.0** Maven artifacts. Delivered incrementally on the existing demo dir
(`SKaiNET-embedded/sl2610-function-calling/`, the natural root) so the physical repo extraction is the
last, mechanical step:
- `demo.env.example` + `demo.env` (git-ignored) — the single config surface: `BOARD`, `TORQ_PKG`,
  `GEMMA_GGUF`, board dest dirs. Every script sources it instead of hardcoding.
- `bootstrap.sh` — one setup entry: resolve the Torq toolchain into `.toolchain/`, generate `demo.env`,
  point at the FunctionGemma GGUF + Moonshine weights.
- Config-externalize `scripts/{deploy.sh,compile-gemma.sh,iree-compile-torq*.sh,moonshine_compile_preprocessor.sh}`.
- README front door: the three-command story.
- **Open item (documented, not hidden):** the `kgemma` FunctionGemma exporter is a Gradle task in
  `SKaiNET-transformers`, not a published CLI — a *standalone* clone must either fetch that repo or the
  exporter must ship as a runnable artifact. Tracked in `docs/BOARD-RUNBOOK.md` → "extract to standalone repo."

### P2 — Toolchain currency: re-validate & decide → *honors "latest libs/tools"*
Policy chosen with the user: **re-validate the newest Torq release; adopt if it passes the canary gate,
else keep g165e12a and document the pin.** Board OS SDK is already current (`scarthgap_6.12_v2.4.0`).
- Decision + procedure captured in `docs/TOOLCHAIN-PIN.md` (supersedes the raw audit in
  the workspace-root `docs/torq-debug-notes/torq-toolchain-update.md`).
- Canary gate = compile encoder + Gemma with the candidate, deploy, check the silent-zeros probe + the
  token oracle `[262146,236769,3255,718,498,1373,262152,106]`. Needs board + the private wheel.

### P3 — Fully self-compiled, zero Python → *done-when #3*
- Flip `MoonshineRunner` defaults to our self-compiled encoder (`MOONSHINE_ENCODER_VMFB`) + decoder
  (`MOONSHINE_DECODER_VMFB`); demote the vendor `modelDir` fallback to explicit opt-in.
- Reimplement Silero VAD in Kotlin (DSL or ported gate logic) to kill `scripts/vad_capture.py` — the last
  runtime Python. Retire bring-up stubs `HloBridge.kt` / `SdpaTrace.kt` (PLAN 2.4).
- Steps + verification in `docs/BOARD-RUNBOOK.md` (needs board).

### P4 — Usable performance via KV-cache → *done-when #4*
- Board-verify the Gemma KV 2-graph loop (`GEMMA_KV=1` → `gemma-prefill.vmfb` + `gemma-with-past.vmfb`,
  driver `GemmaKvDecoder`); confirm the flagged unknowns (arg order, K-vs-V output order, entry-fn names).
- Board-verify the Moonshine KV decode (`MOONSHINE_KV=1`, `MoonshineKvDecoder`).
- Record before/after in `docs/PERF-LOGBOOK.md`. Needs board time — steps in `docs/BOARD-RUNBOOK.md`.

### P5 — Finetuning recipe (custom commands) → *done-when #2, the downstream goal*
- `docs/FINETUNING.md`: define new tools → dataset in the Octopus-v2 prompt format → LoRA train (offline
  Python OK — it's training, not the runtime) → merge → GGUF → `GEMMA_GGUF=… compile-gemma.sh board` →
  `deployBoard` → verify. The finetuning story is literally **"swap the GGUF"** — it reuses the whole
  existing self-compile path unchanged.
- `examples/custom-command/`: a minimal worked example (one new tool + a tiny dataset template) to fork.

## Sequencing
`P0 → P1 → P3 → P4 → P5` (P5 can start in parallel after P1); `P2` slots in whenever the board is free.
Biggest residual risk: the KV-cache K-vs-V / entry-fn unknowns (P4) — the only items needing live board
debugging. Everything else is packaging/wiring/docs over already-proven components.

## Verification (end-to-end, per phase)
- **P1:** clone into a fresh temp dir → `./bootstrap.sh` → `./gradlew :jvmRun` with no edits to any path.
- **P2:** candidate toolchain passes the canary + token oracle; decision written to `docs/TOOLCHAIN-PIN.md`.
- **P3:** board run with vendor `modelDir` unset and no Python process (verify `ps` on-board during `listen`).
- **P4:** `VOICECC_PROFILE=1 voicecc gen "turn the light on"` shows KV latency well under re-decode baseline;
  token stream matches the oracle.
- **P5:** run `examples/custom-command/` end-to-end → new spoken command routes to the new tool.
