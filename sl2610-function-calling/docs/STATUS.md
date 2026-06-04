# sl2610-voice-cc-kt — status, know‑how & handoff

Self‑contained record so this work can continue from any repo/machine. Last
updated 2026‑06‑02. (The fullest running log also lives in the Claude memory at
`~/.claude/projects/-home-miso-projects-coral/memory/` — machine‑local, NOT in
git; this doc is the portable copy.)

## Goal
Re‑implement the Python `sl2610-voice-cc` voice command‑and‑control app as a
**native Kotlin Multiplatform** app on **SKaiNET** + **SKaiNET‑transformers**,
where heavy models go **DSL → StableHLO → IREE** (Torq for the NPU, llvm‑cpu
aarch64+NEON for CPU), deployed to the **Astra Machina SL2610** board.

Hardware mapping (matches the sample): **Moonshine ASR → Torq NPU**;
**FunctionGemma‑270M LLM → CPU**.

## Workspace (multi‑repo, no top‑level git)
- `sl2610-voice-cc-kt/` — THIS app (KMP: `jvm` host + `linuxArm64` board).
- `SKaiNET/` (branch `feature/397-coral-npu-docs`) — NN DSL → StableHLO (`skainet-compile-hlo`), CPU backend, `.irpa` writer (`skainet-io-iree-params`).
- `SKaiNET-transformers/` (branch `develop`) — Gemma/Llama runtimes, GGUF loader, `GGUFTokenizer`. Built via composite `-PuseLocalSkainet=true`.
- `sl2610-voice-cc/` — the Python reference app (Octopus‑v2 prompt, compact codec, Moonshine runner).
- `skainet-whisper/` — UNRELATED sibling (a Whisper STT app for an Amlogic box). Not a dependency. See its `moonshine.md`.
- `build-mlir/` — scratch: vmfbs, .irpa, mlir, the `torqpkg`/`abenv`/`piperpkg` venvs.

## Module layout (this app — Gradle multi‑module)
Split so the LLM is reusable in other KMP apps (not welded to this app's wiring):
- **`:runtime`** (KMP jvm+linuxArm64) — `voicecc.runtime.IreeRuntime`: the generic
  IREE vmfb runner (subprocesses the board's `iree-run-module`). No SKaiNET deps →
  reusable for ANY vmfb (LLM, ASR, VAD).
- **`:llm`** (KMP) — the reusable FunctionGemma module:
  - commonMain: `CompactCodec` + `ToolCall` (pure Kotlin, no deps). `ToolCall` is the
    module's OWN type — deliberately NOT the app's `actions.Intent`, so `:llm` doesn't
    depend on the app. The consumer maps `ToolCall` → its own action type.
  - linuxArm64Main: `GemmaDecoder` — give it a vmfb + `.irpa` + gguf; `generate(text)`
    applies the Octopus‑v2 template, runs the greedy decode loop (incl. the free‑
    tokenizer‑during‑gen RAM trick) via `:runtime`, returns tool‑call text + `ToolCall`s.
- **`:asr`** (KMP jvm+linuxArm64) — `voicecc.asr.MoonshineRunner`: Moonshine ASR on the
  Torq NPU. `transcribe(wav)` drives the prebuilt Synaptics vmfbs via the version‑matched
  `torq.runtime` subprocess (`scripts/moonshine_npu.py`). Pure platform/posix, no SKaiNET
  deps → reusable.
- **`:vad`** (KMP jvm+linuxArm64) — `voicecc.vad.VadSegmenter`: Silero speech segmenter.
  `segments(source, device, onSegment)` streams one wav per utterance from the light
  capture+VAD helper (`scripts/vad_capture.py`); knows nothing about ASR/LLM (the caller
  decides per‑segment action). Pure platform/posix, no SKaiNET deps → reusable.
- **root (`:`)** (the app) — `App`/`Main`/`Pipeline`, `actions/` (ActionRouter+Intent);
  depends on `:llm` + `:asr` + `:vad` + `:runtime`. `Pipeline` is pure wiring:
  `runPipeline` = `MoonshineRunner` → `GemmaDecoder` → map `ToolCall`→`Intent` →
  ActionRouter; `runListen` = `VadSegmenter` → `runPipeline` per utterance. jvmMain keeps
  the host export tooling (`export/`, DAG→StableHLO tasks).
- Reuse a module elsewhere: `include(":llm")` (+`:runtime`), `implementation(project(":llm"))`,
  then `GemmaDecoder(vmfb, irpa, gguf).generate(text).calls`; likewise `:asr` →
  `MoonshineRunner().transcribe(wav)`, `:vad` → `VadSegmenter().segments(src, dev) { wav -> … }`.

## Board (Astra Machina SL2610 `rdk`)
- aarch64 Linux, **1.9 GB RAM** (tight!), 20 GB /home. adb+ssh at `192.168.3.26` (per‑boot DHCP). BusyBox (`head -n`, no `timeout`).
- `/dev/torq` NPU; `iree-run-module` has `local-task`(CPU) + `torq` HAL.
- Sample venv `/home/root/sl2610-voice-cc/.venv` (py3.12) has the **version‑matched** `torq.runtime`, `silero_vad_notorch`, `sounddevice`, onnxruntime.
- Models: `/home/root/sl2610-examples/models/...` — FunctionGemma gguf + `Synaptics/moonshine-tiny-bf16-torq/{encoder,decoder,decoder_with_past}.vmfb,tokenizer.json,preprocessor.onnx`.
- Our artifacts pushed to `/home/root/ireetest/` (vmfbs+irpa) and `/home/root/voicecc-kt/` (binary, scripts, test wavs).

## STATUS — what works ✅
- **Phase 1 (LLM swap) DONE.** FunctionGemma‑270M via SKaiNET DSL→StableHLO→IREE:
  - **Numeric parity** vs llama.cpp: vmfb == llama 4/4 argmax, cos 0.999; eager==vmfb bit‑exact (the IREE lowering is exact).
  - **Behavioral parity**: byte‑identical `<tool_N>(args)` tool calls on the demo prompts (host A/B).
  - Full gemma `jvmTest` suite green (75 tests).
- **Phase 2 (bridge) DONE, CPU + NPU.** Our own vmfbs run on the board CPU; the SKaiNET→Torq→NPU path is proven (bf16 matmul/conv → correct on `/dev/torq`).
- **Phase 3 (ASR) DONE.** Moonshine on the Torq NPU + live mic + Silero VAD, driven from Kotlin.
- **Phase 4 (end‑to‑end) WORKS on board.** `voicecc {pipeline,listen} <wav>`:
  `wav/mic → Silero VAD → Moonshine ASR (NPU) → FunctionGemma (OUR f16 vmfb, CPU) → CompactCodec → ActionRouter`.
  Verified: "Turn the light on." → `<tool_0>(state="on")` → **set_lights(state=on) [ok]**, EXIT 0, one Kotlin/Native binary.

## Runtime consumption — TWO paths (important; not all "Python‑wrapped")
The Python wrapping is ONLY for the prebuilt vendor models. Our SKaiNET‑built
artifacts are consumed natively.

| piece | created with | run on board via | Python? |
|---|---|---|---|
| FunctionGemma (LLM) | **SKaiNET** (DSL→StableHLO→IREE → our vmfb + .irpa) | Kotlin/Native → **native `iree-run-module`** subprocess | **no** |
| Moonshine (ASR) | Synaptics **prebuilt** vmfbs (our recompile crashes the torq compiler) | Kotlin/Native → **venv `torq.runtime`** subprocess | yes |
| Silero VAD | prebuilt onnx | Kotlin/Native → **venv `silero_vad_notorch`** | yes |

- The SKaiNET‑created LLM vmfb/.irpa are NOT touched by Python — Kotlin runs them
  through the native `iree-run-module` C binary.
- Python is needed for the prebuilt Torq Moonshine vmfbs only because they require
  the *version‑matched* `torq.runtime` (the system `iree-run-module` rejects them
  on version grounds).
- Caveat: even the LLM path is a *subprocess*, not yet in‑process — it shells out
  to `iree-run-module`. The native‑binding work (libiree+Torq cinterop) would make
  BOTH paths in‑process and drop Python entirely (and let Kotlin run the prebuilt
  Moonshine vmfbs directly too).

## App commands (`voicecc <mode>` on the board)
- `gen` — greedy decode our gemma vmfb (hardcoded prompt).
- `asr <wav>` — Moonshine ASR on NPU.
- `pipeline <wav>` — wav → ASR → LLM → action.
- `listen [mic|<wav>] [device]` — Silero VAD → per‑utterance pipeline. `listen mic`
  auto‑selects the USB mic (C920, sounddevice idx 4) — the ALSA "default" is the
  klamath i2s card with no mic attached (silent); pass an explicit index to override
  (recommend `listen mic 4`). **Memory‑safe on the 1.9GB board**: mic mode captures ONE
  utterance, the VAD helper then EXITS (frees ~250MB Silero + the mic) so the 900MB+ LLM
  gen runs with the board to itself, then re‑listens (loop until Ctrl‑C). Without this the
  resident VAD + gen OOM‑kill the gen subprocess (empty `[2/4 llm]`). The helper also
  orphan‑exits (getppid==1) so interrupted runs don't leak a mic‑holding python. The
  listener is crash‑proof: ASR failures + over‑long transcripts (>20 prompt tokens, the
  seq=24 vmfb budget) are skipped, not fatal. Verified live: C920 capture → VAD segments →
  Moonshine ASR(NPU) → loop, all robust. LIMITATION: commands must be SHORT (≤~20 prompt
  tokens) and acoustically CLEAN/close‑mic — Moonshine produces fluent‑but‑wrong text on
  noisy/far‑field audio. Clean wav (`pipeline`/`listen cmd.wav`) → set_lights[ok] proven.
- `iree-smoke` — tiny vmfb+.irpa on CPU (binding sanity).

## Build / deploy recipes
- Board binary: `./gradlew linkReleaseExecutableLinuxArm64` → `BOARD=root@<ip> bash scripts/deploy.sh` (ssh `cat` stream; handles libcrypt.so.1 compat).
- Run gemma tests: `./gradlew :llm-inference:gemma:jvmTest -PuseLocalSkainet=true -PgemmaTestMaxHeap=22g [-PseqLen=24]` (in SKaiNET‑transformers).
- LLM vmfb pipeline (host): bake (RealGemmaBakeIrpaTest) → `add_argmax_perpos.py` (seq=24 per‑pos argmax) → `make_f16.py` (f16 weights+MLIR, halves .irpa) → compile. Weights via safetensors → `iree-convert-parameters` → .irpa (NOT SKaiNET's IrpaWriter — its header is IREE‑incompatible).
- CPU vmfb: `iree-compile --iree-input-type=stablehlo --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-triple=aarch64-unknown-linux-gnu --iree-llvmcpu-target-cpu-features=+neon`.
- **Torq NPU vmfb**: `scripts/iree-compile-torq.sh` — uses the **torq_compiler g165e12a wheel** (matches board; stock IREE 3.11 emits a `[Ch]` bytecode feature the board rejects). KEY FIX: a Torq vmfb embeds aarch64 host code linked via a PATH `ld`; shim `ld`→ bundled `iree-lld -flavor gnu` first on PATH (else `Relocations in generic ELF (EM:183)`). Flags `--iree-hal-target-device=torq --torq-hw=SL2610`.
- Toolchains (host): `build-mlir/torqpkg` (torq compiler, py3.12 `--target` install), docker `coral-iree-toolchain` (stock IREE 3.11), `build-mlir/abenv` (llama-cpp-python, gguf, onnx for A/B), `build-mlir/piperpkg` (Piper TTS for command wavs).

## The bugs we fixed (the hard‑won know‑how)
SKaiNET‑transformers gemma (committed on `develop`):
1. `Gemma4WeightLoader` missed the `gemma3` arch prefix → blockCount 34 → demanded blk.18. Added "gemma3".
2. `kvSharedLayers` defaulted 20 (a gemma3n/4 value) → firstSharedLayer=‑2 crash. Plain gemma3 = 0.
3. **KV cache during tracing**: `KVCache.update()` uses non‑traceable `copyToFloatArray` → froze 36 zero K/V leaves. Strip the cache before tracing (prefill needs none).
4. **Attention scale** hardcoded 1.0 (a Gemma‑4 claim) → over‑sharpened softmax. gemma3 = 1/√head_dim. → `attentionScale=null`.
5. **qk‑norm double‑offset**: gemma3 gguf bakes (1+w) into q/k norm; `qkNormUnitOffset=true` added +1 again (proven: q/k‑norm weights mean ~1.84). → `false`.
6. **Spurious v_norm**: gemma3 has no v_norm; `vNormNoScale` hardcoded true. → `false`.
SKaiNET (compile‑hlo): boxing‑free FloatArray weight externalization (finalize stored `.toList()` → 2.7GB OOM on the embedding); `AttentionOperationsConverter` now consumes the explicit SDPA mask operand (was a TODO → sliding layers exported UNMASKED).
Signature/diagnostic finds: RoPE SPLIT_HALF == llama NEOX (bit‑exact); Q5_1/Q6_K/Q8_0 dequant bit‑exact vs gguf; the export emits spurious leaf outputs (the multi‑output trace artifact) — the real logits are the LAST output.

## Board‑specific gotchas
- **Version‑matched runtime**: prebuilt Synaptics NPU vmfbs (Moonshine, yolo…) are STALE on the updated board (bytecode 15 vs 16; torq exe v0 vs v1). System `iree-run-module --device=torq` REJECTS them; the venv `torq.runtime` (matched) RUNS them. → reuse prebuilt via the matched runtime, OR recompile with the g165e12a wheel.
- **RAM (1.9 GB)**: FP32 gemma .irpa (1.74 GB) OOMs → use **f16** (831 MB, ~905 MB RSS). In the pipeline, FREE the `GGUFTokenizer` (vocab 262153) during the gen subprocess + `GC.collect()`, reload to detokenize. Run heavy models SEQUENTIALLY.
- **Output OOM**: old board `iree-run-module` OOMs formatting a 262153‑wide result and ignores `--output=@file` → emit a SMALL output (in‑graph argmax → token id).

## KNOWN LIMITATIONS / not‑cool bits (fix these)
- **Moonshine/ASR + VAD wrap Python** (`vad_capture.py`, `moonshine_npu.py` via the venv `torq.runtime`/`silero`, subprocessed by Kotlin) — see "Runtime consumption — two paths". The LLM is already Python‑free (native `iree-run-module`); only the prebuilt vendor models are Python‑wrapped. Still interim/ugly. PROPER: pure‑Kotlin/Native **cinterop to the matched `libiree` (with the torq HAL)** + the IREE runtime C API — load module once, set inputs, invoke — dropping Python entirely. The matched runtime is currently a CPython extension (`_runtime.cpython‑312…so`); need the underlying libiree of that version for a clean C‑API. (skainet‑whisper's `libiree_whisper.so` JNI is a model for the binding.) Same for ALSA mic capture (drop `sounddevice`/arecord → libasound cinterop).
- **LLM speed ~7 s/token** (re‑run‑prefill, no KV‑cache, f16→f32 convert each step, bandwidth‑bound). `lm_head`‑last‑position only saves ~16%. Real fixes: (a) **KV‑cache decode** — needs refactoring SKaiNET MHA so the cache is forward I/O (it's internal mutable state today) + dynamic‑position RoPE; (b) **Q5_K dequant‑in‑graph** (less bandwidth, like llama.cpp). Both are framework features. For a real‑time shipping app the pragmatic option is **Hybrid**: run the LLM on llama.cpp (CPU) like the sample, keep our IREE LLM as the pure‑SKaiNET/NPU path.
- **Moonshine torq recompile crashes** the g165e12a compiler (`getWeightMemoryFormat: Invalid weight conversion fp32→bf16`, then segfault) even though the MLIR runs on CPU. bf16 conv+matmul compile individually → it's the full graph. → reuse prebuilt vmfbs, or get Synaptics' onnx→torq recipe / a fixed compiler. (`scripts/moonshine_get_mlir.py` gets the static encoder MLIR.)

## Next steps
1. Native `libiree`+torq cinterop binding (kill the Python wrappers) — the biggest "make it real" item.
2. LLM speed: KV‑cache decode export (MHA cache→I/O) and/or Q5_K dequant‑in‑graph; or Hybrid llama.cpp for shipping.
3. Live mic test on the board (`voicecc listen mic`, pass `--device` for USB mic).
4. More command wavs / real ActionRouter handlers (GPIO/LED/MQTT).

## Pointers
- Scripts: `scripts/` (deploy, iree‑compile‑{cpu,torq}, make_f16, add_argmax_{tail,perpos}, add_lmhead_lastpos, moonshine_get_mlir, moonshine_npu, vad_capture, make_command_wav).
- Reproducible artifacts in `build-mlir/` (host): gemma‑gen‑f16 vmfb + irpa, etc.
- Commit trail tells the story: `git log --oneline` here and in SKaiNET / SKaiNET‑transformers.
