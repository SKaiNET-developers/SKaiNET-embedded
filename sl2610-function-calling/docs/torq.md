# Torq NPU: Moonshine encoder `MatMulPattern.cpp:57` — problem statement & handoff

**Audience:** an AI/engineer picking up the task of compiling the SKaiNET-DSL Moonshine
encoder for the Synaptics Torq NPU (SL2610/SL2619). This document is self-contained:
it embeds the reproducer generators, the controlled experiments already run, the one
root cause already fixed, and the open one. Read it top to bottom.

---

## 0. TL;DR

- **Goal:** run our own-compiled Moonshine-tiny **encoder** on the Torq NPU, Python-free.
- **The recipe works.** A *hand-written* 6-layer encoder (`enc6.mlir`) with our tiling
  (seq-tiled attention + hidden-tiled FFN) **compiles and links on Torq v2.0.0**. Our IR
  is valid; it also compiles on `llvm-cpu`.
- **The blocker is a Torq compiler LAYOUT bug**, surfaced as
  `compiler/torq/Conversions/TorqHLToTorqHW/MatMulPattern.cpp:57:
  Assertion 'matA.dim(MatA::K) == matB.dim(MatB::K)' failed` — on the attention·V matmul,
  whose *shapes are byte-identical* to enc6's working matmul. It's physical-layout
  assignment, not shapes.
- **Root cause #1 — FOUND & FIXED:** K must be transposed to `[H,D,Sk]` **before** slicing
  head-groups, not sliced-then-transposed per group. Proven by a 4-way controlled experiment.
  Fix is in `TorqAttentionTilingPass.kt`.
- **Root cause #2 — SOLVED (2026-07-04).** The trigger is the **K transpose formulation**, not
  a diffuse "global" effect: the pass emitted the [H,D,Sk] K as a direct `transpose[0,2,1]` on
  the [H,Sk,D] source. That mis-seeds Torq's layout solver and trips `MatMulPattern.cpp:57` on
  the downstream AV matmul — even though the AV operands' local producer chains are byte-identical
  to a compiling graph. **Fix:** route the transpose THROUGH the [Sk,H,D] ordering —
  `[H,Sk,D] --[1,0,2]--> [Sk,H,D] --[1,2,0]--> [H,D,Sk]` — i.e. the final [H,D,Sk] transpose must
  be a `[1,2,0]` FROM a [Sk,H,D] tensor (exactly how the known-good enc6/layer1 build K), never a
  `[0,2,1]` on [H,Sk,D]. One formula covers RoPE and no-RoPE. Applied in `TorqAttentionTilingPass.kt`.
  Verified: the **full 6-layer DSL encoder now compiles on Torq v2.0.0** (and llvm-cpu), and a
  1-layer encoder **runs correctly on the SL2610 NPU**.
- **The full 6-layer encoder RUNS on the NPU (2026-07-04).** Compiled with stable v2.0.0 and run
  via the stable `torq.runtime.VMFBInferenceRunner` (the same API the vendor demo uses): loads and
  executes, all 6 layers, exit 0. **NOT a Synaptics runtime bug.** An earlier scare — the bare
  `torq-run-module` CLI segfaulting at ≥2 layers with `LRAM read IOCTL failed` /
  `torq_read_lram(): LRAM segment at addr 0x0 not found` — is a **bare-CLI artifact**: that CLI
  doesn't set up LRAM segments the way `VMFBInferenceRunner` does. Through the Python runtime API
  the multi-layer module loads fine. Toolchain lock-step still matters: vendor prebuilt vmfbs are
  `executable runtime version 0` (older compiler + the board's `2.0.0a1` alpha runtime); ours are
  `version 2` (stable v2.0.0 compiler + the stable v2.0.0 runtime, both validated on-board — the
  stable `torq-run-module` md5-matches the public release). Run stable-compiled vmfbs with the
  stable runtime, not the alpha, and vice-versa.
- **Minimal repro for Synaptics (compile #2, now fixed our side):** two semantically-identical
  tiled-attention blocks that differ ONLY in K transpose ordering; one compiles, one asserts.

---

## 1. Model + tiling parameters

Moonshine-tiny encoder (`MoonshineConfig`): `dim=288`, `encoderLayers=6`, `nHeads=8`,
`headDim=36`, `ffnDim=1152`, `vocabSize=32768`, `maxFrames=165` (the sequence length S).
Weights are **bf16-native** at the matmul (fp32 weights crash Torq earlier at
`Kernel.cpp:2602 getWeightMemoryFormat`; this is already solved — do not regress it).

Tiling that fits SL2610 LRAM (all empirically verified on v2.0.0):
- **Attention:** head-groups of ≤4 heads (8 heads overflow in one dispatch) **and**
  query-sequence tiling into ≤83 positions/chunk (2 chunks: 83 + 82). Each query chunk
  attends to the full K/V (flash-style). Shrinks the `[g,Sq,Sk]` scores intermediate.
- **FFN:** hidden-dimension tiling. The FFN weights (`[288,1152]`+`[1152,288]` ≈ 1.3 MB)
  dominate, so sequence tiling does NOT help — split the 1152 hidden dim into 4×288 column
  chunks of W1 / row chunks of W2, `up_c=A·W1c → gelu → p_c=gc·W2c`, and **sum** the 4
  partials. Never materialize the full `[165,1152]` intermediate.

---

## 2. Toolchain & how to reproduce a compile

Stable **Torq v2.0.0** (public: github.com/synaptics-torq/torq-compiler releases). Host x86
compiler unpacked at `…/scratchpad/torq2/release/` (ephemeral — re-download if gone):

- `iree-compile`, `iree-opt`, `iree-lld` in
  `release/python/compiler/iree/compiler/_mlir_libs/`; shared libs in `release/lib`.
- Cross-linking on x86 for aarch64 host code needs an `ld` shim → `iree-lld -flavor gnu`.

Compile wrapper (the `tc2.sh` used throughout):

```bash
W=…/scratchpad/torq2
MLIRLIBS=$W/release/python/compiler/iree/compiler/_mlir_libs
mkdir -p $W/shim
printf '#!/usr/bin/env bash\nexec "%s" -flavor gnu "$@"\n' "$MLIRLIBS/iree-lld" > $W/shim/ld
chmod +x $W/shim/ld
# tc2.sh:
export PATH="$W/shim:$PATH" LD_LIBRARY_PATH="$W/release/lib:$MLIRLIBS"
timeout 200 $MLIRLIBS/iree-compile \
  --iree-input-type=stablehlo --iree-hal-target-device=torq --torq-hw=SL2610 "$@"
```

Success = a `.vmfb` is produced. Failure prints the assertion (use
`--mlir-disable-threading --mlir-print-ir-before-all` to see the exact `torq_hl.matmul`
operands at the abort). CPU sanity: swap the two torq flags for
`--iree-hal-target-backends=llvm-cpu` — the DSL encoder compiles clean there (proves IR valid).

**Board:** aarch64 runtime staged at `/home/root/torq2` (non-destructive; the g165e12a venv
that runs FunctionGemma is untouched). Run a vmfb:
`/home/root/torq2/torq_libs/torq-run-module --device=torq --torq_hw_type=astra_machina …`.
Verified: a v2.0.0-compiled identity matmul returns correct `1x4xbf16=[1 2 3 4]` on the NPU.

**Generate the DSL encoder MLIR** (composite build against local `../SKaiNET`, no publish):

```bash
cd SKaiNET-transformers
# env vars reach the forked test JVM (system props -D do NOT):
ENC_LAYERS=1 ENC_2D=1 MOONSHINE_MLIR_OUT=/abs/out.mlir \
  ./gradlew :llm-inference:moonshine:jvmTest \
  --tests "sk.ainet.models.moonshine.MoonshineEncoderMlirDumpTest" \
  -PuseLocalSkainet=true --rerun-tasks -q
```
`ENC_LAYERS` (default 6), `ENC_2D=1` traces a 2-D `[S,D]` input (drops the batch-of-1 in
projections), `MOONSHINE_MLIR_OUT` picks the output path. The test registers
`TorqAttentionTilingPass` + `TorqFfnTilingPass` via `TargetOptimizers.registerDagPasses("torq")`
then dumps StableHLO.

---

## 3. The failure

```
iree-compile: compiler/torq/Conversions/TorqHLToTorqHW/MatMulPattern.cpp:57:
  LogicalResult mlir::syna::torq::OpPattern<torq_hl::MatMulOp>::transform(...):
  Assertion `matA.dim(MatA::K) == matB.dim(MatB::K)' failed.
```

Single-threaded dump shows the failing `torq_hl.matmul` is the **AV** (attention·value):
operands `(matA=scores[4,83,165], params[2xi32], out[4,83,36], matB=V[4,165,36])`, all in
`lram`. Dimensionally K=165 lines up on both — so the assertion firing means Torq assigned
one operand a **transposed physical layout**, so its logical K no longer holds. This is a
layout-inference decision, driven by the operands' *producer chains*.

The AV `dot_general` is byte-identical (`batching [0]x[0], contracting [2]x[1]`) in the
working `enc6` and the failing DSL output.

---

## 4. Controlled experiments already run (the science)

All single-variable, oracle = "does `iree-compile … --torq-hw=SL2610` still assert 57?".
`iree-reduce`/`mlir-reduce` are NOT in the v2.0.0 release, so this was done by hand-crafted
generators (below) and targeted MLIR edits.

| # | Variant | Result | Conclusion |
|---|---|---|---|
| 1 | `enc6` hand-written 6-layer (clean, 2D, no RoPE) | **OK** | recipe + IR valid |
| 2 | `enc6` + live leading-1 round-trip `[8,165,36]→[1,8,165,36]→[8,165,36]` on q/k/v | FAIL | a *live* round-trip breaks it |
| 3 | `enc6` + a **dead** unsqueeze `[8,165,36]→[1,8,165,36]` (unused) | **OK** | dead reshapes are harmless — only live round-trips matter |
| 4 | `enc6` + RoPE-style reshape `[8,165,36]→[8,165,18,2]→[8,165,36]` on q/k | **OK** | RoPE interleave reshape is NOT the trigger |
| 5 | `cmb` K=transpose-then-slice, output=direct | **OK** | — |
| 6 | `cmb` K=transpose-then-slice, output=round-trip | **OK** | **output round-trip is IRRELEVANT** |
| 7 | `cmb` K=slice-then-transpose, output=direct | FAIL | **K ordering is root cause #1** |
| 8 | `cmb` K=slice-then-transpose, output=round-trip | FAIL | — |
| 9 | DSL 2D 1-layer, K-fix applied | FAIL | #1 alone insufficient on full function |
| 10 | DSL 2D 1-layer, K-fix + RoPE removed | FAIL | root cause #2 is global, not RoPE |
| 11 | Single matmul `[165,N]` up to N=1280; `t14` grouped attention; `cmb_enc6` | OK | isolated pieces fine |
| 12 | `--torq-{enable-transpose-optimization,enable-tosa-identity,disable-linalg-slicing,disable-dispatch-fusion,disable-slicing}` on the failing encoder | FAIL | no flag fixes #2 |

Key deductions: the AV operands in the failing DSL output are, after root-cause-#1 fix and
RoPE removal, produced by the *same* op chains as enc6 (V: `proj→reshape→transpose→slice`;
scores: `softmax(QK)`), yet it still fails — so Torq's layout solver is influenced by the
**whole function** (LayerNorm/FFN/projection structure elsewhere), not just the AV's local
producers.

---

## 5. Root cause #1 — FIXED: transpose K before slicing

`TorqAttentionTilingPass` originally emitted, per head-group:
`kg = slice(K[H,S,D], head range) → [g,S,D]`, then `kt = transpose(kg,[0,2,1]) → [g,D,S]`.
Torq mis-lays-out that per-group transpose. The **fix** (already applied) transposes the
full K once, then slices:

```kotlin
// once, before the head loop:
val kTS = spec(h, d, sk)                                  // [H, D, Sk]
val kT = op("transpose", mapOf("permutation" to listOf(0,2,1)), listOf(k3s), kTS)
           .also { edges += Triple(kSrc, it, 0) }
// per head-group: just slice the already-transposed K
val ktS = spec(g, d, sk)                                  // [g, D, Sk]
val kt = op("slice", mapOf("start_indices" to listOf(s,0,0),
                           "limit_indices" to listOf(e,d,sk),
                           "strides" to listOf(1,1,1)), listOf(kTS), ktS)
           .also { edges += Triple(kT to 0, it, 0) }
```

This mirrors how the working `enc6` builds K (`transpose(kr,[1,2,0]) → [H,D,S]` first, then
slice). It is necessary but **not sufficient** for the full DSL encoder (root cause #2).

The pass also already: (a) bypasses the model's input `unsqueeze` so head-group slices pull
from the clean `[H,S,D]` source (`resolveFold`), and (b) seq-tiles the query. See
`SKaiNET-transformers/llm-inference/moonshine/src/jvmTest/kotlin/sk/ainet/models/moonshine/TorqAttentionTilingPass.kt`
and `TorqFfnTilingPass.kt`.

---

## 6. Root cause #2 — SOLVED: K transpose must route through [Sk,H,D]

Found by the §4 method in reverse — *reducing* the failing DSL layer toward the known-good
`layer1`, one feature at a time. Ruled out (each removed from the failing file, still asserts):
the 18 dead `[1,8,165,36]` graph outputs, the QKV/O weight pre-transposes, the FFN weight
pre-transpose. The **single feature that flips FAIL→OK** is the K transpose:

- **DSL (FAIL):** K reaches the QK matmul as `[H,Sk,D] --transpose[0,2,1]--> [H,D,Sk]`. With
  RoPE the [H,Sk,D] is the RoPE output; without RoPE it is the model's `transpose[1,0,2]` of the
  `[Sk,H,D]` reshape (a literal double-transpose). Either way the *final* `[0,2,1]`-on-`[H,Sk,D]`
  is what Torq lays out wrong; the bad layout propagates and trips `MatMulPattern.cpp:57` on the
  AV matmul (whose own operands are byte-identical to a compiling graph — hence the earlier
  "global" red herring).
- **FIX (OK):** build K as `[H,Sk,D] --[1,0,2]--> [Sk,H,D] --[1,2,0]--> [H,D,Sk]`. The final
  `[1,2,0]` FROM a `[Sk,H,D]` tensor is exactly how enc6/layer1 build K. One formula fixes both
  the RoPE and no-RoPE emission (without RoPE the paired `[1,0,2]`s canonicalize away to a single
  `[1,2,0]`). Implemented in `TorqAttentionTilingPass.kt` (the `kShd`→`kT` two-hop).

Proven by controlled single-variable experiments on Torq v2.0.0 (compile oracle = does it still
assert 57): direct-`[0,2,1]` FAIL, route-through-`[Sk,H,D]` OK, on both the RoPE and no-RoPE
single-layer DSL exports; then regenerated from the patched pass and compiled the full 6-layer.
This also explains xfail #1408 — same K-layout class.

---

## 7. How the handcrafted variants are built (why they compile)

These generators are the durable artifacts (the `scratchpad/torq2/*.mlir` are ephemeral).
They emit **StableHLO** compiled with the §2 wrapper.

### 7a. `enc6` — the KNOWN-GOOD full encoder (compiles on Torq)

Structure per layer, all activations 2-D `[S,D]=[165,288]` (no batch-of-1):

1. **LayerNorm** (mean/var, no batch): `sum=reduce_add(x,dim=1)/N`; `c=x-mean_bcast`;
   `var=reduce_add(c*c,dim=1)/N`; `std=sqrt(var+eps)`; `norm=c/std_bcast`; `norm*γ + β`.
2. **QKV projections** (2-D): `x[165,288] · W[288,288] → [165,288]`,
   `dot_general contracting [1]x[0]`.
3. reshape `[165,288]→[165,8,36]`; **Q,V** `transpose[1,0,2]→[8,165,36]`;
   **K** `transpose[1,2,0]→[8,36,165]` **(one transpose to [H,D,S], BEFORE any slice)**.
4. Per head-group `g∈{[0:4],[4:8]}`: `Kg=slice→[4,36,165]`, `Vg=slice→[4,165,36]`.
   Per query-seq chunk `[0:83],[83:165]`: `Qc=slice(Qh,[hs:he, ss:se, :])→[4,sc,36]`;
   `scores=Qc·Kg` `[4,sc,36]·[4,36,165] contracting [2]x[1] → [4,sc,165]`;
   `*scale`; softmax over dim2 (`max→sub→exp→sum→div`); `out=attn·Vg`
   `[4,sc,165]·[4,165,36] contracting [2]x[1] → [4,sc,36]`.
   `concat(seq chunks, dim1) → [4,165,36]`.
5. `concat(head-groups, dim0) → [8,165,36]`; `transpose[1,0,2]→[165,8,36]`;
   `reshape→[165,288]`; **O-proj** `·Wo → [165,288]`; residual add.
6. **LayerNorm**; **FFN hidden-tiled** (4 chunks): `W1c=W1[:, c*288:(c+1)*288]`,
   `W2c=W2[c*288:(c+1)*288, :]`; `up_c=ln·W1c→[165,288]`; `g_c=tanh(up_c)` (GELU proxy);
   `p_c=g_c·W2c→[165,288]`; `sum(p_0..p_3)`; residual add.

The full Python generator (parameterized by `NL` layers; weights are function args
`%wq_l …`, `%w1_l/%w2_l`) is embedded verbatim in the session transcript that produced
`enc6.mlir`; reproduce with `NL=6`. Its distinguishing choices vs the DSL: **2-D activations
throughout**, **K transposed to [H,D,S] before slicing**, single combined head+seq slice on
Q, and no batch-of-1 reshapes anywhere.

### 7b. `cmb` — the 4-way isolator (single-layer attention, proj→attn→O-proj)

Toggles `kstyle ∈ {enc6, dsl}` and `outstyle ∈ {direct, rt}`:

- `kstyle=enc6`: `kh=transpose(kr,[1,2,0])→[8,36,165]`, then per group `Kg=slice→[4,36,165]`.
- `kstyle=dsl`: `kh=transpose(kr,[1,0,2])→[8,165,36]`, per group `Kgs=slice→[4,165,36]`,
  then `Kg=transpose(Kgs,[0,2,1])→[4,36,165]`.  **← the only line that flips pass/fail**
- `outstyle=rt`: insert `reshape [8,165,36]→[1,8,165,36]→[8,165,36]` before the O-proj
  transpose (proven irrelevant).

Result table = rows 5–8 of §4. `cmb_enc6_direct.mlir` (OK) vs `cmb_dsl_direct.mlir` (FAIL)
is the **minimal reproducer** — semantically identical, differ only in K ordering. Ship this
pair to Synaptics for root cause #1; build an analogous minimal pair for #2 once isolated.

---

## 8. Reproducer inventory (regenerate; scratchpad is ephemeral)

- `enc6.mlir` — hand-written 6-layer, **compiles** (§7a generator, NL=6).
- `cmb_{enc6,dsl}_{direct,rt}.mlir` — the 4-way (§7b).
- `layer1.mlir` — one full layer (LN+seq-tiled attn+residual+LN+hidden-tiled FFN+residual), compiles.
- `ffnh4c.mlir` — hidden-tiled FFN alone, compiles; `t14_grouped.mlir` — grouped attention, compiles.
- `moonshine-encoder.mlir` — the DSL pass output (`llm-inference/moonshine/build/build-mlir/`).
  Fails on Torq (#2), compiles on `llvm-cpu`.

---

## 9. Paths to running on board

- **A (guaranteed demo):** emit the whole encoder in `enc6`'s proven op-structure wired to
  real Moonshine weights (safetensors via the DSL loader → constants or an `.irpa`). Decouples
  "runs on NPU" from taming #2. Bounded work: weight-baking, not open-ended debugging.
- **B (clean DSL path):** finish §6 — fix each remaining global-layout trigger with the
  controlled method. Correct end-state; several iterations.
- **C (escalate):** send the §7 minimal repro + xfail #1408 to Synaptics. #2 is their bug.
- **CPU fallback** (`llvm-cpu`) always compiles/runs — ASR ships regardless of NPU timing.

Recommended: **A + C in parallel**, then B when #2 is fixed upstream.

---

## 10. Guardrails / things already ruled out (do not re-litigate)

- Not the tiling, not our shapes, not fp32 weights (bf16-native, solved), not the FFN,
  not the output round-trip, not RoPE's reshape, not dead reshapes, not any `--torq-*` flag.
- The input `unsqueeze` round-trip is already bypassed in the pass; the K-ordering is already
  fixed. The remaining work is root cause #2 (global layout) via path A/B/C above.
- Keep StableHLO **HW-agnostic**: all Torq-specific logic lives in the `TorqAttentionTilingPass`
  / `TorqFfnTilingPass` (registered via `TargetOptimizers`), NOT in core or the model. These
  passes will later move to `SKaiNET-embedded` (they are board-HW code).
```

---

## Milestone — Moonshine ASR end-to-end (2026-07-05)

The DSL encoder is **proven faithful** and the full ASR pipeline runs end-to-end on the
Torq simulator.

- **Blow-up fixed:** compile with `--torq-disable-slices` (Torq's slice-based NSS execution
  mis-aliased buffers in large graphs → 1e23/NaN; disabling it routes around the bug).
- **f32 recipe fixes:** LayerNorm computed in f32 (`LayerNormalization.forward` +
  `convertLayerNorm`); RoPE cos/sin tables and rotation in f32 (`RoPE.applyRoPEInterleavedOps`).
- **End-to-end test** (`test.wav` = "One, two, three."): frontend → encoder → decoder
  (HF `UsefulSensors/moonshine-tiny` via transformers, injecting our encoder output):
  - Our DSL encoder on **IREE llvm-cpu**, real weights: cos **0.99999** vs reference →
    transcribes **"One, two, three." exactly**. The model + weight mapping are correct.
  - Our DSL encoder on **Torq sim** (`--torq-disable-slices`): cos **0.717** → transcribes
    "You" (wrong). Runs e2e, no NaN, bounded — but the per-layer precision residual
    compounds (1 layer 0.857 → 6 layers 0.717), below the decoder's tolerance.

**Last mile:** tighten per-layer Torq attention precision — the vendor's `bf16×bf16→f32`
accumulation for the QK/AV matmuls (softmax is already f32), same recipe as the f32
LayerNorm/RoPE. Then 6 layers stay above tolerance and Torq transcribes correctly.
