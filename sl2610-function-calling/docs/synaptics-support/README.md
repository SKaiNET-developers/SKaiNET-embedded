# Torq compiler: StableHLO encoder fragments into ~40 dispatches (returns zeros on SL2610), where your shipped encoder is 1 fused dispatch

**Where to file this**
- **Primary (recommended):** GitHub issue on the compiler repo — https://github.com/synaptics-torq/torq-compiler/issues
  (public, Apache-2.0, Issues enabled — this is the compiler team's repo). Attach `encoder_1layer.mlir`.
- **Formal support ticket:** Astra Support Portal — https://synacsm.atlassian.net/servicedesk/customer/portal/543
- Docs referenced: https://synaptics-torq.github.io/torq-compiler/v/latest/

---

## Message

Hi Torq team,

First — thank you for shipping the compiler and runtime as open IREE/MLIR. We're building our own
ML framework (**SKaiNET**, a Kotlin-multiplatform NN library): we author models in a small NN DSL,
trace them to a compute graph, and emit **StableHLO**, which we then hand to your `iree-compile`
(Torq fork) for the SL2610 (Astra Machina, Astra SDK `scarthgap_6.12_v2.4.0`). The open toolchain is
exactly what let us get this far, and we'd love your guidance on one issue.

**Symptom.** Our self-authored Moonshine-tiny **encoder** StableHLO compiles fine but fragments into
**~40 dispatches**, and on the SL2610 NPU (`--device=torq --torq_hw_type=astra_machina`) it returns
**all zeros**. Your own shipped encoder for the same model
(`.../models/Synaptics/moonshine-tiny-bf16-torq/encoder.vmfb`) compiles to a **single fused dispatch**
and runs correctly on the same board. So the model math is fine — the difference is dispatch
segmentation/fusion.

**What we've verified (SL2610, board + your runtimes):**

| graph | dispatches | result on `--device=torq` |
|---|---|---|
| your `encoder.vmfb` (moonshine-tiny-bf16-torq) | **1** | RUNS (nonzero, sane) |
| our 2-matmul chain / matmul→softmax→matmul | 1 / 2 | RUNS |
| **our 1-layer encoder** (`encoder_1layer.mlir`, attached) | **41** | **ALL ZEROS** |
| our full 6-layer encoder | ~240 | ALL ZEROS |

- Dispatch count is the discriminator: 1–2 dispatch graphs run; the many-dispatch encoder returns zeros.
- The dtype recipe matches yours (we cast matmul inputs to bf16 → `bf16×bf16→f32` projections + bf16
  attention interior + f32 LayerNorm reductions) — the zeros persist, so it isn't a dtype issue.
- It isn't the NPU clock (`devmem 0xf7e104b0 32 0x216`): your encoder runs bit-identically at the boot
  clock and 0x216.
- If we **disable tiling** (compile the attention block un-tiled to reduce dispatch count), the compiler
  instead **crashes in CSS codegen** — `iree-lld: cannot open /tmp/css_linalg.generic_*.o` after an
  abort, i.e. it can't lower the un-tiled bf16 attention/softmax as one kernel. So we're stuck between
  "tiled → too many dispatches → runtime zeros" and "un-tiled → CSS codegen crash."

**Our questions:**
1. **Fusion:** How do we get `iree-compile` to fuse the encoder into one (or few) dispatch(es), the way
   your `encoder.vmfb` is built? Is there a flag / pass / pipeline (segmentation-fusion, dispatch-region
   policy) we should be enabling?
2. **Runtime limit:** Is there a per-invocation **dispatch-count limit** on the SL2610 runtime, and what
   is it? (Our 1–2 dispatch graphs run; ~40 return zeros silently, with no error.)
3. **Import path:** Does importing from **StableHLO** vs ONNX change dispatch segmentation? Is there a
   recommended StableHLO shape/op form for good fusion on Torq?
4. **CSS codegen crash:** Is the `css_linalg.generic` codegen abort (un-tiled bf16 attention/softmax) a
   known issue, and is it fixed in a newer build?
5. **Compiler build:** The public **v2.0.0** compiler fragments our graph; the build that produced your
   moonshine `encoder.vmfb` fuses it to 1 dispatch. Is that build/config available, or can you share the
   compile recipe/flags used for the shipped moonshine encoder?

Repro attached (`encoder_1layer.mlir`, self-contained StableHLO, weights as inputs). Exact commands and
observed output below. Happy to provide the 6-layer graph, the board runtime logs, or a smaller synthetic
if useful. Thanks a lot!

— [your name], SKaiNET

---

## Reproducer

`encoder_1layer.mlir` — one Moonshine-tiny encoder layer (pre-norm attention + GELU FFN), authored in
the SKaiNET DSL and emitted as StableHLO. Self-contained; weights are function inputs. bf16 activations,
f32 LayerNorm reductions (the vendor recipe).

### Compile (Torq `iree-compile`, v2.0.0)
```
iree-compile \
  --iree-input-type=stablehlo \
  --iree-hal-target-device=torq --torq-hw=SL2610 \
  --torq-fallback-f32-to-host \
  encoder_1layer.mlir -o encoder_1layer.vmfb
```

### Observed: 41 dispatches
```
strings encoder_1layer.vmfb | grep -oE 'dispatch_[0-9]+' | sort -u | wc -l
# -> 41
```

### Run on the SL2610 NPU → all zeros
```
iree-run-module --module=encoder_1layer.vmfb --function=main \
  --device=torq --torq_hw_type=astra_machina \
  --input=... (weights + [1,165,288] bf16 activation) --output=@out.bin
# out.bin is all zeros; your 1-dispatch encoder.vmfb on the same board is nonzero/correct.
```

### Contrast (same board): your fused encoder runs
```
strings .../moonshine-tiny-bf16-torq/encoder.vmfb | grep -oE 'dispatch_[0-9]+' | sort -u | wc -l
# -> 1   (runs correctly, in bf16[1,288,207] -> out bf16[1,207,288], nonzero)
```

### The un-tiled variant instead crashes the compiler
Compiling the same layer with attention **un-tiled** (one large block) aborts in CSS codegen:
```
iree-lld: error: cannot find linker script /tmp/css_linalg.generic_0-*.ld
iree-lld: error: cannot open /tmp/css_linalg.generic_0-*.o: No such file or directory
... Aborted (core dumped)
```
