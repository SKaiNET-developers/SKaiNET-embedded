# Toolchain pin & currency policy

**Question this answers:** are we running the latest libs/tools on the board, and if not, *why not* —
as a conscious, re-validated decision rather than drift?

**Policy (chosen 2026-07-11):** *re-validate the newest Torq release; adopt it if it passes the canary
gate, otherwise keep the current pin and document the reason here.* This file is that record. The raw
evaluation notes live in [`../../../docs/torq-debug-notes/torq-toolchain-update.md`](../../../docs/torq-debug-notes/torq-toolchain-update.md).

## Current pins (measured 2026-07-03, re-affirmed 2026-07-11)

| Layer | Pinned now | Newest known | Status |
|---|---|---|---|
| Astra Linux SDK (board OS/BSP) | `scarthgap_6.12_v2.4.0` (Poky 5.0.9) | `v2.4.0` (2026-05-30) | **Current — no action** |
| Kernel | Linux `6.12.62` aarch64 | — | Current |
| **Torq compiler** (host) | `torq_compiler 0.dev0+…g165e12a` (dev snapshot, 2026-05-28) | stable **v2.0.0** (2026-06-20) | **Pinned deliberately — see below** |
| **Torq runtime** (board) | `torq_runtime 2.0.0a1` (alpha) | stable **2.0.0** | Locked to the compiler (bytecode skew otherwise) |
| IREE (host CPU / conformance) | `3.11.0` | in use | Current |

The **board OS is already current.** The one intentionally-old component is the **Torq compiler/runtime
pair**, held at the `g165e12a` snapshot.

## Why the Torq compiler is pinned to g165e12a (not stable v2.0.0)

v2.0.0 was downloaded (public, no login) and tested against our reproducers on 2026-07-03. Result:
1. **It does not fix the blocker.** The core limitation — large matmul intermediates don't fit LRAM
   (`[165,1152]` FFN output, `[8,165,165]` attention) — still overflows on stable v2.0.0
   (`TileAndFusePass.cpp:607 checkModuleFitsInMemory`, `failed to allocate LRAM addresses`). So this is a
   *standing Synaptics limitation*, not merely a dev-snapshot bug.
2. **It changes matmul-layout rules.** Our g165e12a-tuned attention tiling trips `MatMulPattern.cpp:57`
   on the full encoder under v2.0.0 — the tiling passes would need re-tuning.
3. **Lock-step risk.** Compiler↔runtime must match (the "Ch" bytecode-feature rejection is exactly this
   skew). Switching the board would put the **working g165e12a FunctionGemma path** at risk.

**Decision: keep g165e12a as the production pin.** It is the toolchain that (a) compiles FunctionGemma to
board-runnable bytecode and (b) matches the board's `2.0.0a1` runtime. This is the right call *today* even
though it is not the newest label — "latest" that regresses the working path is not an improvement.

## When to re-run this decision

Re-validate whenever any of these changes:
- a Torq release **newer than v2.0.0** appears (v2.1+), especially one whose notes mention Tile-and-Fuse /
  LRAM allocation fixes;
- Synaptics responds to the escalation (repro `docs/torq-debug-notes/torq-repro/a2_mm2.mlir`, "reproduces on stable v2.0.0")
  with a tiling recipe or a fix;
- the board OS/runtime is bumped for another reason (re-pair the compiler).

## Re-validation procedure (the canary gate)

Run the candidate compiler through this gate; **adopt only if every step is green.** Needs the board + the
candidate wheel.

```bash
# 0. stage the candidate WITHOUT touching the production pin
export TORQ_PKG=/path/to/torq-<candidate>/…/torqpkg     # in a throwaway shell / demo.env override

# 1. the standing LRAM reproducer must now COMPILE (it fails today on g165 and v2.0.0)
"$TORQ_PKG/iree/compiler/_mlir_libs/iree-compile" --iree-input-type=stablehlo \
   --iree-hal-target-device=torq --torq-hw=SL2610 \
   ../../docs/torq-debug-notes/torq-repro/a2_mm2.mlir -o /tmp/mlp2.vmfb          # green => tiling fixed

# 2. compile the two shipping models with the candidate
scripts/compile-gemma.sh board                                   # FunctionGemma
./gradlew moonshineEncoderMlir && scripts/iree-compile-torq.sh \
   build/mlir/moonshine-encoder.mlir build/mlir/enc-candidate.vmfb

# 3. deploy + run the SILENT-ZEROS canary + token oracle on the board
BOARD=root@<ip> ./gradlew deployBoard
#   FunctionGemma oracle: "turn the light on" -> [262146,236769,3255,718,498,1373,262152,106]
#                         = <tool_0>(state="on")<end>
#   encoder canary: output must be non-zero (a multi-dispatch build returns all-zeros on-device)

# 4. regression sweep — the LLM path, not just ASR
#   re-run skainet-iree-conformance on the pinned pair; confirm FunctionGemma still token-for-token.
```

- **All green →** bump `TORQ_PKG` in `demo.env.example`, update the version rows above, bump the board
  runtime to the matching version, and record the new pin + date here.
- **Any red →** keep g165e12a, append the finding + date to the "Why … pinned" section, and (if the LRAM
  repro still fails) strengthen the Synaptics escalation with the newer version number.

## Related
- The load-bearing runtime guard that actually catches a mismatch at load time:
  `SKaiNET-transformers/llm-runtime/gemma-iree/…/IreeRuntime.kt` (stock IREE 3.x bytecode is rejected by the
  board — vmfbs MUST be built with the Torq-fork compiler).
- Open escalation: our self-compiled encoder fragments into ~40 dispatches (returns zeros on-device) while
  the vendor `encoder.vmfb` is one fused dispatch — see `docs/synaptics-support/README.md`. Getting the
  vendor's fusing compiler / tiling recipe is the real unblock for a fully self-compiled NPU encoder.
