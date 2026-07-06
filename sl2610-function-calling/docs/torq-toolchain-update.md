# Torq toolchain update — safe procedure for the SL2610 edge board

**Why this exists.** We ship compiled `.vmfb` models to an edge NPU where the compiler, the board
runtime, and the firmware each have their own version, and mismatches fail **silently** (a vmfb
loads and returns zeros, or is rejected outright). This runbook is the *only* sanctioned way to
change the Torq compiler or board runtime. It is **non-destructive, canary-gated, and reversible**.

The single source of truth is [`scripts/torq-toolchain.lock`](../scripts/torq-toolchain.lock).
Two tools enforce it:
- `scripts/torq-board-audit.sh` — read-only; prints the board's runtime landscape and exits non-zero on drift.
- `scripts/torq-verify.sh` — the **canary gate**; compiles tiny known-answer models and runs them on the
  board's real runtime, printing a PASS/FAIL matrix. `--matrix` tries every compiler × runtime.

The canaries (`scripts/torq-canaries/*.mlir`) each catch a distinct failure class we actually hit:
`identity_1x4` (exec-format loads), `chain2_64` (simple NSS execution), **`chain8_64` (3+-dispatch
silent-zeros)**, `softmax_64` (CSS-kernel execution).

---

## Current state (2026-07-07)

- Board OS: Astra SDK **`scarthgap_6.12_v2.4.0`**, but the on-board Torq runtimes are all 2.0.0-era:
  `torq-runtime/2.0.0_beta` (native `/usr/bin`, exec **v1**), `torq_runtime-2.0.0a1` (the venv the app
  calls, exec v0/lenient), `torq_runtime-2.0.0` (`/home/root/torq2-stable`, exec **v2**).
- Pinned compiler: `sl2610-iree:v2.0.0` (emits exec **v2**); canonical runtime: `torq2-stable`.
- **Canary verdict = PARTIAL:** `identity`+`chain2` PASS on the matched pair; **`chain8`+`softmax` FAIL
  on every pair** (zeros/error/compile-crash). No available (compiler, runtime) pair executes a
  3+-dispatch graph → multi-dispatch models (the ASR encoder/decoder) are **not trustworthy on
  `--device=torq`**. `NPU_MULTIDISPATCH_TRUSTED=no` in the lockfile. The CPU floor
  (`--device=local-task`, verified "One, two, three." e2e) is the shippable path.
- Two known latent mismatches surfaced by the audit: the app (`MoonshineDecoder.kt` `torqBin`) calls
  the **alpha** venv runtime, not the canonical `torq2-stable`; and the g165e12a host compiler
  (`scripts/iree-compile-torq.sh`, now quarantined) emits the wrong exec-format for ASR.

To make multi-dispatch NPU models trustworthy we need the **SDK-v2.4.0-matching** Torq toolchain
(a compiler that both compiles our models AND emits what the v2.4.0 firmware executes), or the SyNAP
path. Getting that artifact is the trigger for this runbook.

---

## Procedure — updating the compiler and/or board runtime

**Never overwrite the working runtimes.** Install alongside; flip only after the canary passes.

1. **Obtain the matched pair.** From `github.com/synaptics-torq/torq-compiler` releases /
   `ghcr.io/synaptics-torq/torq-compiler/compiler:*` (or the SDK v2.4.0 image), get the compiler wheel
   and the **matching runtime** artifact. Record both versions.
2. **Baseline.** `scripts/torq-board-audit.sh` and `scripts/torq-verify.sh --matrix` — save the current
   PASS/FAIL matrix so you can prove improvement (or catch regression).
3. **Install the new compiler (host).** Build a new docker image `sl2610-iree:<newtag>` from
   `scripts/.docker/Dockerfile.iree` (set `TORQ_VERSION`), OR a venv. Do NOT retag the existing pin yet.
4. **Install the new runtime (board), non-destructively.** `scp` it to a **new** path, e.g.
   `/home/root/torqNEW/`. Leave `/usr/bin`, the FunctionGemma venv, and `torq2-stable` untouched.
   Add the new (compiler, runtime) to the arrays in `scripts/torq-verify.sh` (`COMPILERS`/`RUNTIMES`).
5. **Gate.** `scripts/torq-verify.sh --compiler <new> --runtime <newpath-id>`. Read the matrix:
   - **All four canaries PASS** (esp. `chain8_64`) → the new pair executes multi-dispatch. Proceed.
   - `chain8_64`/`softmax_64` still FAIL → the new toolchain does **not** fix the hardware/execution
     limit; STOP, keep the current pin, record the result. (This is exactly what the gate is for.)
6. **Flip the canonical (only on all-PASS).** Update `scripts/torq-toolchain.lock`:
   `COMPILER_ID`, `COMPILER_EMITS_EXEC`, `BOARD_RUNTIME_ID/PATH/VERSION`, `EXEC_FORMAT`,
   `CANARY_*`, `NPU_MULTIDISPATCH_TRUSTED=yes`. Retag the image to the pin if desired.
7. **Repoint the app** to the canonical runtime: update `torqBin`/`torqLibs` in
   `src/linuxArm64Main/.../MoonshineDecoder.kt` to `BOARD_RUNTIME_PATH`, rebuild + `deployBoard`.
   Re-run `torq-board-audit.sh` — it must now report **OK** (no DRIFT).
8. **Verify e2e** and commit the lock + script/doc changes together.

## Rollback
- `git checkout scripts/torq-toolchain.lock` (and the `MoonshineDecoder.kt` path) → back to the prior pin.
- The old runtime dir is still on the board; nothing was overwritten. Re-run the audit to confirm.
- The CPU floor never depended on any of this and keeps working throughout.

## Reflash (higher risk, only if Phase 0 says so)
If no runtime on the *current* image matches a working compiler, the board must be reflashed to the
Astra SDK release that ships the needed runtime. Follow `coral/astra/FLASH_ASTRA_MACHINA_SL2610.md`,
then **re-run steps 2–8** — a reflash resets the entire runtime landscape, so re-establish the lock
from scratch.
