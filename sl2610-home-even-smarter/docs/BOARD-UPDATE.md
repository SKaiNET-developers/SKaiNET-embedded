# Board, Torq tools and NPU update procedure

The board image, the Torq NPU kernel module, the Torq runtime and the Torq compiler are versioned **as a set**.
Synaptics releases one Torq framework per Astra SDK. A model compiled by one Torq version and run on another
either fails to load or, worse, loads and returns zeros. This document is the only supported way to move the set.
[`board/board.lock`](../board/board.lock) records the set the app is verified against.

| Astra SDK (board image) | Torq framework (compiler + runtime + `syna_npu.ko`) | Notes |
|---|---|---|
| scarthgap_6.12_v2.4.0 | 2.0.x | **current board**. Self-compiled multi-dispatch NPU graphs return zeros. |
| scarthgap_6.12_v2.5.0 | 2.1.x | NPU devfreq: default clock 800 MHz → 1 GHz. Generic linalg slicing (`--torq-disable-slicing` if slower). |
| scarthgap_6.12_v2.5.0+ | 2.2.x (2.2.1 = 2026-09-23) | Kernel-module dependencies built for Astra 2.5. Fixes "runtime auto mode returning zeros or SIGSEGV". Tile-and-fuse improvements, LayerNorm on the NSS, `torq-decode-llm` (LLM decode on the NPU). Still IREE 3.10. |
| scarthgap_6.12_v2.6.0 / v2.6.1 | (2.2.x expected) | 2.6.0 has the plain `sl2619_scarthgap` eMMC image. 2.6.1 lists only coralboard variants for sl2619. |

Since 2.1 every Torq release ships `astra-release.tar.gz`. It contains a prebuilt `lib/syna_npu.ko` and
`tool/astra-sl-torq-run-module`, built for the **matching** Astra kernel. So a newer Torq runtime on the board
needs the matching board image, not just a new wheel.

**Target:** Astra SDK **scarthgap_6.12_v2.6.0** (`sl2619_scarthgap`, eMMC) and Torq **2.2.1**. Take v2.5.0 if
v2.6.0 does not ship a Torq ≥ 2.2 driver (checked in step 6).

## What the board looks like (2026-10-16)

- `Synaptics SL2619 RDK`, boots from **eMMC** (`root=/dev/mmcblk0p12`), kernel 6.12.62, SDK revision
  `202605280005_202605291432` (v2.4.0).
- `/` is 2.3 GB and **98 % full** (54 MB free). Never install into `/`; everything of ours lives in `/home/root`.
- `/home` is a separate 22.8 GB partition (`mmcblk0p18`). A full eMMC flash is assumed to **wipe it**.
- Several Torq runtimes are installed side by side (native `/usr/bin` 2.0.0_beta, two venvs, `torq2`, `torq2-stable`,
  `torq-v2.0.0-official`). The app does not use any of them. It links its own runtime (`libiree_torq_runtime.a`,
  built from the Torq sources).
- Mali-G31 (1 core). `libmali` exports a Vulkan ICD, but no ICD json is registered.
- Audio: only the I2S card. No USB mic attached.

## Procedure

Steps 1–3 and 5–9 run from the host over ssh. Step 4 needs **physical access** (USB cable, boot switch).

### 1. Baseline (read-only)
```
BOARD=root@<ip> scripts/board-audit.sh                # must print "OK: board matches the lock"
```
Record the canary matrix and the app numbers on the current set:
- Canaries: `sl2610-function-calling/scripts/torq-verify.sh` (identity_1x4, chain2_64, chain8_64, softmax_64).
- App: golden set via WAV replay, NLU ms/token, ASR RTF, RSS.

Write them into the log. These are the numbers the update has to match or beat.

### 2. Backup
```
BOARD=root@<ip> scripts/board-backup.sh               # vendor models, examples, runtimes, test files → build/board-backup/<date>
FULL=1 BOARD=root@<ip> scripts/board-backup.sh        # also the regenerable trees (~8 GB)
```
Check `SHA256SUMS`. The irreplaceable item is the **vendor Moonshine encoder** (`moon/`,
`sl2610-examples/models/…/encoder.vmfb`), which is today's only working NPU encoder.

### 3. Host tools (side by side, nothing replaced)
1. **Compiler:** `pip install torq-compiler==2.2.1` into a new venv, or build the docker image
   `sl2610-iree:v2.2.1` from the Dockerfile with `TORQ_VERSION=2.2.1`. Keep `sl2610-iree:v2.0.0`.
   Note: 2.2.1 wheels are `manylinux_2_39`, so the host needs glibc ≥ 2.39 (or use the docker image).
2. **App runtime:** rebuild `libiree_torq_runtime.a` from torq-compiler tag `v2.2.1` with the R1 recipe
   (`native/iree-torq-runtime/build-runtime.sh`, Kotlin/Native toolchain and sysroot). Check the IREE fork pin
   (`git -C torq-compiler submodule status`) and the bytecode/exec format versions it accepts.
3. **Board artifacts:** download `astra-release.tar.gz` and `torq_runtime-2.2.1-…-aarch64.whl` from the
   v2.2.1 release. Record their sha256 in the log.
4. **Board image:** download `sl2619_scarthgap_6.12_v2.6.0.zip` (and keep `…_v2.4.0.zip` for rollback) plus the
   USB flashing tool `synaptics-astra/astra-update` (v2.0.4). Record the sha256 values.

### 4. Flash the board (physical; wipes the board)
Follow the Astra SDK flashing guide for SL2619 eMMC using `astra-update` (USB boot). Before you start:
- the backup from step 2 is verified
- the v2.4.0 image is downloaded (rollback)
- you have serial console access (`ttyS0`, 115200)

After the first boot:
- Note the new IP (the board may get a different address).
- Re-enable root ssh and authorize the host key.
- Recreate `/home/root`.

### 5. Post-flash audit (read-only)
`scripts/board-audit.sh` will now report DRIFT, as expected. Check it against the target:
- `KERNEL`, `SDK_REVISION` = v2.6.0. `/dev/torq` present.
- `syna_npu` vermagic matches the kernel.
- `NPU_DEVFREQ` is reported (1 GHz expected).
- Root free space is at least 200 MB.
- Mali still present; whether a Vulkan ICD json is registered now (input for the GPU spike).
- ALSA cards; plug in the USB mic → `arecord -l` shows it.

### 6. NPU driver and runtime
1. Check which Torq version the image's `syna_npu.ko` belongs to (`modinfo syna_npu`, `dmesg | grep -i torq`).
   If it predates 2.2, take `lib/syna_npu.ko` from `astra-release.tar.gz`. It must have the **same vermagic** as
   `uname -r`; if not, stop and use the matching SDK. Install it under `/lib/modules/$(uname -r)/updates/`,
   then run `depmod -a` and reboot.
   Do not `insmod` over a loaded module on a board without serial access.
2. Install the 2.2.1 runtime **next to** anything else: `/home/root/torq-2.2.1/` (wheel or
   `astra-sl-torq-run-module`). It is used only for the canaries; the app links its own runtime.

### 7. Gate (decides whether the update is kept)
Compile the canaries with the **2.2.1** compiler and run them on the **2.2.1** runtime:

| Canary | 2.0.0 result | Required |
|---|---|---|
| identity_1x4 | PASS | PASS |
| chain2_64 | PASS | PASS |
| chain8_64 (3+ dispatches) | FAIL (zeros) | PASS → multi-dispatch is trustworthy |
| softmax_64 | FAIL | PASS |

Then run the real NPU graphs:
- Our self-compiled Moonshine v2 encoder: zeros before. Golden-compare against CPU, cos-sim ≥ 0.999.
- The vendor encoder: exec v0, so it may need recompiling or the R7 patch.

**If chain8/softmax still fail,** keep the new image only if nothing else regressed, and keep
`NPU_MULTIDISPATCH_TRUSTED=no`.

### 8. Rebuild and verify the app
1. Recompile every board graph with the 2.2.1 compiler: Moonshine on the NPU, Gemma and the Moonshine CPU parts.
   CPU graphs must come from the fork compiler too, because upstream IREE vmfbs are rejected.
2. Re-materialize the cartridges and serve them from `:server`.
3. Link the app against the 2.2.1 `libiree_torq_runtime.a`, then `scripts/deploy-tui.sh`.
4. Golden set via WAV replay and live mic; record the numbers next to the step-1 baseline.

### 9. Flip the lock
Update `board/board.lock` (current values ← target, new target cleared, `NPU_MULTIDISPATCH_TRUSTED`).
`scripts/board-audit.sh` must print OK. Commit the lock, the scripts and the log entry together.

## Rollback
- **Before step 4:** nothing on the board changed; discard the host-side 2.2.1 tools.
- **After step 4:** flash `sl2619_scarthgap_6.12_v2.4.0` with `astra-update`, restore the step-2 backup to
  `/home/root`, redeploy the app built against the 2.0.0 runtime, run `scripts/board-audit.sh` (must be OK).
- **After step 6 only** (module swapped, image kept): restore the module the image shipped with, `depmod -a`, reboot.
