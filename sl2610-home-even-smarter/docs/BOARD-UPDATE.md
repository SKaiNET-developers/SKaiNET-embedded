# Board, Torq tools and NPU update: step-by-step instructions

How to move the SL2619 board and the host tools from **Astra SDK v2.4.0 + Torq 2.0.0** to **Astra SDK v2.6.0 +
Torq 2.2.x**. The guide covers downloads, the flashing tool, the image, the NPU module, the host compiler, the
verification gate and the rollback.

Every download is pinned to the sha256 published with its GitHub release. [`board/board.lock`](../board/board.lock)
records the set the app is verified against.

---

## 0. Background

### Why the versions move together

The board image, the Torq NPU kernel module (`syna_npu.ko`), the Torq runtime on the board and the Torq compiler on
the host are versioned **as one set**. Synaptics releases one Torq framework per Astra SDK. A model compiled by one
Torq version and run on another either fails to load (`version N does not match expected version M`) or, worse,
loads and returns zeros.

| Astra SDK (board image) | Released | Kernel | Torq in the image | Matching Torq framework |
|---|---|---|---|---|
| scarthgap_6.12_v2.4.0 | 2026-05-30 | 6.12.62 | 2.0.0 | 2.0.x, **current board** |
| scarthgap_6.12_v2.5.0 | 2026-07-31 | 6.12.62 | 2.1.x | 2.1.x |
| scarthgap_6.12_v2.6.0 | 2026-09-30 | 6.12.62 | **2.2.0** | 2.2.x, **target** |
| scarthgap_6.12_v2.6.1 | 2026-10-13 | 6.12.62 | 2.2.x | sl2619 *coralboard* variants only, no plain `sl2619` (RDK) image |

### Why update

Our blocker is NPU graphs with three or more chained dispatches returning zeros or nothing. That is the self-compiled
Moonshine encoder, and canaries `chain8_64` / `softmax_64`. Torq 2.1/2.2 bring:
- a fix for "runtime auto mode returning zeros or SIGSEGV"
- tile-and-fuse improvements and LayerNorm on the NSS
- `torq-decode-llm`, an LLM decode driver on the NPU
- NPU devfreq (800 MHz → 1 GHz)
- a prebuilt `syna_npu.ko` in every release

Astra v2.6.0 additionally brings NPU DVFS, eMMC tuning and newer firmware.

### What was measured on 2026-10-16 (current board, Torq 2.0 driver)

| Compiler × runtime | identity_1x4 | chain2_64 | chain8_64 | softmax_64 |
|---|---|---|---|---|
| 2.0.0 × torq2-stable 2.0.0 | PASS | PASS | no output | wrong values (mean 64) |
| 2.2.1 × 2.2.1 `astra-sl-torq-run-module` | PASS | PASS | no output | no output |

The new runtime alone does not fix it on the **old driver**. The open question is the 2.2 driver, which needs
Path A or Path B below.

### Two paths

| | Path A: module trial on the current image | Path B: full flash to Astra v2.6.0 |
|---|---|---|
| What changes | `syna_npu.ko` loaded by hand from Torq 2.2.1 (same vermagic `6.12.62`); nothing written to `/` | entire eMMC: boot chain, kernel, rootfs, **`/home`** |
| Reversible by | reboot (the old module loads again) | flashing v2.4.0 back + restoring the backup |
| Physical access | only if the kernel hangs (power cycle) | **yes** (USB-C cable, USB_BOOT and RESET buttons) |
| Answers | does the 2.2 driver + 2.2.1 runtime fix multi-dispatch? | everything, plus devfreq, DVFS, GPU stack, new kernel modules |

Recommendation: do Path A first. It is cheap and answers the main question. Do Path B when someone is at the board,
whatever Path A shows, because it also moves the image to what Synaptics tests the 2.2 runtime with.

---

## 1. What you need

**Host:** Linux x86_64 with:
- `curl`, `sha256sum`, `unzip`, `git` and **`git-lfs`** (`sudo apt install git-lfs`)
- `docker`
- `python3` with `numpy` and `ml_dtypes`, for the canary checks
- `sudo`, because `astra-update` needs USB access
- About **3 GB** free: both images zipped are about 1.4 GB, the unpacked eMMC image about 1.5 GB

**Board side (Path B):**
- a **USB-C cable** for the USB 2.0 Type-C port next to the Ethernet port, data-capable
- the board's power supply
- network cable
- physical access to the **SW7 USB_BOOT** and **SW6 RESET** buttons on the I/O board
- the `SD_BOOT` jumper **not** fitted, so the board boots from eMMC
- optional but recommended: a 3.3 V USB-to-serial adapter on UART0 of the 40-pin header, 115200 8N1. See
  "Connecting to an external USB-to-serial adapter" in the Astra documentation.

**Accounts:** none. Everything is public on GitHub.

---

## 2. Downloads

All in one step, verified against the published sha256:

```
scripts/board-update-download.sh ~/board-update all          # or pick sets: image rollback torq usb-tool
```

| Set | File | Source | Size | sha256 |
|---|---|---|---|---|
| image | `sl2619_scarthgap_6.12_v2.6.0.zip` | [Astra SDK v2.6.0](https://github.com/synaptics-astra/sdk/releases/tag/scarthgap_6.12_v2.6.0) | 725 MB | `9b15b709a45b0e4b5d6c1532277e2465aee960f862961805150e220fb7ac665f` |
| rollback | `sl2619_scarthgap_6.12_v2.4.0.zip` | [Astra SDK v2.4.0](https://github.com/synaptics-astra/sdk/releases/tag/scarthgap_6.12_v2.4.0) | 705 MB | `7011b4fb6f4808834d80ff5f9096f6e596c87f68ee28c59cf6807cb082561de9` |
| torq | `astra-release.tar.gz` (`lib/syna_npu.ko`, `tools/astra-sl-torq-run-module`) | [Torq v2.2.1](https://github.com/synaptics-torq/torq-compiler/releases/tag/v2.2.1) | 0.6 MB | `f71693087ec336674366e254197fb622fb939e353dfd34c745e766c1edfdf433` |
| torq | `torq_runtime-2.2.1-cp312-cp312-manylinux_2_39_aarch64.whl` | Torq v2.2.1 | 4.3 MB | `7c17c5c627670314119f3a868f0fa6a877f718ab30efbfe8ac2088b3fe93de4e` |
| torq | `torq_compiler-2.2.1-cp312-cp312-manylinux_2_39_x86_64.whl` | Torq v2.2.1 | 87 MB | `37b8d56f6e148db0dda229b7b231cc6ad06ab699639d9cc3ac459cb9680da259` |
| torq | `user_manual.zip` | Torq v2.2.1 | 3.3 MB | `b08b6034559ba413915f1626828b1fb5d41a8355866e119368019d16365c453c` |
| usb-tool | [`synaptics-astra/usb-tool`](https://github.com/synaptics-astra/usb-tool) @ `91d4000af92f`: `astra-update` v2.0.4 (`bin/linux/x86_64/astra-update`) + USB boot images (`astra-usbboot-images/sl2619`) | Git LFS | 3.4 MB tool | pinned commit |

Not needed: the Yocto SDK toolchains (`*-toolchain-5.0.9.sh*`). The app is built with the Kotlin/Native toolchain
and sysroot. The `oobe`, `nand`, `spi_boot` and `usb_boot` images are other variants. Ours is the plain eMMC image
`sl2619_scarthgap`.

What is inside the image zip (checked remotely on 2026-10-16): `eMMCimg/` with
- `preboot`, `key`, `tzk`, `sysmgr`, `bl`, `boot`, `fastlogo`
- `rootfs_s.subimg.0-2` for both `rootfs_a` and `rootfs_b` (2560 MB each)
- **`home_s.subimg` → partition 18 = `/home`**

Both `emmc_image_list` and `emmc_image_list_full` write the home partition. **Flashing wipes `/home/root`.**

The `sl2619` USB boot image set (`astra-usbboot-images/sl2619/manifest.yaml`):
- board `rdk`, ddr4, 2 GB
- USB IDs: ROM USB boot `06cb:019e`, fastboot `18d1:4ee0`

---

## 3. Before you change anything (both paths)

### 3.1 Baseline
```
export BOARD=root@<board-ip>
scripts/board-audit.sh                      # must end with "OK: board matches the lock"
scripts/board-canaries.sh                   # 2.0.0 compiler × torq2-stable runtime
```
Record the app numbers as well:
- the golden set (`hes-tui --wav …`, then `g`)
- NLU ms/token and ASR time once real engines run

Copy the outputs into the log.

### 3.2 Backup (mandatory for Path B, recommended for A)
```
scripts/board-backup.sh ~/board-backup/$(date +%Y%m%d)       # vendor models, examples, runtimes, test files (~1.4 GB)
FULL=1 scripts/board-backup.sh ~/board-backup/full-$(date +%Y%m%d)   # also the regenerable trees (~8.4 GB)
(cd ~/board-backup/<date> && sha256sum -c --quiet SHA256SUMS && echo backup OK)
```
The irreplaceable file is the vendor Moonshine encoder, today's only working NPU encoder:
`sl2610-examples/models/Synaptics/moonshine-tiny-bf16-torq/encoder.vmfb`.

Also save the board's identity for the rollback check:
```
ssh $BOARD 'cat /etc/build*; uname -a; modinfo syna_npu' > ~/board-backup/<date>/board-info.txt
```

### 3.3 Host compiler for Torq 2.2.1
The 2.2.1 compiler wheel is `manylinux_2_39`, so it needs glibc ≥ 2.39; the Debian trixie base provides that.
```
docker build -f board/docker/Dockerfile.torq-compiler -t sl2610-iree:v2.2.1 \
  --build-arg TORQ_VERSION=2.2.1 --build-arg MANYLINUX=2_39 \
  --build-arg BASE=python:3.12-slim-trixie board/docker
docker run --rm --entrypoint sh sl2610-iree:v2.2.1 -c 'pip show torq-compiler | grep ^Version'   # → 2.2.1
```
Keep `sl2610-iree:v2.0.0` (default build args). It is the rollback compiler.

### 3.4 App runtime library for Torq 2.2.1
```
TORQ_TAG=v2.2.1 native/iree-torq-runtime/build-runtime.sh ~/work/torq-2.2.1    # IREE fork pin 1c6d87471fa9…
```
Build into a new work directory and keep the 2.0.0 one (`43014101c8ab…`). Then run the smoke test from
`native/iree-torq-runtime` against it (`build-smoke.sh`, `run-on-board.sh`). It must list the `torq` driver and pass
the CPU and NPU adds.

---

## 4. Path A: NPU module trial on the current image

Nothing is written outside `/home/root/torq-2.2.1`. A reboot restores the original module.
If the kernel hangs, the board needs a **power cycle**, so do this when someone can reach the board or its power.

1. **Copy the 2.2.1 board files** (already done on 2026-10-16 for the runtime tool):
   ```
   R=~/board-update/torq-2.2.1/astra-release
   ssh $BOARD 'mkdir -p /home/root/torq-2.2.1'
   ssh $BOARD 'cat > /home/root/torq-2.2.1/astra-sl-torq-run-module && chmod +x /home/root/torq-2.2.1/astra-sl-torq-run-module' < $R/tools/astra-sl-torq-run-module
   ssh $BOARD 'cat > /home/root/torq-2.2.1/syna_npu.ko' < $R/lib/syna_npu.ko
   ```
2. **Check compatibility before loading.** All of these must hold, otherwise stop:
   ```
   ssh $BOARD 'uname -r; modinfo /home/root/torq-2.2.1/syna_npu.ko | grep vermagic'   # both 6.12.62 … aarch64
   ssh $BOARD 'lsmod | grep syna_npu; ls -l /dev/torq'                               # loaded, used by 0
   ssh $BOARD 'ps | grep -E "[t]orq|[i]ree-run|[h]es-tui"'                           # nothing using the NPU
   ```
3. **Swap the module in memory only:**
   ```
   ssh $BOARD 'rmmod syna_npu && insmod /home/root/torq-2.2.1/syna_npu.ko && dmesg | tail -n 20 && ls -l /dev/torq'
   ```
   If `insmod` refuses (`disagrees about version of symbol`, `Invalid module format`), the 2.4.0 kernel and the
   2.2.1 module are not ABI-compatible. Run `insmod /lib/modules/6.12.62/updates/syna_npu.ko` to restore, record
   the message, and go to Path B.
4. **Gate:**
   ```
   scripts/board-canaries.sh sl2610-iree:v2.2.1 /home/root/torq-2.2.1/astra-sl-torq-run-module ""
   ```
5. **Restore:** `ssh $BOARD reboot`. The stock module loads from `/lib/modules/6.12.62/updates/` again. Then
   `scripts/board-audit.sh` must print OK.

Do **not** copy the 2.2.1 module into `/lib/modules`. A module that crashes during boot needs a reflash to recover.

---

## 5. Path B: flash Astra SDK v2.6.0

### 5.1 Prepare the flashing tool
```
cd ~/board-update/usb-tool
rm -rf eMMCimg && unzip -q ../sl2619_scarthgap_6.12_v2.6.0.zip      # creates ./eMMCimg next to update_emmc.sh
ls eMMCimg/emmc_image_list eMMCimg/rootfs.subimg.gz                  # sanity
```
`update_emmc.sh` runs `sudo ./bin/linux/x86_64/astra-update`, which flashes `./eMMCimg` and picks the boot image
from `astra-usbboot-images/`. To run without sudo, install a udev rule for `06cb:019e` and `18d1:4ee0`.

### 5.2 Flash
1. Shut the board down cleanly: `ssh $BOARD poweroff`. Leave it powered (or power it back on).
2. Connect the USB-C cable from the host to the board's **USB 2.0 Type-C port** (next to Ethernet).
   Optional: open the serial console (`picocom -b 115200 /dev/ttyUSB0`).
3. On the host, in `~/board-update/usb-tool`: `./update_emmc.sh`. It waits for a device.
4. Put the board into USB boot mode:
   1. press and hold **RESET (SW6)**
   2. while holding it, press and hold **USB_BOOT (SW7)**
   3. release **RESET**, keep holding **USB_BOOT**
   4. release **USB_BOOT** once the tool reports the device (or the serial console shows USB boot)
5. `astra-update` downloads U-Boot over USB and writes every partition. Progress bars show per image.
6. **Wait for the board to reset by itself.** The tool can report success before the last write has finished.
   A manual reset at this point can leave invalid partitions, which the Astra documentation notes especially for
   SL2610.
7. Log: if anything fails, `astra-update` keeps its temp directory and prints its path. Check
   `astradevicemanager.log` there. Add `-D -l stdout` for debug output.

### 5.3 First boot
1. Find the board's address: from the serial console (`ip -4 addr show eth0`) or from the DHCP server.
   It may differ from before.
2. Log in as `root` (serial console, or ssh if the image allows it). Install the host key:
   ```
   ssh-copy-id root@<new-ip>      # or: cat ~/.ssh/id_*.pub | ssh root@<new-ip> 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'
   export BOARD=root@<new-ip>
   ```
3. Audit. DRIFT against the old lock is expected. Check the target values:
   ```
   scripts/board-audit.sh
   ```
   - `SDK_REVISION` is newer than `202605280005_202605291432`, and `KERNEL=6.12.62`
   - `NPU_MODULE` present, `NPU_VERMAGIC` matches the kernel, `DEV_TORQ=present`
   - `NPU_DEVFREQ` reports a frequency (devfreq and DVFS are new in this image)
   - `ROOT_FREE` should be well above the old 54 MB
   - `VULKAN_ICD`: still empty, or now registered? (input for the GPU spike)
   - Torq runtimes in the image: `ls /usr/bin | grep -i -E "torq|iree"`, version strings via
     `grep -aoE "torq-runtime/[0-9][^ ]*" <binary>`
4. Plug in the USB microphone: `arecord -l` must list a USB card.
   Record the card number for `hes-tui --mic plughw:<card>,0`.

### 5.4 Torq on the new image
- **Kernel module:** the image ships the Torq 2.2.0 module. Torq 2.2.1 changes nothing on the board side (PyPI
  publishing, simulator XRAM, ONNX dependency pins), so keep the image's module. Only if `modinfo syna_npu` or
  `dmesg | grep -i torq` show something older than 2.2: load `astra-release/lib/syna_npu.ko` the Path A way
  (in memory) and verify. Install it under `/lib/modules/$(uname -r)/updates/` plus `depmod -a` only after it
  passed the gate *and* serial access is available.
- **Runtime for the canaries:** copy `astra-sl-torq-run-module` to `/home/root/torq-2.2.1/` as in Path A step 1.
  Alternatively install the wheel into a venv under `/home/root` (`pip install torq_runtime-2.2.1-…-aarch64.whl`).
  The app itself does not use either; it links `libiree_torq_runtime.a` built in 3.4.

### 5.5 Restore our files
Only what is still needed. The vendor examples and models may now come with the image or from `torq-examples`:
```
cd ~/board-backup/<date>
tar cf - moon sl2610-examples cmd.wav | ssh $BOARD 'cd /home/root && tar xf -'
```

---

## 6. Gate (decides whether the update is kept)

```
scripts/board-canaries.sh sl2610-iree:v2.2.1 /home/root/torq-2.2.1/astra-sl-torq-run-module ""
```

| Canary | Tests | Required |
|---|---|---|
| identity_1x4 | exec format loads, trivial graph | PASS |
| chain2_64 | 2 chained matmuls | PASS |
| **chain8_64** | 3+ dispatches, the silent-zeros case | PASS → multi-dispatch is trustworthy |
| softmax_64 | CSS kernels (exp/reduce) | PASS |

Then run the real graphs:
- the self-compiled Moonshine v2 encoder, golden-compared against CPU (cos-sim ≥ 0.999)
- the vendor encoder, exec v0: rejected by 2.x runtimes without the R7 patch; recompile or replace it from
  `torq-examples`

Outcome:
- **All PASS:** set `NPU_MULTIDISPATCH_TRUSTED=yes` and continue with section 7.
- **chain8/softmax still fail:** keep the new image if nothing regressed, keep `NPU_MULTIDISPATCH_TRUSTED=no`, and
  send the canary set + logs to Synaptics (`docs/synaptics-support` in the function-calling sample has a template).

---

## 7. Rebuild and verify the app

1. Recompile every board graph with `sl2610-iree:v2.2.1`: Moonshine NPU graphs, Gemma and the Moonshine CPU parts.
   CPU graphs must come from the Torq fork compiler too, because upstream IREE vmfbs are rejected
   (`required module features [Ch]`).
2. Re-materialize the cartridges, serve them (`scripts/serve-cartridges.sh`), and download and verify them on the
   board.
3. Link the app against the 2.2.1 `libiree_torq_runtime.a`, then `scripts/deploy-tui.sh`.
4. Golden set via WAV replay and live mic. Compare with the 3.1 baseline: latency per stage, RSS.

## 8. Update the lock

Edit `board/board.lock`:
- `BOARD_SDK=scarthgap_6.12_v2.6.0`, plus SDK revision, kernel and module path as audited
- `TORQ_VERSION=2.2.1`, `TORQ_COMPILER=sl2610-iree:v2.2.1`, `TORQ_IREE_BASE`
- `APP_RUNTIME`, `NPU_MULTIDISPATCH_TRUSTED`
- clear the `TARGET_*` lines

`scripts/board-audit.sh` must print OK. Commit the lock, scripts and log entry together.

---

## 9. Rollback

| Situation | Action |
|---|---|
| Before flashing | nothing on the board changed; delete `/home/root/torq-2.2.1` if wanted |
| Path A, module swapped in memory | `ssh $BOARD reboot`; if it hangs: power cycle; then `scripts/board-audit.sh` = OK |
| Path B, new image does not work | flash `sl2619_scarthgap_6.12_v2.4.0.zip` exactly as in 5.1–5.2, restore `/home/root` from the backup (5.5, plus `torq2-stable`, `torq2`, `demos`), redeploy the app built against the 2.0.0 runtime, `scripts/board-audit.sh` = OK |
| Host tools | `sl2610-iree:v2.0.0` and the 2.0.0 runtime work directory are kept side by side; `board/board.lock` in git is the record |

## 10. Troubleshooting

- **`astra-update` sees no device:** `lsusb` must show `06cb:019e` in USB boot mode, then `18d1:4ee0` (fastboot)
  once U-Boot runs. Check the cable carries data, use the Type-C port next to Ethernet, repeat the button sequence
  and hold USB_BOOT longer. Run with `sudo` or install a udev rule.
- **Board boots the old system or from SD:** remove the `SD_BOOT` jumper.
- **No network after flashing:** use the serial console; the MAC address and DHCP lease may have changed.
- **`pip install` of the 2.2.1 wheels fails on the host:** they need glibc ≥ 2.39. Use the docker image (3.3).
- **Canaries say VERSION:** the compiler and runtime are from different Torq versions; use the matching pair.
- **Root filesystem full:** never install into `/`; everything of ours lives in `/home/root`.

## References

- Astra SDK releases: <https://github.com/synaptics-astra/sdk/releases>
- Astra documentation, *Updating the eMMC* / *Running Astra Update*:
  <https://synaptics-astra.github.io/doc/v/latest/linux/index.html>
- SL2600 hardware manual (buttons, boot straps): <https://synaptics-astra.github.io/doc/v/latest/hw/sl2600.html>
- USB tool + boot images: <https://github.com/synaptics-astra/usb-tool>
- astra-update source: <https://github.com/synaptics-astra/astra-update>
- Torq releases and release notes: <https://github.com/synaptics-torq/torq-compiler/releases>
- Torq examples (Moonshine, Gemma on the NPU): <https://github.com/synaptics-torq/torq-examples>
