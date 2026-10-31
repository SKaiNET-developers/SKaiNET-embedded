# Board update: flash Astra SDK v2.6.0 (Torq 2.2)

One path, in order. Every step has a **Check**. Do not start the next step until the check passes. If a check
fails, follow its **If not** line, or stop and go to [Rollback](#rollback).

Steps marked **[at the board]** need someone at the board, with a USB-C cable and the buttons. All other steps run
on the host and can be done remotely.

What changes: the whole eMMC (boot chain, kernel, root filesystem and **`/home`**), from Astra v2.4.0 + Torq 2.0.0
to Astra v2.6.0 + Torq 2.2. Why and what was measured before: see [Appendix A](#appendix-a-background).

After that, [Part 4](#part-4-torq-230-after-the-flash) moves the host compiler and the app runtime to Torq 2.3.0
(2026-10-19). Part 4 changes no partition: it only adds files under `/home/root/torq-2.3.0`, and a reboot undoes
its one driver test. Do Parts 1–3 first, so a problem can be traced either to the image or to 2.3.0.

---

## Setup (every terminal you use)

```
export BOARD=root@192.168.3.26                                   # changes after the flash (step 9)
export HES=~/projects/neuroSKai/project/HES
export UPD=$HES/board-update                                     # downloads + flashing tool
export BAK=$HES/board-backup/20261017-preflash                   # backup of /home
export APP=$HES/SKaiNET-embedded/sl2610-home-even-smarter        # this repo
cd $APP
```

---

## Part 1: before going to the board (host)

The preparation was done on 2026-10-17. Each step below only checks that it is still in place.

### Step 1. Image and rollback image are downloaded

```
echo "9b15b709a45b0e4b5d6c1532277e2465aee960f862961805150e220fb7ac665f  $UPD/sl2619_scarthgap_6.12_v2.6.0.zip" | sha256sum -c
echo "7011b4fb6f4808834d80ff5f9096f6e596c87f68ee28c59cf6807cb082561de9  $UPD/sl2619_scarthgap_6.12_v2.4.0.zip" | sha256sum -c
```
**Check:** both lines end with `OK`.
**If not:** `scripts/board-update-download.sh $UPD image rollback`, then repeat.

### Step 2. The flashing tool holds the v2.6.0 image

```
ls $UPD/usb-tool/eMMCimg/TAG*
ls $UPD/usb-tool/eMMCimg/emmc_image_list $UPD/usb-tool/eMMCimg/home_s.subimg
$UPD/usb-tool/bin/linux/x86_64/astra-update --help | head -n 1
```
**Check:** the tag reads `…rootfs-20260929…` (the v2.6.0 build), both files exist, and the tool prints
`Astra Update Utility`.
**If not:** `cd $UPD/usb-tool && rm -rf eMMCimg && unzip -q ../sl2619_scarthgap_6.12_v2.6.0.zip`, then repeat.

### Step 3. The backup is complete and intact

```
(cd $BAK && sha256sum -c --quiet SHA256SUMS && echo backup OK)
ls $BAK/root/sl2610-examples/models/Synaptics/moonshine-tiny-bf16-torq/encoder.vmfb    # vendor NPU encoder
```
**Check:** prints `backup OK`, and the encoder file exists.

The encoder is Synaptics' prebuilt Moonshine-tiny NPU encoder. We cannot compile an equivalent ourselves (our own
Moonshine encoder crashes the Torq compiler, and self-compiled multi-dispatch graphs returned zeros on Torq 2.0). It
is not lost if the backup is: the same file (sha256 `28ef6573…`) is at Hugging Face revision `4a9b84ff11a1`, and
`scripts/board-update-download.sh $UPD moonshine` fetches it together with Synaptics' newer 2026-09-29 and
2026-10-19 builds (see Appendix B).

Anything written to the board's `/home` after 2026-10-17 18:00 is **not** in this backup. If you changed the board
since then, make a new one and use it as `$BAK`:
```
scripts/board-backup.sh $HES/board-backup/$(date +%Y%m%d-%H%M)
```
(`/home/root/hes-221` is missing on purpose: test binaries, rebuilt from the host.)

### Step 4. Host tools for afterwards

```
docker images sl2610-iree --format '{{.Tag}}'          # v2.2.1 and v2.0.0
ls native/iree-torq-runtime/dist-221/lib               # app runtime for Torq 2.2.1
ls build/packs                                         # the two packs the app uses
```
**Check:** both docker tags are listed, `dist-221/lib` has `libiree_torq_runtime.a`, and `build/packs` has the
`asr-…` and `nlu-…` packs.
**If not:** see [Appendix B](#appendix-b-rebuilding-the-host-side), then repeat.

### Step 5. The board is in the known state

```
scripts/board-audit.sh
```
**Check:** the last line is `OK: board matches the lock`. Keep this output: it is the "before" record.

---

## Part 2: flash (at the board)

### Step 6. [at the board] Shut down, connect USB

1. `ssh $BOARD poweroff`, wait 10 s.
2. Check the `SD_BOOT` jumper is **not** fitted.
3. Connect a data-capable USB-C cable from the host to the board's **USB 2.0 Type-C port next to Ethernet**.
4. Keep Ethernet connected. Power the board on if it went off.

**Check:** on the host, `lsusb | grep -E '06cb:019e|18d1:4ee0'` prints nothing yet. That is expected, because the
board is not in USB boot mode.

### Step 7. [at the board] Start the flashing tool

```
cd $UPD/usb-tool && ./update_emmc.sh
```
It asks for the sudo password, then waits for a device. Leave it running.

`update_emmc.sh` finds the board by its USB ID (`06cb:019e`). astra-update v2.0.4 renamed the chip `sl2610` to
`sl2619`: if you call `astra-update` directly, use `--chip sl2619 --board rdk`, not `sl2610`.

**Check:** the tool is waiting and has not exited with an error.

### Step 8. [at the board] Put the board into USB boot mode and flash

1. Press and hold **RESET (SW6)**.
2. While holding it, press and hold **USB_BOOT (SW7)**.
3. Release **RESET**, keep holding **USB_BOOT**.
4. Release **USB_BOOT** when the tool reports the device.

**Check (during the flash):** in a second terminal, `lsusb | grep -E '06cb:019e|18d1:4ee0'` first shows `06cb:019e`
(ROM USB boot), then `18d1:4ee0` (fastboot). The tool shows a progress bar per image: `preboot`, `key`, `tzk`,
`sysmgr`, `bl`, `boot`, `fastlogo`, `rootfs`, `home`.
**If not** (no device after 30 s): check the cable carries data and is in the port next to Ethernet, repeat the button
sequence, and hold USB_BOOT longer.

**Wait for the board to reset by itself.** The tool can report success before the last write has finished. Do
**not** press RESET or unplug power now: a manual reset at this point can leave invalid partitions.

**Check (after):** the tool reports success, and the board reboots on its own (LEDs, or Ethernet link comes back).
**If not:** the tool keeps its temp directory and prints its path. Keep `astradevicemanager.log` from there and
rerun step 7 with debug output: `sudo ./bin/linux/x86_64/astra-update -D -l stdout`.

### Step 9. [at the board] First login

1. Unplug the USB-C cable.
2. Find the board's IP address. It can change, because the MAC address or DHCP lease may differ. Look it up in the
   router's DHCP list, or try the old one.
3. On the host:
   ```
   export BOARD=root@<new-ip>
   ssh-keygen -R ${BOARD#root@}           # the board has a new host key
   ssh $BOARD 'cat /etc/build* | grep SYNA_SDK_REVISION; uname -r'
   ```

**Check:** `SYNA_SDK_REVISION` is newer than `202605280005_202605291432`, and the kernel is `6.12.62`. Login works
without a password, as before.
**If not:** no address or no login: the first boot after a flash can take up to 10 minutes before the network and
ssh come up (seen with the v2.4.0 images). Wait 10 minutes, power-cycle once with PWR IN, and wait again. Still
nothing: use a 3.3 V USB-serial adapter on UART0 of the 40-pin header (115200 8N1), and run `ip -4 addr show eth0`
there.

From here on, nobody needs to be at the board, except for step 15 (plugging in the microphone).

---

## Part 3: verify the new image (host)

### Step 10. Audit

```
scripts/board-audit.sh
```
DRIFT lines against the old lock are expected. **Check** these values in the output:
- `KERNEL=6.12.62`, `SDK_REVISION` from step 9
- `NPU_MODULE` present, `NPU_VERMAGIC` matches the kernel, `DEV_TORQ=present`
- `NPU_DEVFREQ` shows a frequency (new in this image)
- `ROOT_FREE` is well above the old 54 MB

Also record, for later:
- `VULKAN_ICD`: still empty, or now registered?
- `ssh $BOARD 'ls /usr/bin | grep -i -E "torq|iree"'`: the Torq tools the image ships

**If not** (`DEV_TORQ` missing or the module is absent): run
`ssh $BOARD 'dmesg | grep -i -E "torq|npu" | tail -n 20'`, keep the output and stop. The image is fine for the CPU
app, but the NPU part needs investigation.

### Step 11. Restore our files

The flash wiped `/home`. The new image's `/home/weston` is left alone.
```
tar -C $BAK -cf - root skainet-tinyllama knbuild | ssh $BOARD 'tar -C /home -xf -'
grep -v -E '  \./weston/' $BAK/SHA256SUMS | ssh $BOARD 'cd /home && sha256sum -c - | grep -v ": OK$"; echo restore checked'
```
**Check:** the second command prints only `restore checked`. Any other line names a file that did not arrive intact.
**If not:** rerun the `tar` line, then the check.

### Step 12. NPU gate (canaries)

The Torq 2.2.1 runtime tool came back with step 11 (`/home/root/torq-2.2.1/astra-sl-torq-run-module`).
```
scripts/board-canaries.sh sl2610-iree:v2.2.1 /home/root/torq-2.2.1/astra-sl-torq-run-module ""
```
**Check:**

| Canary | Required | Before the update |
|---|---|---|
| identity_1x4 | PASS | PASS |
| chain2_64 | PASS | PASS |
| **chain8_64** | PASS | no output |
| softmax_64 | PASS | no output / wrong values |

The update is kept either way, as long as nothing regressed against the "before" column. What changes is the
`NPU_MULTIDISPATCH_TRUSTED` value in step 16:
- all PASS: `yes`
- chain8_64 / softmax_64 still fail: `no`. Send the canary set and logs to the vendor support.

**If** identity_1x4 or chain2_64 fails, that is a regression: keep the output and stop here.

### Step 13. The app, unchanged

The app links its own runtime and runs on the CPU, so the binaries restored in step 11 should still work. This step
separates "the image broke something" from "the rebuild broke something". The packs were not in the backup
(regenerable), so install them first:
```
scripts/install-dev-packs.sh
ssh $BOARD 'cd /home/root/hes-tui && LD_LIBRARY_PATH=lib ./hes-asr-check.kexe ~/.homeevensmarter/cartridges/asr-* \
  /home/root/hes-wav/golden/g*.wav /home/root/hes-wav/ab_u*.wav /home/root/hes-wav/beckett5s.wav'
ssh $BOARD 'cd /home/root/hes-tui && LD_LIBRARY_PATH=lib ./hes-nlu-check.kexe ~/.homeevensmarter/cartridges/nlu-*'
ssh $BOARD 'cd /home/root/hes-tui && LD_LIBRARY_PATH=lib ./hes-app-check.kexe --verify \
  /home/root/hes-wav/golden/g01.wav /home/root/hes-wav/golden/g07.wav /home/root/hes-wav/golden/g10.wav'
```
**Check** against the numbers from 2026-10-17 (int8 packs, NLU chunk 4):

| | before the update |
|---|---|
| ASR | RTF ≈ 0.73, finals 16/16 as recorded |
| NLU | ≈ 610 ms/token, golden 11/12, p50 ≈ 8.9 s |
| app g01 / g07 / g10 | 16.2 / 11.2 / 9.4 s per command, 3/3 tools |

The results must be the same. Times may differ a little, because the CPU clocks can change with DVFS.
**If not:** the binary does not start (glibc, `libasound`, `libcrypt`): note the error, it is an image change.
`ls -l /lib/libcrypt*` must still show `libcrypt.so.2`. Different results: keep the output and stop.

### Step 14. The app against the Torq 2.2.1 runtime

```
./gradlew -PireeDist=native/iree-torq-runtime/dist-221 \
  :tui:linkAsrCheckReleaseExecutableLinuxArm64 :tui:linkNluCheckReleaseExecutableLinuxArm64 \
  :tui:linkAppCheckReleaseExecutableLinuxArm64 :tui:linkReleaseExecutableLinuxArm64
scripts/deploy-tui.sh --no-build
for b in hes-asr-check hes-nlu-check hes-app-check; do
  cat tui/build/bin/linuxArm64/*/$b.kexe | ssh $BOARD "cat > /home/root/hes-tui/$b.kexe && chmod +x /home/root/hes-tui/$b.kexe"; done
```
Then rerun the three checks of step 13.
**Check:** the same results as in step 13.

### Step 15. [at the board] Microphone

Plug in the USB microphone.
```
ssh $BOARD 'arecord -l'
```
**Check:** a `USB` card is listed next to `klamathasoc`. Note its card number, for
`ssh -t $BOARD /home/root/hes-tui/hes-tui --mic plughw:<card>,0`.

### Step 16. Record the new state

Edit `board/board.lock`:
- `BOARD_SDK=scarthgap_6.12_v2.6.0`, and the SDK revision, kernel and module path from step 10
- `TORQ_VERSION=2.2.1`, `TORQ_COMPILER=sl2610-iree:v2.2.1`
- `APP_RUNTIME` (2.2.1 if step 14 passed), and `NPU_MULTIDISPATCH_TRUSTED` from step 12
- delete the `TARGET_*` lines

```
scripts/board-audit.sh
```
**Check:** `OK: board matches the lock`. Commit the lock together with a log entry holding the outputs of steps 10,
12 and 13.

The image update is done. Continue with Part 4.

---

## Part 4: Torq 2.3.0 (after the flash)

Torq 2.3.0 (2026-10-19) brings faster tiling, slicing and LRAM allocation, an opt-in stream-schedule prefetch, and
SR200 support. It is still on the IREE 3.10 fork (`synaptics-torq/iree@585aecc`), so the app's runtime ABI is
unchanged. Its `astra-release` ships its own `syna_npu.ko` (vermagic `6.12.62`) next to the runner, while the v2.6.0
image carries the 2.2.0 module. Step 19 finds out whether the 2.3.0 runtime needs the 2.3.0 module.

Every step here is reversible: delete `/home/root/torq-2.3.0`, reboot, and the board is back at the end of Part 3.

### Step 17. Host tools for 2.3.0

```
(cd $UPD/torq-2.3.0 && ls astra-release/lib/syna_npu.ko astra-release/tools/astra-sl-torq-run-module)
docker images sl2610-iree --format '{{.Tag}}'          # now also v2.3.0
cat native/iree-torq-runtime/dist-230/BUILD-INFO
```
**Check:** both files exist, `v2.3.0` is listed, and `BUILD-INFO` reads `torq: v2.3.0`,
`iree: 585aeccf0b7873983b38e5db8b28825c88699e65`, `exec_v0_patch: no`.
**If not:** see [Appendix B](#appendix-b-rebuilding-the-host-side) (the 2.3.0 rows), then repeat.

### Step 18. Put the 2.3.0 runner on the board

```
ssh $BOARD 'mkdir -p /home/root/torq-2.3.0'
tar -C $UPD/torq-2.3.0/astra-release -cf - . | ssh $BOARD 'tar -C /home/root/torq-2.3.0 -xf -'
ssh $BOARD '/home/root/torq-2.3.0/tools/astra-sl-torq-run-module --help 2>&1 | head -n 3'
```
**Check:** the runner prints its usage. It needs glibc 2.38 or newer, which the v2.6.0 image has.
**If not** (`GLIBC_2.38 not found`): the image is older than v2.6.0. Go back to step 9.

### Step 19. NPU gate on 2.3.0 (canaries)

With the image's own driver first:
```
scripts/board-canaries.sh sl2610-iree:v2.3.0 /home/root/torq-2.3.0/tools/astra-sl-torq-run-module ""
```
**Check:** every canary that passed in step 12 passes here too.

**If** canaries fail with `VERSION` (`does not match expected version`), try the 2.3.0 driver. This lasts only until
the next reboot:
```
ssh $BOARD 'modinfo -F vermagic /home/root/torq-2.3.0/lib/syna_npu.ko; uname -r'      # same kernel version
ssh $BOARD 'rmmod syna_npu && insmod /home/root/torq-2.3.0/lib/syna_npu.ko && ls -l /dev/torq'
scripts/board-canaries.sh sl2610-iree:v2.3.0 /home/root/torq-2.3.0/tools/astra-sl-torq-run-module ""
```
Write down which driver the result is for. If 2.3.0 works only with its own module, stop after this step: keeping
that module across reboots is a change to the root filesystem and a separate decision. `ssh $BOARD reboot` returns
to the image driver. If the board stops responding (the v2.6.0 release notes list a hang during the Torq demo as
known issue 39459), power-cycle it.

**If** anything regresses against step 12: stay on 2.2.1. Run `ssh $BOARD reboot` and
`ssh $BOARD rm -rf /home/root/torq-2.3.0`. The board is at the end of Part 3, and Part 4 ends here.

### Step 20. The app against the Torq 2.3.0 runtime

```
./gradlew -PireeDist=native/iree-torq-runtime/dist-230 \
  :tui:linkAsrCheckReleaseExecutableLinuxArm64 :tui:linkNluCheckReleaseExecutableLinuxArm64 \
  :tui:linkAppCheckReleaseExecutableLinuxArm64 :tui:linkReleaseExecutableLinuxArm64
scripts/deploy-tui.sh --no-build
for b in hes-asr-check hes-nlu-check hes-app-check; do
  cat tui/build/bin/linuxArm64/*/$b.kexe | ssh $BOARD "cat > /home/root/hes-tui/$b.kexe && chmod +x /home/root/hes-tui/$b.kexe"; done
```
Then rerun the three checks of step 13.
**Check:** the same results as in steps 13 and 14.
**If not:** go back to the 2.2.1 binaries with step 14, and keep the output.

### Step 21. Record 2.3.0

Edit `board/board.lock`:
- `TORQ_VERSION=2.3.0`, `TORQ_COMPILER=sl2610-iree:v2.3.0`, `TORQ_IREE_BASE=3.10.0.dev` (`synaptics-torq/iree@585aecc`)
- `APP_RUNTIME` (2.3.0, `dist-230`), and `NPU_MULTIDISPATCH_TRUSTED` from step 19
- in a comment on `BOARD_NPU_MODULE`: which driver step 19 passed with (image 2.2.0 or Torq 2.3.0)
- delete the `NEXT_TORQ_VERSION` line

```
scripts/board-audit.sh
```
**Check:** `OK: board matches the lock`. Commit the lock together with a log entry holding the outputs of steps 19
and 20.

Next: recompile the board graphs with `sl2610-iree:v2.3.0` and, if step 19 passed, move the Moonshine encoder to the
NPU.

---

## Rollback

Use this when the new image does not boot, has no network, or regressed in step 12 or 13.

1. Unpack the v2.4.0 image into the flashing tool:
   ```
   cd $UPD/usb-tool && rm -rf eMMCimg && unzip -q ../sl2619_scarthgap_6.12_v2.4.0.zip
   ls eMMCimg/TAG*
   ```
   **Check:** the tag reads `…rootfs-20260530013802…` (the v2.4.0 build).
2. **[at the board]** Steps 6 to 9 exactly as before.
   **Check:** `SYNA_SDK_REVISION` = `202605280005_202605291432`.
3. Step 11 (restore), then `scripts/install-dev-packs.sh`. Then redeploy the app built against the 2.0.0 runtime:
   `./gradlew` without `-PireeDist`, followed by `scripts/deploy-tui.sh`.
4. ```
   git checkout board/board.lock && scripts/board-audit.sh
   ```
   **Check:** `OK: board matches the lock`, and step 13 shows the "before" numbers.

The 2.0.0 compiler (`sl2610-iree:v2.0.0`) and the 2.0.0 app runtime (`native/iree-torq-runtime/dist`) stay on the
host next to the 2.2.1 ones.

---

## Appendix A: background

The board image, the Torq NPU kernel module (`syna_npu.ko`), the Torq runtime on the board and the Torq compiler on
the host are versioned **as one set**. A model compiled by one Torq version and run on another either fails to load
(`version N does not match expected version M`) or, worse, loads and returns zeros.

| Astra SDK (board image) | Released | Kernel | Torq |
|---|---|---|---|
| scarthgap_6.12_v2.4.0 | 2026-05-30 | 6.12.62 | 2.0.0, **before** |
| scarthgap_6.12_v2.6.0 | 2026-09-30 | 6.12.62 | 2.2.0 in the image, 2.2.1 framework, **after** |
| scarthgap_6.12_v2.6.1 | 2026-10-13 | 6.12.62 | coralboard variants only, no plain `sl2619` image |
| (no new image) | 2026-10-19 | 6.12.62 | Torq 2.3.0 release: runner + `syna_npu.ko` in `astra-release`, Part 4 |

Google's "Coral Compiler v1.0" (`google-coral/coralnpu-compiler`, announced 2026-10-20) is a separate compiler with
its own `coralnpu` HAL. It is not part of this set; the SL2619 path stays on Torq. Upstream IREE 3.13 is not
either: vmfbs from upstream IREE 3.11 and newer are rejected by the Torq-fork runtime (`required module features
[Ch]`).

Why: NPU graphs with three or more chained dispatches return zeros or nothing on Torq 2.0 (the self-compiled
Moonshine encoder, canaries `chain8_64` / `softmax_64`). Torq 2.1/2.2 bring:
- a fix for "runtime auto mode returning zeros or SIGSEGV"
- tile-and-fuse improvements, LayerNorm on the NSS
- `torq-decode-llm`
- NPU devfreq (800 MHz → 1 GHz)

Astra v2.6.0 adds NPU DVFS, eMMC tuning and newer firmware.

Measured on the old image (2026-10-16/17):
- The 2.2.1 compiler and runtime on the **old driver** do not fix multi-dispatch: chain8_64 and softmax_64 give
  no output. The 2.2 driver from the new image is the open question that step 12 answers.
- The app linked against the 2.2.1 runtime already runs the current packs unchanged on the old image: same ASR
  finals, NLU 610–640 ms/token.

The image zip holds `eMMCimg/` with `preboot`, `key`, `tzk`, `sysmgr`, `bl`, `boot`, `fastlogo`, `rootfs_s.subimg.*`
and **`home_s.subimg` (partition 18 = `/home`)**. That is why the flash wipes `/home`. The USB boot image set used is
`astra-usbboot-images/sl2619` (board `rdk`, DDR4, 2 GB).

## Appendix B: rebuilding the host side

Only needed if step 4 fails.

| What | How |
|---|---|
| downloads | `scripts/board-update-download.sh $UPD all` (sets: `image rollback torq usb-tool`), each verified against the release sha256 |
| compiler image | `docker build -f board/docker/Dockerfile.torq-compiler -t sl2610-iree:v2.2.1 --build-arg TORQ_VERSION=2.2.1 --build-arg MANYLINUX=2_39 --build-arg BASE=python:3.12-slim-trixie board/docker` |
| app runtime | `TORQ_TAG=v2.2.1 native/iree-torq-runtime/build-runtime.sh native/iree-torq-runtime/work-221`, then `OUT=native/iree-torq-runtime/dist-221 native/iree-torq-runtime/build-dist.sh native/iree-torq-runtime/work-221` |
| packs | `scripts/stage-asr-pack.sh` and `scripts/stage-nlu-pack.sh`, after the blueprint recipes (see the script headers) |
| 2.3.0 downloads | `scripts/board-update-download.sh $UPD torq230` |
| Moonshine NPU encoder | `scripts/board-update-download.sh $UPD moonshine` (three pinned Hugging Face revisions, below) |
| 2.3.0 compiler image | `docker build -f board/docker/Dockerfile.torq-compiler -t sl2610-iree:v2.3.0 --build-arg TORQ_VERSION=2.3.0 --build-arg MANYLINUX=2_39 --build-arg BASE=python:3.12-slim-trixie board/docker` |
| 2.3.0 app runtime | `TORQ_TAG=v2.3.0 native/iree-torq-runtime/build-runtime.sh native/iree-torq-runtime/work-230`, then `OUT=native/iree-torq-runtime/dist-230 native/iree-torq-runtime/build-dist.sh native/iree-torq-runtime/work-230` |

Downloads, pinned:

| File | Source | sha256 |
|---|---|---|
| `sl2619_scarthgap_6.12_v2.6.0.zip` (725 MB) | [Astra SDK v2.6.0](https://github.com/synaptics-astra/sdk/releases/tag/scarthgap_6.12_v2.6.0) | `9b15b709a45b0e4b5d6c1532277e2465aee960f862961805150e220fb7ac665f` |
| `sl2619_scarthgap_6.12_v2.4.0.zip` (705 MB) | [Astra SDK v2.4.0](https://github.com/synaptics-astra/sdk/releases/tag/scarthgap_6.12_v2.4.0) | `7011b4fb6f4808834d80ff5f9096f6e596c87f68ee28c59cf6807cb082561de9` |
| `astra-release.tar.gz` | [Torq v2.2.1](https://github.com/synaptics-torq/torq-compiler/releases/tag/v2.2.1) | `f71693087ec336674366e254197fb622fb939e353dfd34c745e766c1edfdf433` |
| `torq_runtime-2.2.1-cp312-cp312-manylinux_2_39_aarch64.whl` | Torq v2.2.1 | `7c17c5c627670314119f3a868f0fa6a877f718ab30efbfe8ac2088b3fe93de4e` |
| `torq_compiler-2.2.1-cp312-cp312-manylinux_2_39_x86_64.whl` | Torq v2.2.1 | `37b8d56f6e148db0dda229b7b231cc6ad06ab699639d9cc3ac459cb9680da259` |
| usb-tool | [`synaptics-astra/usb-tool`](https://github.com/synaptics-astra/usb-tool) @ `91d4000af92f` (Git LFS) | pinned commit |
| `astra-release.tar.gz` | [Torq v2.3.0](https://github.com/synaptics-torq/torq-compiler/releases/tag/v2.3.0) | `0d3ecf53dfd7906362742cd3ad6188b36fb55886380b4d4f18b0b6f46d75cfa4` |
| `torq_runtime-2.3.0-cp312-cp312-manylinux_2_39_aarch64.whl` | Torq v2.3.0 | `bdfe92470f4ac3f95bbe6ad11e6d9987c4e615762f29d32ca6a8ea8d723be6d2` |
| `torq_compiler-2.3.0-cp312-cp312-manylinux_2_39_x86_64.whl` | Torq v2.3.0 | `b809e13f1b3d16784bea5e63e12b2846d718f1dba55f0248ed46278f6bd51d7d` |
| `encoder.vmfb` (14.5 MB, the board's copy, Torq 2.0) | [HF `Synaptics/moonshine-tiny-bf16-torq` @ `4a9b84ff11a1`](https://huggingface.co/Synaptics/moonshine-tiny-bf16-torq/tree/4a9b84ff11a1) | `28ef65731f1790f949284c0f4d546b03df3144aea28902d83470c5dee7d8500a` |
| `encoder.vmfb` (19.2 MB, 2026-09-29, likely Torq 2.2) | [HF @ `ef128b544c7b`](https://huggingface.co/Synaptics/moonshine-tiny-bf16-torq/tree/ef128b544c7b) | `71ff97581260de19774d16742d3cb237ba6c8edc0ae488f8c401826b1e4d4492` |
| `encoder.vmfb` (19.1 MB, 2026-10-19, likely Torq 2.3.0) | [HF @ `d3be4736dc4e`](https://huggingface.co/Synaptics/moonshine-tiny-bf16-torq/tree/d3be4736dc4e) | `e9f3910adeebaea81280308cc34a19d39fad0a29a8b5efdb9fdce2424b2d42fa` |

## References

- Astra documentation, *Updating the eMMC*: <https://synaptics-astra.github.io/doc/v/latest/linux/index.html>
- SL2600 hardware manual (buttons, boot straps): <https://synaptics-astra.github.io/doc/v/latest/hw/sl2600.html>
- USB tool: <https://github.com/synaptics-astra/usb-tool>
- Torq releases: <https://github.com/synaptics-torq/torq-compiler/releases>
- Torq examples: <https://github.com/synaptics-torq/torq-examples>
