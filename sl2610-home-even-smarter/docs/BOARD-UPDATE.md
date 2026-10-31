# Board update: Home Even Smarter on Astra v2.6.0 + Torq 2.3.0

One path, in order: flash the new image, put the Torq 2.3.0 release tools on the board, and run Home Even Smarter
built on the current SKaiNET libraries and cartridge blueprints. Every step has a **Check**. Do not start the next
step until the check passes. If a check fails, follow its **If not** line, or go to [Rollback](#rollback).

| | |
|---|---|
| image | Astra SDK scarthgap_6.12_v2.6.0, kernel 6.12.62 |
| NPU tools | Torq 2.3.0 (IREE 3.10 fork, `synaptics-torq/iree@585aecc`) |
| blueprints | SKaiNET-cartridge-blueprints 0.2.2, cartridge spec v0.4 |
| libraries | SKaiNET 0.62.1, SKaiNET-transformers 0.63.0, Kotlin 2.4.21 |

Steps marked **[at the board]** need someone at the board, with a USB-C cable and the buttons. All other steps run
on the host and can be done remotely.

**Clean slate.** The flash rewrites the whole eMMC (boot chain, kernel, root filesystem and **`/home`**), and nothing
from the old board is restored afterwards. The only thing taken from the old backup is the folder of test recordings
(step 13). Everything Home Even Smarter needs is built on the host in Part 1. Why and what was measured before: see
[Appendix A](#appendix-a-background).

---

## Setup (every terminal you use)

```
export BOARD=root@192.168.3.26                                   # changes after the flash (step 9)
export HES=~/projects/neuroSKai/project/HES
export UPD=$HES/board-update                                     # downloads + flashing tool
export BLU=$HES/SKaiNET-cartridge-blueprints                     # blueprints 0.2.2
export APP=$HES/SKaiNET-embedded/sl2610-home-even-smarter        # this repo
cd $APP
```

The commands use `${NAME}` braces where a colon follows a variable, so they also work in zsh.

---

## Part 1: build everything on the host

### Step 1. Download the image, the Torq 2.3.0 release and the flashing tool

```
scripts/board-update-download.sh $UPD image torq230 usb-tool
```
Each file is checked against the sha256 published with its release. Files already present and valid are skipped.

**Check:** every line reads `ok`, the `astra-release` line lists `lib/syna_npu.ko` and
`tools/astra-sl-torq-run-module`, and the last line is `done`.
**If not:** `CHECKSUM MISMATCH`: delete that file and rerun. `git-lfs is required`: `sudo apt install git-lfs`, then
rerun.

### Step 2. Load the v2.6.0 image into the flashing tool

```
cd $UPD/usb-tool && rm -rf eMMCimg && unzip -q ../sl2619_scarthgap_6.12_v2.6.0.zip && cd $APP
ls $UPD/usb-tool/eMMCimg/TAG*
$UPD/usb-tool/bin/linux/x86_64/astra-update --help | head -n 1
```
**Check:** the tag reads `…rootfs-20260929…` (the v2.6.0 build), and the tool prints `Astra Update Utility`.

### Step 3. Torq 2.3.0 compiler and app runtime

```
docker images sl2610-iree --format '{{.Tag}}'
cat native/iree-torq-runtime/dist-230/BUILD-INFO
```
**Check:** `v2.3.0` is listed, and `BUILD-INFO` reads `torq: v2.3.0`,
`iree: 585aeccf0b7873983b38e5db8b28825c88699e65`, `exec_v0_patch: no`.
**If not:** build them (the runtime build cross-compiles the Torq fork of IREE for the board and takes a while):
```
docker build -f board/docker/Dockerfile.torq-compiler -t sl2610-iree:v2.3.0 \
  --build-arg TORQ_VERSION=2.3.0 --build-arg MANYLINUX=2_39 \
  --build-arg BASE=python:3.12-slim-trixie board/docker
TORQ_TAG=v2.3.0 native/iree-torq-runtime/build-runtime.sh native/iree-torq-runtime/work-230
OUT=native/iree-torq-runtime/dist-230 native/iree-torq-runtime/build-dist.sh native/iree-torq-runtime/work-230
```

### Step 4. Model packs from blueprints 0.2.2

Run the two blueprint recipes for the board target (`cpu-sl26xx`), then stage the packs the app installs.
```
cd $BLU
P=$APP/cartridges/profiles
A=:blueprints:asr-moonshine-v2-streaming-iree
./gradlew ${A}:compileEnFrontend ${A}:compileEnEncoder ${A}:compileEnAdapter \
  ${A}:compileEnPrefill ${A}:compileEnWithPast ${A}:weightsEn \
  -Pprofile=$P/asr-moonshine.cpu-sl26xx.en.json
N=:blueprints:nlu-functiongemma-270m-iree
./gradlew ${N}:compileWithPast ${N}:compilePrefillWithPast ${N}:compilePrefillAt \
  ${N}:parametersWithPast ${N}:parametersPrefillWithPast ${N}:parametersPrefillAt \
  -Pprofile=$P/nlu-functiongemma.cpu-sl26xx.home.json
cd $APP
scripts/stage-asr-pack.sh
scripts/stage-nlu-pack.sh
ls build/packs
```
**Check:** both Gradle runs end with `BUILD SUCCESSFUL`, and `build/packs` holds
`asr-moonshine-v2-streaming-iree-cpu-sl26xx-dev` and `nlu-functiongemma-270m-iree-cpu-sl26xx-dev`.
**If not** (compiler image not found): build the pack compiler image from `$HES/skainet-embedded-vendors`:
`TORQ_VERSION=2.3.0 TORQ_MANYLINUX=2_39 TORQ_BASE_IMAGE=python:3.12-slim-trixie synaptics-torq/docker/build.sh`.

The board graphs are CPU graphs compiled by the blueprints' pinned Torq-fork compiler (`iree-tools-torq` in each
`blueprint.json`). Use the blueprints branch `chore/torq-2.3.0`, which pins `skainet/iree-compiler-torq:2.3.0`, so
compiler and board runtime come from the same Torq release; blueprints 0.2.2 itself still pins 2.0.0 (same IREE 3.10
fork base, so it should load too, but that pairing is the less certain one). The first run downloads the model
sources named in the profiles.

### Step 5. Build Home Even Smarter against the Torq 2.3.0 runtime

```
./gradlew -PireeDist=native/iree-torq-runtime/dist-230 \
  :tui:linkReleaseExecutableLinuxArm64 :tui:linkAsrCheckReleaseExecutableLinuxArm64 \
  :tui:linkNluCheckReleaseExecutableLinuxArm64 :tui:linkAppCheckReleaseExecutableLinuxArm64
```
**Check:** `BUILD SUCCESSFUL`, and `ls tui/build/bin/linuxArm64/*/*.kexe` lists `hes-tui`, `hes-asr-check`,
`hes-nlu-check` and `hes-app-check`.

---

## Part 2: flash (at the board)

### Step 6. [at the board] Shut down, connect USB

1. `ssh $BOARD poweroff`, wait 10 s.
2. Check the `SD_BOOT` jumper is **not** fitted.
3. Connect a data-capable USB-C cable from the host to the board's **USB 2.0 Type-C port next to Ethernet**.
4. Keep Ethernet connected. Power the board on (PWR IN) if it went off.

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

The two buttons are on the right edge of the I/O board: **RESET (SW6)** lower, **USB_BOOT (SW7)** upper.

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
2. Find the board's IP address in the router's DHCP list, or try the old one. It can change, because the MAC
   address or DHCP lease may differ.
3. On the host:
   ```
   export BOARD=root@<new-ip>
   ssh-keygen -R ${BOARD#root@}           # the board has a new host key
   ssh $BOARD 'cat /etc/build* | grep SYNA_SDK_REVISION; uname -r'
   ```

**Check:** `SYNA_SDK_REVISION` is newer than `202605280005_202605291432`, the kernel is `6.12.62`, and login works
without a password.
**If not:** no address or no login: the first boot after a flash can take up to 10 minutes before the network and
ssh come up. Wait 10 minutes, power-cycle once with PWR IN, and wait again. Still nothing: use a 3.3 V USB-serial
adapter on UART0 of the 40-pin header (115200 8N1), and run `ip -4 addr show eth0` there.

From here on, nobody needs to be at the board, except for step 15 (the microphone).

---

## Part 3: Torq 2.3.0 and Home Even Smarter on the board (host)

Steps 10–14 also run in one go, with each step's output saved to `build/bringup/<time>/` for step 16:
```
scripts/board-bringup.sh            # or a subset, e.g. scripts/board-bringup.sh 13 14
```
It stops after a failed step 11 or 13, records audit drift and canary failures without stopping, and never swaps the
NPU driver. Then check its output against the steps below.

### Step 10. Audit the new image

```
scripts/board-audit.sh
```
DRIFT lines against the old lock are expected; step 16 updates the lock. **Check** these values in the output:
- `KERNEL=6.12.62`, and the `SDK_REVISION` from step 9
- `NPU_MODULE` present, `NPU_VERMAGIC` matches the kernel, `DEV_TORQ=present`
- `NPU_DEVFREQ` shows a frequency

**If not** (`DEV_TORQ` or the module is missing): keep this output and
`ssh $BOARD 'dmesg | grep -i -E "torq|npu" | tail -n 20'`. The CPU app can still run (go on with step 13), but the
NPU steps need investigating.

### Step 11. Put the Torq 2.3.0 release tools on the board

```
ssh $BOARD 'mkdir -p /home/root/torq-2.3.0'
tar -C $UPD/torq-2.3.0/astra-release -cf - . | ssh $BOARD 'tar -C /home/root/torq-2.3.0 -xf -'
ssh $BOARD '/home/root/torq-2.3.0/tools/astra-sl-torq-run-module --help 2>&1 | head -n 3'
```
**Check:** the runner prints its usage. It needs glibc 2.38 or newer, which the v2.6.0 image has.
**If not** (`GLIBC_2.38 not found`): the image is older than v2.6.0. Go back to step 9.

### Step 12. NPU gate: canaries on Torq 2.3.0

With the image's own NPU driver first:
```
scripts/board-canaries.sh sl2610-iree:v2.3.0 /home/root/torq-2.3.0/tools/astra-sl-torq-run-module ""
```

| Canary | Goal | On the old board |
|---|---|---|
| identity_1x4 | PASS | PASS |
| chain2_64 | PASS | PASS |
| **chain8_64** | PASS | no output |
| softmax_64 | PASS | no output / wrong values |

**Check:** `identity_1x4` and `chain2_64` pass. If all four pass, multi-dispatch NPU graphs are trusted
(`NPU_MULTIDISPATCH_TRUSTED=yes` in step 16), and the Moonshine encoder can move to the NPU later. If chain8_64 /
softmax_64 still fail, it is `no`: send the canary set and logs to the vendor support.

**If** canaries fail with `VERSION` (`does not match expected version`), try the 2.3.0 driver from the release. It
stays loaded only until the next reboot:
```
ssh $BOARD 'modinfo -F vermagic /home/root/torq-2.3.0/lib/syna_npu.ko; uname -r'      # same kernel version
ssh $BOARD 'rmmod syna_npu && insmod /home/root/torq-2.3.0/lib/syna_npu.ko && ls -l /dev/torq'
scripts/board-canaries.sh sl2610-iree:v2.3.0 /home/root/torq-2.3.0/tools/astra-sl-torq-run-module ""
```
Write down which driver passed. Keeping the 2.3.0 module across reboots is a change to the root filesystem and a
separate decision. `ssh $BOARD reboot` returns to the image driver. If the board stops responding (the v2.6.0
release notes list a hang during the Torq demo as known issue 39459), power-cycle it. The app itself runs on the
CPU, so steps 13–15 do not depend on this gate.

### Step 13. Install Home Even Smarter, its packs and the test recordings

```
scripts/deploy-tui.sh --no-build
for b in hes-asr-check hes-nlu-check hes-app-check; do
  cat tui/build/bin/linuxArm64/*/$b.kexe | ssh $BOARD "cat > /home/root/hes-tui/$b.kexe && chmod +x /home/root/hes-tui/$b.kexe"; done
BOARD=$BOARD scripts/install-dev-packs.sh
tar -C $HES/board-backup/20261017-preflash/root -cf - hes-wav | ssh $BOARD 'tar -C /home/root -xf -'
ssh $BOARD 'ls /home/root/hes-tui /home/root/.homeevensmarter/cartridges; ls /home/root/hes-wav/golden | head -n 3'
```
`deploy-tui.sh` also creates the private `lib/` directory with the `libcrypt.so.1` link the binaries need.
`install-dev-packs.sh` defaults to the old board address, so pass `BOARD` explicitly.

**Check:** `hes-tui` holds the four `.kexe` files and `lib/`, the store lists both packs, and the golden recordings
(`g01.wav` …) are there.

### Step 14. Run the three checks

```
ssh $BOARD 'cd /home/root/hes-tui && LD_LIBRARY_PATH=lib ./hes-asr-check.kexe ~/.homeevensmarter/cartridges/asr-* \
  /home/root/hes-wav/golden/g*.wav /home/root/hes-wav/ab_u*.wav /home/root/hes-wav/beckett5s.wav'
ssh $BOARD 'cd /home/root/hes-tui && LD_LIBRARY_PATH=lib ./hes-nlu-check.kexe ~/.homeevensmarter/cartridges/nlu-*'
ssh $BOARD 'cd /home/root/hes-tui && LD_LIBRARY_PATH=lib ./hes-app-check.kexe --verify \
  /home/root/hes-wav/golden/g01.wav /home/root/hes-wav/golden/g07.wav /home/root/hes-wav/golden/g10.wav'
```

| | Goal | Last measured (old image, 2026-10-17) |
|---|---|---|
| ASR | finals as recorded | RTF ≈ 0.73, 16/16 |
| NLU | golden 11/12 or better | ≈ 610 ms/token, p50 ≈ 8.9 s |
| app g01 / g07 / g10 | 3/3 tools | 16.2 / 11.2 / 9.4 s per command |

**Check:** the goals are met. The timings are for reference: the new image has NPU and CPU DVFS, so they move a
little.
**If not:** the binary does not start: `ls -l /home/root/hes-tui/lib` must show the `libcrypt.so.1` link to the
image's `libcrypt.so.2`; rerun step 13. A pack fails to load (module or version error): keep the output. It is the
compiler/runtime pairing named in step 4.

### Step 15. [at the board] Microphone, then the real thing

Plug in the USB microphone.
```
ssh $BOARD 'arecord -l'
ssh -t $BOARD 'cd /home/root/hes-tui && LD_LIBRARY_PATH=lib ./hes-tui.kexe --mic plughw:<card>,0'
```
**Check:** a `USB` card is listed next to `klamathasoc`. Put its number in the second command. Say "turn on the
light": the app shows the transcript and calls the tool.

### Step 16. Record the new board state

Edit `board/board.lock`:
- `BOARD_SDK=scarthgap_6.12_v2.6.0`, plus the SDK revision, kernel and module path from step 10
- `TORQ_VERSION=2.3.0`, `TORQ_COMPILER=sl2610-iree:v2.3.0`, `TORQ_IREE_BASE=3.10.0.dev` (`synaptics-torq/iree@585aecc`)
- `APP_RUNTIME` (2.3.0, `dist-230`), and `NPU_MULTIDISPATCH_TRUSTED` from step 12
- a comment on `BOARD_NPU_MODULE` saying which driver step 12 passed with (image 2.2.0 or Torq 2.3.0)
- delete the `TARGET_*` and `NEXT_TORQ_VERSION` lines

```
scripts/board-audit.sh
```
**Check:** `OK: board matches the lock`. Commit the lock together with a log entry holding the outputs of steps 10,
12 and 14.

Next, if all four canaries passed in step 12: move the Moonshine encoder to the NPU. Our own SKaiNET encoder (the
blueprint's `export/en/encoder.mlir`) compiles for the NPU with Torq 2.3.0 when `--torq-disable-css` is added: 15
Torq executables, no CPU fallback modules. Without that flag the attention softmax fails with "CSS program
allocations … exceed maximum CSS stack size" (the June weight-format crash of Torq 2.0 is gone). Whether its output
is correct is the next board test.

---

## Rollback

Use this only when v2.6.0 does not boot or has no network.

1. Download the v2.4.0 image and load it into the flashing tool:
   ```
   scripts/board-update-download.sh $UPD rollback
   cd $UPD/usb-tool && rm -rf eMMCimg && unzip -q ../sl2619_scarthgap_6.12_v2.4.0.zip
   ls eMMCimg/TAG*
   ```
   **Check:** the tag reads `…rootfs-20260530013802…` (the v2.4.0 build).
2. **[at the board]** Steps 6 to 9 exactly as before.
   **Check:** `SYNA_SDK_REVISION` = `202605280005_202605291432`.
3. ```
   git checkout board/board.lock && scripts/board-audit.sh
   ```

The 2.0.0 compiler (`sl2610-iree:v2.0.0`) and the 2.0.0 app runtime (`native/iree-torq-runtime/dist`) stay on the
host, next to the 2.2.1 and 2.3.0 ones, for building the app for the old image.

---

## Appendix A: background

The board image, the Torq NPU kernel module (`syna_npu.ko`), the Torq runtime on the board and the Torq compiler on
the host are versioned **as one set**. A model compiled by one Torq version and run on another either fails to load
(`version N does not match expected version M`) or, worse, loads and returns zeros.

| Astra SDK (board image) | Released | Kernel | Torq |
|---|---|---|---|
| scarthgap_6.12_v2.4.0 | 2026-05-30 | 6.12.62 | 2.0.0, **before** |
| scarthgap_6.12_v2.6.0 | 2026-09-30 | 6.12.62 | 2.2.0 in the image, **after** |
| scarthgap_6.12_v2.6.1 | 2026-10-13 | 6.12.62 | coralboard variants only, no plain `sl2619` image |
| (no new image) | 2026-10-19 | 6.12.62 | Torq 2.3.0 release: runner + `syna_npu.ko` in `astra-release`, steps 11–12 |

Torq 2.3.0 brings faster tiling, slicing and LRAM allocation, an opt-in stream-schedule prefetch, and SR200 support.
It is still on the IREE 3.10 fork (`synaptics-torq/iree@585aecc`), so the app's runtime ABI is unchanged. Its
`astra-release` ships its own `syna_npu.ko` (vermagic `6.12.62`) next to the runner, while the v2.6.0 image carries
the 2.2.0 module. Step 12 finds out whether the 2.3.0 runtime needs the 2.3.0 module.

Google's "Coral Compiler v1.0" (`google-coral/coralnpu-compiler`, announced 2026-10-20) is a separate compiler with
its own `coralnpu` HAL. It is not part of this set; the SL2619 path stays on Torq. Upstream IREE 3.13 is not
either: vmfbs from upstream IREE 3.11 and newer are rejected by the Torq-fork runtime (`required module features
[Ch]`).

Why update: NPU graphs with three or more chained dispatches return zeros or nothing on Torq 2.0 (the self-compiled
Moonshine encoder, canaries `chain8_64` / `softmax_64`). Torq 2.1–2.3 bring:
- a fix for "runtime auto mode returning zeros or SIGSEGV"
- tile-and-fuse improvements, LayerNorm on the NSS
- `torq-decode-llm`
- NPU devfreq (800 MHz → 1 GHz)

Astra v2.6.0 adds NPU DVFS, eMMC tuning and newer firmware.

Measured on the old image (2026-10-16/17):
- The 2.2.1 compiler and runtime on the **old driver** do not fix multi-dispatch: chain8_64 and softmax_64 give
  no output. The newer driver is the open question that step 12 answers.
- The app linked against the 2.2.1 runtime already ran the packs unchanged on the old image: same ASR finals, NLU
  610–640 ms/token.

The vendor Moonshine NPU encoder (Synaptics' prebuilt `encoder.vmfb`) is not needed by the app, which runs ASR on the
CPU packs. It is re-downloadable from Hugging Face at pinned revisions (`scripts/board-update-download.sh $UPD
moonshine`, Appendix B), including the board's old copy (revision `4a9b84ff11a1`, sha256 `28ef6573…`).

The image zip holds `eMMCimg/` with `preboot`, `key`, `tzk`, `sysmgr`, `bl`, `boot`, `fastlogo`, `rootfs_s.subimg.*`
and **`home_s.subimg` (partition 18 = `/home`)**. That is why the flash wipes `/home`. The USB boot image set used is
`astra-usbboot-images/sl2619` (board `rdk`, DDR4, 2 GB).

## Appendix B: rebuilding the host side

| What | How |
|---|---|
| downloads | `scripts/board-update-download.sh $UPD all` (sets: `image rollback torq torq230 moonshine usb-tool`), each verified against the release sha256 |
| 2.3.0 compiler image | `docker build -f board/docker/Dockerfile.torq-compiler -t sl2610-iree:v2.3.0 --build-arg TORQ_VERSION=2.3.0 --build-arg MANYLINUX=2_39 --build-arg BASE=python:3.12-slim-trixie board/docker` |
| 2.3.0 app runtime | `TORQ_TAG=v2.3.0 native/iree-torq-runtime/build-runtime.sh native/iree-torq-runtime/work-230`, then `OUT=native/iree-torq-runtime/dist-230 native/iree-torq-runtime/build-dist.sh native/iree-torq-runtime/work-230` |
| packs | the blueprint recipes of step 4, then `scripts/stage-asr-pack.sh` and `scripts/stage-nlu-pack.sh` |
| 2.3.0 pack compiler image | in `$HES/skainet-embedded-vendors`: `TORQ_VERSION=2.3.0 TORQ_MANYLINUX=2_39 TORQ_BASE_IMAGE=python:3.12-slim-trixie synaptics-torq/docker/build.sh` → `skainet/iree-compiler-torq:2.3.0` |
| 2.2.1 compiler image | `docker build -f board/docker/Dockerfile.torq-compiler -t sl2610-iree:v2.2.1 --build-arg TORQ_VERSION=2.2.1 --build-arg MANYLINUX=2_39 --build-arg BASE=python:3.12-slim-trixie board/docker` |
| 2.2.1 app runtime | `TORQ_TAG=v2.2.1 native/iree-torq-runtime/build-runtime.sh native/iree-torq-runtime/work-221`, then `OUT=native/iree-torq-runtime/dist-221 native/iree-torq-runtime/build-dist.sh native/iree-torq-runtime/work-221` |

Downloads, pinned:

| File | Source | sha256 |
|---|---|---|
| `sl2619_scarthgap_6.12_v2.6.0.zip` (725 MB) | [Astra SDK v2.6.0](https://github.com/synaptics-astra/sdk/releases/tag/scarthgap_6.12_v2.6.0) | `9b15b709a45b0e4b5d6c1532277e2465aee960f862961805150e220fb7ac665f` |
| `sl2619_scarthgap_6.12_v2.4.0.zip` (705 MB) | [Astra SDK v2.4.0](https://github.com/synaptics-astra/sdk/releases/tag/scarthgap_6.12_v2.4.0) | `7011b4fb6f4808834d80ff5f9096f6e596c87f68ee28c59cf6807cb082561de9` |
| usb-tool | [`synaptics-astra/usb-tool`](https://github.com/synaptics-astra/usb-tool) @ `91d4000af92f` (Git LFS, astra-update 2.0.4) | pinned commit |
| `astra-release.tar.gz` | [Torq v2.3.0](https://github.com/synaptics-torq/torq-compiler/releases/tag/v2.3.0) | `0d3ecf53dfd7906362742cd3ad6188b36fb55886380b4d4f18b0b6f46d75cfa4` |
| `torq_runtime-2.3.0-cp312-cp312-manylinux_2_39_aarch64.whl` | Torq v2.3.0 | `bdfe92470f4ac3f95bbe6ad11e6d9987c4e615762f29d32ca6a8ea8d723be6d2` |
| `torq_compiler-2.3.0-cp312-cp312-manylinux_2_39_x86_64.whl` | Torq v2.3.0 | `b809e13f1b3d16784bea5e63e12b2846d718f1dba55f0248ed46278f6bd51d7d` |
| `astra-release.tar.gz` | [Torq v2.2.1](https://github.com/synaptics-torq/torq-compiler/releases/tag/v2.2.1) | `f71693087ec336674366e254197fb622fb939e353dfd34c745e766c1edfdf433` |
| `torq_runtime-2.2.1-cp312-cp312-manylinux_2_39_aarch64.whl` | Torq v2.2.1 | `7c17c5c627670314119f3a868f0fa6a877f718ab30efbfe8ac2088b3fe93de4e` |
| `torq_compiler-2.2.1-cp312-cp312-manylinux_2_39_x86_64.whl` | Torq v2.2.1 | `37b8d56f6e148db0dda229b7b231cc6ad06ab699639d9cc3ac459cb9680da259` |
| `encoder.vmfb` (14.5 MB, the board's old copy, Torq 2.0) | [HF `Synaptics/moonshine-tiny-bf16-torq` @ `4a9b84ff11a1`](https://huggingface.co/Synaptics/moonshine-tiny-bf16-torq/tree/4a9b84ff11a1) | `28ef65731f1790f949284c0f4d546b03df3144aea28902d83470c5dee7d8500a` |
| `encoder.vmfb` (19.2 MB, 2026-09-29, likely Torq 2.2) | [HF @ `ef128b544c7b`](https://huggingface.co/Synaptics/moonshine-tiny-bf16-torq/tree/ef128b544c7b) | `71ff97581260de19774d16742d3cb237ba6c8edc0ae488f8c401826b1e4d4492` |
| `encoder.vmfb` (19.1 MB, 2026-10-19, likely Torq 2.3.0) | [HF @ `d3be4736dc4e`](https://huggingface.co/Synaptics/moonshine-tiny-bf16-torq/tree/d3be4736dc4e) | `e9f3910adeebaea81280308cc34a19d39fad0a29a8b5efdb9fdce2424b2d42fa` |

## References

- Astra documentation, *Updating the eMMC*: <https://synaptics-astra.github.io/doc/v/latest/linux/index.html>
- SL2600 hardware manual (buttons, boot straps): <https://synaptics-astra.github.io/doc/v/latest/hw/sl2600.html>
- USB tool: <https://github.com/synaptics-astra/usb-tool>
- Torq releases: <https://github.com/synaptics-torq/torq-compiler/releases>
- Torq examples: <https://github.com/synaptics-torq/torq-examples>
- Cartridge blueprints: <https://github.com/SKaiNET-developers/SKaiNET-cartridge-blueprints>
