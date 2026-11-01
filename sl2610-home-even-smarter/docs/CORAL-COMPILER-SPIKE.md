# Spike: Google's Coral compiler v1.0 and the SL2619

Status: research done; compiler build and board probe pending. Nothing here changes the app's pins: the board path
stays on Torq ([BOARD-UPDATE.md](BOARD-UPDATE.md)).

Short answer: Google's Coral compiler v1.0 does not run anything on the SL2619, on our Astra Machina or on Google's
own Coralboard. Its runtime drives simulators and Google's Nexus FPGA boards. On the SL2619, NPU work goes through
Synaptics' Torq toolchain, and the Coral NPU in the chip is most likely the RISC-V core inside Torq.

## Questions

1. Does the Coral compiler accept the StableHLO that SKaiNET emits, and how much of it does it place on Coral NPU?
2. Can its output run on the SL2619, next to or inside the Torq NPU stack?

## What the Coral compiler is

[`google-coral/coralnpu-compiler`](https://github.com/google-coral/coralnpu-compiler) (Apache-2.0, announced
2026-10-20, spike pinned at `d715556c3853`, 2026-10-30) is an **IREE compiler plugin** with a custom `coralnpu` HAL
device, on **IREE 3.12** (ten local patches) and its own LLVM. It targets **Coral NPU**, a RISC-V 32 core: RV32IMF +
Zbb, `zve32f`/`zvl128b` vectors, BF16, and the Zvt v0.4 matrix extension (a 16×16 unit of 64 FP32 MAC PEs). Input is
StableHLO (JAX examples) or TOSA (TFLite examples); there is a `moonshine-jax-aot` example (Moonshine-tiny from the
Keras weights).

- No binary release: built from source (Bazel 9.1.0, or CMake for the compiler alone), clang-19, Python 3.13.
- Runtime execution backends: the MPACT and Spike instruction-set simulators, Verilator RTL, and an "FPGA" backend
  for Google's **Nexus FPGA boards**, driven from a host PC over FTDI USB/SPI (`tests/fpga`: check tests and
  MobileNetV2). **There is no Linux backend for a Coral NPU inside an SoC.** Google's guide likewise names only "the
  MPACT-CoralNPU simulator, Verilator, Spike, or FPGA targets".
- Its placement cost model is explicitly an SL2619 one: "a unified-memory SoC with an embedded Arm Cortex-A55 host,
  e.g. the Synaptics SL2619 Coralboard", Coral NPU at 800 MHz. Dispatches go to the NPU or stay on the A55 host by a
  roofline estimate (`--coralnpu-roofline-speedup-threshold`).

## How it relates to Torq

| | Torq (what the board runs) | Coral compiler |
|---|---|---|
| IREE base | Synaptics fork, 3.10.0.dev (`585aecc`) | upstream 3.12 + patches |
| HAL device | `torq` (`/dev/torq`, `syna_npu.ko`) | `coralnpu` (simulators, FPGA over USB) |
| Hardware model | NSS slices + a **CSS** control core + A55 host | Coral NPU (RV32 + Zve32 + Zvt) + A55 host |

Host-side evidence that the two meet: the Torq 2.3.0 compiler builds its **CSS programs for `rv32imf_zve32x`**,
next to its `torq_hw` CSS task ops, the same RISC-V base ISA as Coral NPU. The CSS is where our Moonshine encoder
fails without `--torq-disable-css` (CSS stack size). Not shown yet: whether the CSS *is* a Coral NPU core (with Zvt),
and whether anything but the Torq runtime can drive it.

Synaptics describes the SL2610 line as the Torq T1 NPU "together with the industry's first production
implementation of the Coral NPU from Google".

## Google's Coralboard

The Coralboard SL (Synaptics/Google, shown at Google I/O 2026; DigiKey 720-KDVSL2619-11, about $100, microSD only)
is built on **the same Astra SL2619 chip** as the Astra Machina ("based on the exact same chip"). It has fewer ports:
one USB, no Ethernet, no on-board microphones. Its NPU examples, Google's I/O demo (YOLOv8) and Gemma 3 270M on the
NPU all use the **Torq compiler and runtime**, not the Coral compiler.

Its images (Astra v2.6.1, `sl2619_coralboard_*`) carry the same `kernel-module-syna-npu` as the rdk images, and
their rootfs SBOM lists no Coral-specific package or firmware. So on today's images Coral NPU is reachable only
through Torq, on either board. A Coralboard adds nothing for this project.

Upstream IREE vmfbs (3.11 and newer) are rejected by the Torq-fork runtime (`required module features [Ch]`), so
Coral-compiler output (3.12) cannot be loaded by the board's runtime either, whatever the device.

## Host part

Everything lives in `$HES/spikes/coral-compiler` (outside the repos):

| File | |
|---|---|
| `src/` | the pinned source tree with submodules (shallow, LLVM checked out once) |
| `Dockerfile`, `build.sh` | Debian trixie + clang-19 build image; CMake build of `iree-compile` only (Release, no Bazel, no simulators) |
| `compile-tests.sh` | compiles the four canaries and the SKaiNET Moonshine encoder in two modes, `npu` (everything on Coral NPU) and `split` (A55 host + Coral NPU), with the affinity report per input |
| `probe-board.sh` | read-only board probe: device tree, NPU module, remoteproc, firmware, dmesg |

Results: _pending the build_ (Release build of LLVM/MLIR/IREE, about 6–10 GB; waiting on host disk space). The
answers would be host-side only: how the Coral compiler splits SKaiNET graphs between the A55 and Coral NPU, checked
in the MPACT simulator. Nothing it produces runs on the board.

## Board part

`BOARD=root@<ip> $HES/spikes/coral-compiler/probe-board.sh` on the old image and again after the flash. It looks for
a Coral NPU / RISC-V node in the device tree, a remoteproc or firmware for it, and what `syna_npu` exposes.

Results: _pending._

## Conclusion so far

The board path stays on Torq. The Coral compiler is interesting for two things: as a second opinion on how SKaiNET
graphs map to a Coral-class NPU (the affinity report), and if Synaptics or Google later ship a Coral NPU runtime for
the SL2619. It is not a replacement for the Torq toolchain on this board.

## Sources

- Google, *MLIR compiler for Coral NPU*: <https://developers.google.com/coral/guides/software/coral-mlir-compiler>
- Google Developers Blog, Coral NPU from model to silicon: <https://developers.googleblog.com/unifying-edge-ai-how-coral-npus-open-standard-brings-portable-energy-viable-intelligence-from-model-to-silicon/>
- Synaptics, *Coralboard vs Machina*: <https://developer.synaptics.com/blog/coralboard-vs-machina>
- Synaptics, SL2610 product line: <https://www.synaptics.com/products/embedded-processors/sl2610-product-line>
- Google, Coralboard example apps: <https://developers.google.com/coral/products/SL2610-demos-examples>
- CNX Software, Coralboard with Astra SL2619: <https://www.cnx-software.com/2026/05/20/coralboard-features-synaptics-astra-sl2619-edge-ai-soc-supports-google-gemma-3-inference/>
- Astra SDK v2.6.1 release (coralboard images, SBOM): <https://github.com/synaptics-astra/sdk/releases/tag/scarthgap_6.12_v2.6.1>
