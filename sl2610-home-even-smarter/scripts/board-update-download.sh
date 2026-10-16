#!/usr/bin/env bash
# Downloads everything the board update needs (docs/BOARD-UPDATE.md) into DIR and verifies every file against the
# sha256 published with its GitHub release. Re-running skips files that are already present and valid.
#   scripts/board-update-download.sh DIR [all|image|rollback|torq|usb-tool] ...
#     image     Astra SDK scarthgap_6.12_v2.6.0 eMMC image for SL2619            (~725 MB)
#     rollback  Astra SDK scarthgap_6.12_v2.4.0 eMMC image (the current board)   (~705 MB)
#     torq      Torq 2.2.1: board runtime + NPU module, runtime wheel, compiler wheel, manuals (~110 MB)
#     usb-tool  synaptics-astra/usb-tool (astra-update v2.0.4 + USB boot images), Git LFS, pinned commit
# Needs curl, sha256sum, unzip, git and git-lfs (usb-tool only).
set -euo pipefail
DIR="${1:?usage: board-update-download.sh DIR [all|image|rollback|torq|usb-tool] ...}"; shift
SETS="${*:-all}"
mkdir -p "$DIR"
SDK=https://github.com/synaptics-astra/sdk/releases/download
TORQ=https://github.com/synaptics-torq/torq-compiler/releases/download/v2.2.1
USB_TOOL_COMMIT=91d4000af92f   # usb-tool main, 2026-10-10: astra-update v2.0.4 (Astra v2.6.0 needs >= 2.0.3)

want() { case " $SETS " in *" all "*|*" $1 "*) return 0;; *) return 1;; esac; }

fetch() { # <file> <url> <sha256>
  local f="$DIR/$1"
  if [ -f "$f" ] && echo "$3  $f" | sha256sum -c --quiet 2>/dev/null; then echo "ok    $1"; return; fi
  echo "fetch $1"
  curl -fsSL --retry 3 -C - -o "$f" "$2" || curl -fsSL --retry 3 -o "$f" "$2"
  echo "$3  $f" | sha256sum -c --quiet || { echo "CHECKSUM MISMATCH: $1"; exit 1; }
  echo "ok    $1"
}

if want image; then
  fetch sl2619_scarthgap_6.12_v2.6.0.zip "$SDK/scarthgap_6.12_v2.6.0/sl2619_scarthgap_6.12_v2.6.0.zip" \
    9b15b709a45b0e4b5d6c1532277e2465aee960f862961805150e220fb7ac665f
fi
if want rollback; then
  fetch sl2619_scarthgap_6.12_v2.4.0.zip "$SDK/scarthgap_6.12_v2.4.0/sl2619_scarthgap_6.12_v2.4.0.zip" \
    7011b4fb6f4808834d80ff5f9096f6e596c87f68ee28c59cf6807cb082561de9
fi
if want torq; then
  mkdir -p "$DIR/torq-2.2.1"
  fetch torq-2.2.1/astra-release.tar.gz "$TORQ/astra-release.tar.gz" \
    f71693087ec336674366e254197fb622fb939e353dfd34c745e766c1edfdf433
  fetch torq-2.2.1/torq_runtime-2.2.1-cp312-cp312-manylinux_2_39_aarch64.whl "$TORQ/torq_runtime-2.2.1-cp312-cp312-manylinux_2_39_aarch64.whl" \
    7c17c5c627670314119f3a868f0fa6a877f718ab30efbfe8ac2088b3fe93de4e
  fetch torq-2.2.1/torq_compiler-2.2.1-cp312-cp312-manylinux_2_39_x86_64.whl "$TORQ/torq_compiler-2.2.1-cp312-cp312-manylinux_2_39_x86_64.whl" \
    37b8d56f6e148db0dda229b7b231cc6ad06ab699639d9cc3ac459cb9680da259
  fetch torq-2.2.1/user_manual.zip "$TORQ/user_manual.zip" \
    b08b6034559ba413915f1626828b1fb5d41a8355866e119368019d16365c453c
  mkdir -p "$DIR/torq-2.2.1/astra-release" && tar xzf "$DIR/torq-2.2.1/astra-release.tar.gz" -C "$DIR/torq-2.2.1/astra-release"
  echo "      astra-release: $(cd "$DIR/torq-2.2.1/astra-release" && find . -type f | sort | tr '\n' ' ')"
fi
if want usb-tool; then
  command -v git-lfs >/dev/null || { echo "git-lfs is required for usb-tool (sudo apt install git-lfs)"; exit 1; }
  if [ ! -d "$DIR/usb-tool/.git" ]; then
    git clone -q https://github.com/synaptics-astra/usb-tool "$DIR/usb-tool"
  fi
  git -C "$DIR/usb-tool" fetch -q origin && git -C "$DIR/usb-tool" checkout -q "$USB_TOOL_COMMIT"
  git -C "$DIR/usb-tool" lfs install --local >/dev/null && git -C "$DIR/usb-tool" lfs pull
  bin="$DIR/usb-tool/bin/linux/x86_64/astra-update"
  [ -s "$bin" ] && head -c 4 "$bin" | grep -q ELF || { echo "astra-update binary missing or still an LFS pointer: $bin"; exit 1; }
  chmod +x "$bin"
  echo "ok    usb-tool @ $USB_TOOL_COMMIT ($(du -h "$bin" | cut -f1) astra-update, boot images: $(ls "$DIR/usb-tool/astra-usbboot-images" | grep -c sl2619) sl2619 sets)"
fi
echo "done: $DIR"
