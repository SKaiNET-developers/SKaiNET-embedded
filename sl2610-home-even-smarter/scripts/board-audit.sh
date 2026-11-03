#!/usr/bin/env bash
# READ-ONLY board audit: what the board runs (SDK, kernel, NPU module, Torq runtimes, GPU, audio, free space),
# compared with board/board.lock. Prints DRIFT lines and exits 1 on any mismatch. Changes nothing on the board.
#   BOARD=root@<ip> scripts/board-audit.sh
# The board is BusyBox: no `timeout`, no `strings`, `head -n N` only.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BOARD="${BOARD:?set BOARD=root@<board-ip>}"
LOCK="$HERE/board/board.lock"

REPORT="$(ssh -o ConnectTimeout=8 "$BOARD" '
  echo "SOC=$(tr -d "\0" < /proc/device-tree/model 2>/dev/null)"
  echo "SDK_REVISION=$(cat /etc/build* 2>/dev/null | grep -a SYNA_SDK_REVISION | cut -d= -f2 | tr -d " ")"
  echo "KERNEL=$(uname -r)"
  echo "NPU_MODULE=$(modinfo -n syna_npu 2>/dev/null || echo MISSING)"
  echo "NPU_VERMAGIC=$(modinfo syna_npu 2>/dev/null | grep vermagic | cut -c17-)"
  echo "DEV_TORQ=$( [ -e /dev/torq ] && echo present || echo MISSING )"
  echo "NPU_DEVFREQ=$(cat /sys/class/devfreq/*synpu*/cur_freq /sys/class/devfreq/*npu*/cur_freq 2>/dev/null | head -n 1)"
  echo "GPU=$(cat /sys/class/misc/mali0/device/gpuinfo 2>/dev/null)"
  echo "VULKAN_ICD=$(ls /etc/vulkan/icd.d /usr/share/vulkan/icd.d 2>/dev/null | tr "\n" " ")"
  echo "ALSA_CAPTURE=$(arecord -l 2>/dev/null | grep -c "^card")"
  echo "USB_AUDIO=$(grep -l -i audio /sys/bus/usb/devices/*/product 2>/dev/null | wc -l)"
  echo "ROOT_FREE=$(df -h / | tail -n 1 | awk "{print \$4}")"
  echo "HOME_FREE=$(df -h /home | tail -n 1 | awk "{print \$4}")"
  for b in /usr/bin/iree-run-module /usr/bin/torq-run-module /usr/bin/astra-sl-torq-run-module $(find /home/root -maxdepth 6 -name torq-run-module 2>/dev/null); do
    [ -x "$b" ] || continue
    v=$(grep -oaE "torq-runtime/[0-9][.A-Za-z0-9_+-]*" "$b" 2>/dev/null | head -n 1)
    echo "RUNTIME=$b ${v:-(no version string)}"
  done
' 2>&1)" || { echo "cannot reach $BOARD"; exit 2; }

echo "=== board audit ($BOARD) ==="
echo "$REPORT" | sed 's/^/  /'

lock() { grep -E "^$1=" "$LOCK" | head -n 1 | cut -d= -f2- | sed 's/[[:space:]]*#.*//;s/[[:space:]]*$//'; }
got() { echo "$REPORT" | grep -E "^$1=" | head -n 1 | cut -d= -f2-; }
rc=0
drift() { echo "DRIFT: $*"; rc=1; }
echo "=== drift vs board/board.lock ==="
[ "$(got KERNEL)" = "$(lock BOARD_KERNEL)" ] || drift "kernel $(got KERNEL) != lock $(lock BOARD_KERNEL)"
[ "$(got NPU_MODULE)" = "$(lock BOARD_NPU_MODULE)" ] || drift "NPU module $(got NPU_MODULE) != lock $(lock BOARD_NPU_MODULE)"
case "$(got NPU_VERMAGIC)" in "$(got KERNEL) "*) ;; *) drift "syna_npu vermagic '$(got NPU_VERMAGIC)' does not match kernel $(got KERNEL)";; esac
[ "$(got DEV_TORQ)" = present ] || drift "/dev/torq missing"
case "$(got SOC)" in *"$(lock BOARD_SOC)"*) ;; *) drift "SoC '$(got SOC)' != lock $(lock BOARD_SOC)";; esac
[ "$rc" = 0 ] && echo "OK: board matches the lock"
exit $rc
