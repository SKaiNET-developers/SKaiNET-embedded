#!/usr/bin/env bash
# Steps 10-14 of docs/BOARD-UPDATE.md on a freshly flashed board, in one run: audit, Torq 2.3.0 release tools,
# canaries, install (app, check binaries, packs, test recordings), the three checks. Each step's output goes to
# build/bringup/<time>/<step>.log, for the log entry of step 16.
#   BOARD=root@<new-ip> scripts/board-bringup.sh [step…]      (default: 10 11 12 13 14)
#   UPD=<board-update dir> WAV=<dir holding hes-wav>          (defaults: the HES workspace layout)
# Expects Part 1 done on the host: $UPD/torq-2.3.0/astra-release, sl2610-iree:v2.3.0, the dist-230 binaries, build/packs.
# The audit (DRIFT against the old lock) and canary failures are recorded, not fatal: the app runs on the CPU.
# Never swaps the NPU driver; that is the manual fallback in step 12.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
export BOARD="${BOARD:?set BOARD=root@<board-ip>}"
UPD="${UPD:-$HERE/../../board-update}"
WAV="${WAV:-$HERE/../../board-backup/20261017-preflash/root}"
TORQ=/home/root/torq-2.3.0
RUNNER=$TORQ/tools/astra-sl-torq-run-module
APPDIR=/home/root/hes-tui
STEPS="${*:-10 11 12 13 14}"
LOG="$HERE/build/bringup/$(date +%Y%m%d-%H%M)"; mkdir -p "$LOG"
cd "$HERE"

declare -A RESULT
step() { # <n> <name> <command…>: runs the command, tees it to the step log, records ok/FAIL
  local n=$1 name=$2; shift 2
  echo; echo "=== step $n: $name ==="
  "$@" 2>&1 | tee "$LOG/$n-$name.log"
  local rc=${PIPESTATUS[0]}
  RESULT[$n]="$([ "$rc" = 0 ] && echo ok || echo "FAIL (exit $rc)")  $name"
  return "$rc"
}

s10() { scripts/board-audit.sh; }

s11() {
  [ -x "$UPD/torq-2.3.0/astra-release/tools/astra-sl-torq-run-module" ] || { echo "missing $UPD/torq-2.3.0/astra-release (step 1)"; return 1; }
  ssh "$BOARD" "mkdir -p $TORQ" &&
  tar -C "$UPD/torq-2.3.0/astra-release" -cf - . | ssh "$BOARD" "tar -C $TORQ -xf -" || return 1
  local usage; usage="$(ssh "$BOARD" "$RUNNER --help 2>&1 | head -n 3")"
  echo "$usage"
  ! grep -q GLIBC <<<"$usage"                                     # GLIBC_2.38 not found: image older than v2.6.0
}

s12() { scripts/board-canaries.sh sl2610-iree:v2.3.0 "$RUNNER" ""; }

s13() {
  scripts/deploy-tui.sh --no-build || return 1
  for b in hes-asr-check hes-nlu-check hes-app-check; do
    f=$(ls tui/build/bin/linuxArm64/*/$b.kexe 2>/dev/null | head -n 1)
    [ -n "$f" ] || { echo "no $b.kexe (step 5)"; return 1; }
    ssh "$BOARD" "cat > $APPDIR/$b.kexe && chmod +x $APPDIR/$b.kexe" < "$f" || return 1
  done
  scripts/install-dev-packs.sh || return 1
  tar -C "$WAV" -cf - hes-wav | ssh "$BOARD" 'tar -C /home/root -xf -' || return 1
  ssh "$BOARD" "ls $APPDIR /home/root/.homeevensmarter/cartridges; ls /home/root/hes-wav/golden | head -n 3"
}

s14() {
  local rc=0
  ssh "$BOARD" "cd $APPDIR && LD_LIBRARY_PATH=lib ./hes-asr-check.kexe ~/.homeevensmarter/cartridges/asr-* \
    /home/root/hes-wav/golden/g*.wav /home/root/hes-wav/ab_u*.wav /home/root/hes-wav/beckett5s.wav" || rc=1
  ssh "$BOARD" "cd $APPDIR && LD_LIBRARY_PATH=lib ./hes-nlu-check.kexe ~/.homeevensmarter/cartridges/nlu-*" || rc=1
  ssh "$BOARD" "cd $APPDIR && LD_LIBRARY_PATH=lib ./hes-app-check.kexe --verify \
    /home/root/hes-wav/golden/g01.wav /home/root/hes-wav/golden/g07.wav /home/root/hes-wav/golden/g10.wav" || rc=1
  return $rc
}

echo "bring-up of $BOARD, steps: $STEPS, logs: $LOG"
for n in $STEPS; do
  case "$n" in
    10) step 10 audit s10 ;;                                      # DRIFT against the old lock is expected
    11) step 11 torq-tools s11 || break ;;
    12) step 12 canaries s12 ;;                                   # identity_1x4 + chain2_64 must pass; see the doc
    13) step 13 install s13 || break ;;
    14) step 14 checks s14 ;;
    *) echo "unknown step $n (10-14)"; exit 2 ;;
  esac
done

echo; echo "=== summary ($LOG) ==="
rc=0
for n in $STEPS; do
  [ -n "${RESULT[$n]:-}" ] || { echo "step $n: not run"; rc=1; continue; }
  echo "step $n: ${RESULT[$n]}"
  case "$n:${RESULT[$n]}" in 10:*|12:*) ;; *FAIL*) rc=1;; esac
done
echo "Compare step 12 with the canary table and step 14 with the goals in docs/BOARD-UPDATE.md, then do steps 15-16."
exit $rc
