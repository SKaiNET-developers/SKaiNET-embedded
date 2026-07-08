#!/usr/bin/env bash
# torq-board-audit.sh — READ-ONLY audit of the SL2610 board's Torq version landscape.
#
# Enumerates every Torq/IREE runtime on the board, its `torq-runtime/<ver>` string, the SDK
# marker, kernel and /dev/torq, and the runtime path the app (MoonshineDecoder.kt) actually calls.
# Then diffs the findings against scripts/torq-toolchain.lock and prints DRIFT lines + exits
# non-zero on any mismatch. Makes NO changes on the board (pure inspection).
#
#   BOARD=root@192.168.3.26 scripts/torq-board-audit.sh
#
# The board is BusyBox — no `strings`, no `head -N` (use `grep -a` / `head -n`).
set -uo pipefail

BOARD="${BOARD:-root@192.168.3.26}"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOCK="$HERE/torq-toolchain.lock"

# The runtime binary the shipped app invokes (keep in sync with MoonshineDecoder.kt `torqBin`).
APP_RUNTIME="/home/root/sl2610-voice-cc/.venv/lib/python3.12/site-packages/torq/_runtime_libs/torq-run-module"

echo "=== SL2610 Torq board audit ($BOARD) ==="

# One remote call; BusyBox-safe. Extract runtime version by grepping the binary as text.
REPORT="$($SSH "$BOARD" '
  echo "SDK=$(cat /etc/synaptics* /etc/astra* /etc/*release* 2>/dev/null | grep -oaE "scarthgap_[0-9.]+_v[0-9.]+" | head -n 1)"
  echo "KERNEL=$(uname -r)"
  echo "DEVTORQ=$(ls -la /dev/torq 2>/dev/null || echo MISSING)"
  echo "APP_RUNTIME_PRESENT=$( [ -e '"$APP_RUNTIME"' ] && echo yes || echo NO )"
  # Candidate runtime binaries (native + every deployed copy) + anything else found.
  CANDS="/usr/bin/iree-run-module '"$APP_RUNTIME"' /home/root/torq2/torq_libs/torq-run-module /home/root/torq2/iree_libs/iree-run-module"
  FOUND=$(find /home/root /usr/bin -name "torq-run-module" -o -name "iree-run-module" 2>/dev/null)
  for b in $CANDS $FOUND; do echo "$b"; done | sort -u | while read -r B; do
    [ -x "$B" ] || continue
    D=$(dirname "$B")
    V=""
    # 1) read the torq_runtime Version from its dist-info METADATA — search the binary dir and up
    #    to 4 parents (covers both wheel/site-packages and release-package layouts).
    P="$D"
    for _ in 1 2 3 4 5; do
      MD=$(ls "$P"/torq_runtime-*.dist-info/METADATA 2>/dev/null | head -n 1)
      [ -n "$MD" ] && { V="torq_runtime-$(grep "^Version:" "$MD" | head -n 1 | awk "{print \$2}")"; break; }
      P=$(dirname "$P"); [ "$P" = "/" ] && break
    done
    # 2) else grep the binary / runtime .so for a compiled-in `torq-runtime/<ver>` build path.
    if [ -z "$V" ]; then
      for SRC in "$B" "$D"/*.so; do
        [ -f "$SRC" ] || continue
        V=$(grep -oaE "torq-runtime/[0-9][.A-Za-z0-9_+-]*" "$SRC" 2>/dev/null | head -n 1)
        [ -n "$V" ] && break
      done
    fi
    [ -z "$V" ] && V="(no version string)"
    echo "RUNTIME=$B|$V"
  done
' 2>&1)"

echo "$REPORT" | grep -vE '^RUNTIME=' | sed 's/^/  /'
echo "  --- runtimes on board ---"
echo "$REPORT" | grep -E '^RUNTIME=' | sed 's/^RUNTIME=/  /'
echo "  app calls: $APP_RUNTIME"

# ---- Drift check vs the lockfile ----
rc=0
if [ ! -f "$LOCK" ]; then
  echo ""; echo "NOTE: no lockfile at $LOCK yet — run scripts/torq-verify.sh to establish the canonical triple."
  exit 0
fi
lock_get(){ grep -oaE "^$1=.*" "$LOCK" | head -n 1 | cut -d= -f2- | sed 's/[[:space:]]*#.*//;s/[[:space:]]*$//'; }
L_SDK="$(lock_get BOARD_SDK)"; L_RTPATH="$(lock_get BOARD_RUNTIME_PATH)"; L_RTVER="$(lock_get BOARD_RUNTIME_VERSION)"
B_SDK="$(echo "$REPORT" | grep -oaE '^SDK=.*' | cut -d= -f2-)"
B_RTLINE="$(echo "$REPORT" | grep -aF "RUNTIME=$L_RTPATH|")"
B_RTVER="${B_RTLINE#*|}"

echo ""; echo "=== drift vs lockfile ==="
[ -n "$L_SDK" ] && [ "$B_SDK" != "$L_SDK" ] && { echo "DRIFT: board SDK '$B_SDK' != lock '$L_SDK'"; rc=1; }
[ -n "$L_RTPATH" ] && ! echo "$REPORT" | grep -qaF "RUNTIME=$L_RTPATH|" && { echo "DRIFT: lock BOARD_RUNTIME_PATH '$L_RTPATH' not present on board"; rc=1; }
[ -n "$L_RTVER" ] && [ -n "$B_RTVER" ] && [ "$B_RTVER" != "$L_RTVER" ] && { echo "DRIFT: runtime '$L_RTPATH' is '$B_RTVER' != lock '$L_RTVER'"; rc=1; }
# The app must call the canonical runtime — else it silently runs vmfbs on a non-pinned runtime.
[ -n "$L_RTPATH" ] && [ "$APP_RUNTIME" != "$L_RTPATH" ] && {
  echo "DRIFT: app (MoonshineDecoder.kt) calls '$APP_RUNTIME' but lock canonical is '$L_RTPATH' — repoint torqBin."; rc=1; }
[ $rc -eq 0 ] && echo "OK — board matches lockfile."
exit $rc
