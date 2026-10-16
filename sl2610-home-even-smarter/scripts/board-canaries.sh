#!/usr/bin/env bash
# Canary gate for a (Torq compiler, board runtime) pair: compiles tiny known-answer graphs, runs them on the board's
# NPU and checks the outputs. Classes: PASS | VERSION (exec format rejected) | ZEROS (ran, all zeros) | COMPILE | ERROR.
#   BOARD=root@<ip> scripts/board-canaries.sh [compiler-image] [runtime-binary] [runtime-lib-dirs]
# Defaults: the compiler image from board/board.lock (TORQ_COMPILER) and the board's torq2-stable runtime.
# chain8_64 is the one that matters: 3+ chained dispatches returned zeros on Torq 2.0.0.
# Writes only under /home/root/hes-canaries on the board; streams files with ssh (BusyBox has no sftp).
# Host needs docker and python3 with numpy + ml_dtypes.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BOARD="${BOARD:?set BOARD=root@<board-ip>}"
lock() { grep -E "^$1=" "$HERE/board/board.lock" | head -n 1 | cut -d= -f2- | sed 's/[[:space:]]*#.*//;s/[[:space:]]*$//'; }
IMAGE="${1:-$(lock TORQ_COMPILER)}"
RT="${2:-/home/root/torq2-stable/torq/_runtime_libs/torq-run-module}"
RTLIBS="${3:-/home/root/torq2-stable/torq/_runtime_libs:/home/root/torq2-stable/iree/_runtime_libs}"
CAN="$HERE/board/canaries"
OUT="$HERE/build/canaries/$(echo "$IMAGE" | tr ':/' '__')"; mkdir -p "$OUT"
WORK=/home/root/hes-canaries
CANARIES="${CANARIES:-identity_1x4 chain2_64 chain8_64 softmax_64}"

echo "=== canaries: compiler $IMAGE × runtime $RT ==="
ssh "$BOARD" "mkdir -p $WORK"
all=0
for c in $CANARIES; do
  vmfb="$OUT/$c.vmfb"
  if ! docker run --rm --user "$(id -u):$(id -g)" -v "$CAN:/in:ro" -v "$OUT:/out" "$IMAGE" "/in/$c.mlir" "/out/$c.vmfb" >"$OUT/$c.compile.log" 2>&1 || [ ! -s "$vmfb" ]; then
    echo "  $c: COMPILE (see $OUT/$c.compile.log)"; all=1; continue
  fi
  specs=$(python3 "$CAN/_io.py" gen "$c" "$OUT")
  ssh "$BOARD" "cat > $WORK/$c.vmfb" < "$vmfb"
  inargs=""; i=0
  for s in $specs; do
    ssh "$BOARD" "cat > $WORK/$c.in$i.bin" < "$OUT/$c.in$i.bin"
    inargs="$inargs --input=$s=@$WORK/$c.in$i.bin"; i=$((i+1))
  done
  start=$(date +%s%N)
  ssh "$BOARD" "cd $WORK && rm -f $c.out.bin && LD_LIBRARY_PATH=$RTLIBS $RT --module=$c.vmfb --function=main --device=torq --torq_hw_type=astra_machina $inargs --output=@$WORK/$c.out.bin" >"$OUT/$c.run.log" 2>&1
  ms=$(( ($(date +%s%N) - start) / 1000000 ))
  ssh "$BOARD" "cat $WORK/$c.out.bin 2>/dev/null" > "$OUT/$c.out.bin"
  chk="FAIL: no output"; [ -s "$OUT/$c.out.bin" ] && chk=$(python3 "$CAN/_io.py" check "$c" "$OUT/$c.out.bin" 2>&1)
  if grep -qiE "does not match expected version" "$OUT/$c.run.log"; then v=VERSION
  elif echo "$chk" | grep -q '^PASS'; then v=PASS
  elif echo "$chk" | grep -qi 'ALL ZEROS'; then v=ZEROS
  else v=ERROR; fi
  [ "$v" = PASS ] || all=1
  printf '  %-13s %-8s %s  (%d ms incl. ssh)\n' "$c" "$v" "$chk" "$ms"
done
ssh "$BOARD" "rm -rf $WORK"
[ $all = 0 ] && echo "GATE: PASS" || echo "GATE: FAIL — multi-dispatch NPU graphs are not trustworthy on this pair"
exit $all
