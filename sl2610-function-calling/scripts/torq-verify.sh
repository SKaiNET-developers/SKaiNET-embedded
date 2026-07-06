#!/usr/bin/env bash
# torq-verify.sh — the canary GATE. Compile tiny known-answer models with a chosen Torq compiler,
# run them on a chosen board runtime, assert the outputs, and print a PASS/FAIL matrix.
#
# This is the single automated check that a (compiler, board-runtime) pair actually WORKS on
# hardware — catching all three failure classes we hit this session:
#   VERSION  : runtime rejects the executable format ("expected version N")
#   ZEROS    : loads + runs but returns all zeros (multi-dispatch hardware limit)
#   COMPILE  : the compiler crashes/errors producing the vmfb
#
#   scripts/torq-verify.sh                          # pinned compiler × board runtime (from lockfile)
#   scripts/torq-verify.sh --matrix                 # all known compilers × all known board runtimes
#   scripts/torq-verify.sh --compiler g165 --runtime native-beta
#
# On an all-PASS run against the pinned pair, stamps CANARY_STATUS=PASS into the lockfile.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CAN="$HERE/torq-canaries"
LOCK="$HERE/torq-toolchain.lock"
BOARD="${BOARD:-root@192.168.3.26}"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10"
SCP="scp -o StrictHostKeyChecking=no"
WORK="/home/root/torq-verify"
PY="${PY:-python3}"                          # host python with numpy + ml_dtypes
CANARIES="identity_1x4 chain2_64 chain8_64 softmax_64"
OUT="$HERE/../build/mlir/canary"; mkdir -p "$OUT"

# ---- compiler backends: id -> a function that compiles $1(.mlir) to $2(.vmfb); echoes EXECFMT hint ----
G165_VENV="${G165_VENV:-}"                    # path to a venv with the g165e12a torq_compiler wheel (emits exec v1)
compile_stable_docker(){ IREE_IMAGE="${IREE_IMAGE:-sl2610-iree:v2.0.0}" bash "$HERE/iree-compile-torq-docker.sh" "$1" "$2" >/dev/null 2>&1; }
compile_g165(){
  [ -n "$G165_VENV" ] || { echo "  (g165: set G165_VENV to a venv with the g165e12a wheel)"; return 3; }
  local ic="$G165_VENV/lib/python3.12/site-packages/iree/compiler/_mlir_libs/iree-compile"
  local lld="$G165_VENV/lib/python3.12/site-packages/iree/compiler/_mlir_libs/iree-lld"
  local shim; shim="$(mktemp -d)"; printf '#!/usr/bin/env bash\nexec "%s" -flavor gnu "$@"\n' "$lld" >"$shim/ld"; chmod +x "$shim/ld"
  PATH="$shim:$PATH" LD_LIBRARY_PATH="$(dirname "$ic")" "$ic" --iree-input-type=stablehlo \
    --iree-hal-target-device=torq --torq-hw=SL2610 --torq-fallback-f32-to-host "$1" -o "$2" >/dev/null 2>&1
  local rc=$?; rm -rf "$shim"; return $rc
}

# ---- board runtimes: id -> "<runtime-binary>|<colon-separated LD_LIBRARY_PATH dirs>" ----
declare -A RUNTIMES=(
  [native-beta]="/usr/bin/iree-run-module|"
  [app-alpha]="/home/root/sl2610-voice-cc/.venv/lib/python3.12/site-packages/torq/_runtime_libs/torq-run-module|/home/root/sl2610-voice-cc/.venv/lib/python3.12/site-packages/torq/_runtime_libs:/home/root/sl2610-voice-cc/.venv/lib/python3.12/site-packages/iree/_runtime_libs"
  [torq2-stable]="/home/root/torq2-stable/torq/_runtime_libs/torq-run-module|/home/root/torq2-stable/torq/_runtime_libs:/home/root/torq2-stable/iree/_runtime_libs"
  [torq2]="/home/root/torq2/torq_libs/torq-run-module|/home/root/torq2/torq_libs:/home/root/torq2/iree_libs"
)
declare -A COMPILERS=( [stable-docker]=compile_stable_docker [g165]=compile_g165 )

classify(){ # given the board run log $1 and the check result $2, return PASS/VERSION/ZEROS/ERROR
  local log="$1" chk="$2"
  grep -qiE "does not match expected version|version .* does not match" "$log" && { echo VERSION; return; }
  echo "$chk" | grep -q '^PASS' && { echo PASS; return; }
  echo "$chk" | grep -qi 'ALL ZEROS' && { echo ZEROS; return; }
  echo ERROR
}

run_pair(){ # $1 compiler-id  $2 runtime-id  -> prints one matrix row; sets PAIR_OK=0/1
  local cid="$1" rid="$2"
  local fn="${COMPILERS[$cid]}" rt="${RUNTIMES[$rid]}"
  local rbin="${rt%%|*}" rlibs="${rt#*|}"; PAIR_OK=1
  local row="  $cid × $rid :"
  for c in $CANARIES; do
    local vmfb="$OUT/${c}.${cid}.vmfb"
    # subshell + monitor-off so a compiler SIGSEGV doesn't spew "Segmentation fault (core dumped)".
    if ! ( set +m; $fn "$CAN/$c.mlir" "$vmfb" ) 2>/dev/null || [ ! -s "$vmfb" ]; then row="$row  $c=COMPILE"; PAIR_OK=0; continue; fi
    local specs; specs=$($PY "$CAN/_io.py" gen "$c" "$OUT")
    $SSH "$BOARD" "mkdir -p $WORK" 2>/dev/null
    $SCP "$vmfb" "$BOARD:$WORK/$c.vmfb" >/dev/null 2>&1
    local inargs="" i=0
    for s in $specs; do $SCP "$OUT/$c.in$i.bin" "$BOARD:$WORK/$c.in$i.bin" >/dev/null 2>&1; inargs="$inargs --input=\"$s=@$WORK/$c.in$i.bin\""; i=$((i+1)); done
    local ld=""; [ -n "$rlibs" ] && ld="LD_LIBRARY_PATH=$rlibs"
    $SSH "$BOARD" "cd $WORK; $ld $rbin --module=$c.vmfb --function=main --device=torq --torq_hw_type=astra_machina $inargs --output=@$WORK/$c.out.bin" >"$OUT/$c.$rid.log" 2>&1
    $SCP "$BOARD:$WORK/$c.out.bin" "$OUT/$c.$rid.out.bin" >/dev/null 2>&1
    local chk="FAIL: no output"; [ -s "$OUT/$c.$rid.out.bin" ] && chk=$($PY "$CAN/_io.py" check "$c" "$OUT/$c.$rid.out.bin" 2>/dev/null)
    local verdict; verdict=$(classify "$OUT/$c.$rid.log" "$chk")
    [ "$verdict" = PASS ] || PAIR_OK=0
    row="$row  $c=$verdict"
    $SSH "$BOARD" "rm -f $WORK/$c.out.bin" 2>/dev/null
  done
  echo "$row"
}

# ---- arg parse ----
SEL_C=""; SEL_R=""; MATRIX=0
while [ $# -gt 0 ]; do case "$1" in
  --matrix) MATRIX=1;; --compiler) SEL_C="$2"; shift;; --runtime) SEL_R="$2"; shift;;
  *) echo "unknown arg $1"; exit 2;; esac; shift; done

echo "=== torq-verify — canary gate ($BOARD) ==="
lock_get(){ [ -f "$LOCK" ] && grep -oaE "^$1=.*" "$LOCK" | head -n 1 | cut -d= -f2- | sed 's/[[:space:]]*#.*//;s/[[:space:]]*$//'; }

if [ $MATRIX -eq 1 ]; then
  ALL_OK=1
  for cid in "${!COMPILERS[@]}"; do for rid in "${!RUNTIMES[@]}"; do
    run_pair "$cid" "$rid"; [ $PAIR_OK -eq 1 ] || ALL_OK=0
  done; done
  echo ""; echo "legend: PASS | VERSION(rejected) | ZEROS(ran but all-zero) | COMPILE(compiler failed) | ERROR"
  exit 0   # matrix mode is discovery; never gate
fi

# single-pair gate (default = lockfile's pinned compiler + board runtime id)
CID="${SEL_C:-$(lock_get COMPILER_KIND)}"; CID="${CID:-stable-docker}"
RID="${SEL_R:-$(lock_get BOARD_RUNTIME_ID)}"; RID="${RID:-torq2-stable}"
[ -n "${COMPILERS[$CID]:-}" ] || { echo "unknown compiler '$CID'"; exit 2; }
[ -n "${RUNTIMES[$RID]:-}" ] || { echo "unknown runtime '$RID'"; exit 2; }
run_pair "$CID" "$RID"
if [ $PAIR_OK -eq 1 ]; then
  echo "GATE: PASS ($CID × $RID)"
  [ -f "$LOCK" ] && { sed -i "s/^CANARY_STATUS=.*/CANARY_STATUS=PASS/;s/^CANARY_DATE=.*/CANARY_DATE=$(date +%F)/" "$LOCK"; }
  exit 0
else
  echo "GATE: FAIL ($CID × $RID) — do NOT trust real vmfbs on this pair. See rows above."
  exit 1
fi
