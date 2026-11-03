#!/usr/bin/env bash
# Deploy r1_smoke + smoke vmfbs to the board (/tmp/r1) and run them in one process.
# Optional: ENCODER=<path on board> runs the prebuilt exec-v0 encoder on torq as well
# (needs the opt-in patch-accept-exec-v0.diff build + TORQ_ACCEPT_EXEC_V0=1).
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd); H=${BOARD:?set BOARD=root@<board-ip>}
ssh "$H" 'mkdir -p /tmp/r1'
for f in r1_smoke add_f32.torqfork.cpu.vmfb add_bf16.torq.vmfb; do ssh "$H" "cat > /tmp/r1/$f" < "$D/out/$f"; done
ssh "$H" 'chmod +x /tmp/r1/r1_smoke'
if [ -n "${ENCODER:-}" ]; then
  ssh "$H" "cd /tmp/r1 && TORQ_ACCEPT_EXEC_V0=1 ./r1_smoke add_f32.torqfork.cpu.vmfb add_bf16.torq.vmfb $ENCODER module.main"
else
  ssh "$H" 'cd /tmp/r1 && ./r1_smoke add_f32.torqfork.cpu.vmfb add_bf16.torq.vmfb'
fi
