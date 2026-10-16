#!/usr/bin/env bash
# Copies what a reflash would destroy and cannot be regenerated from the host: vendor prebuilt models, example
# assets, deployed runtimes and hand-made test files under /home/root. Read-only on the board; writes to DEST.
#   BOARD=root@<ip> scripts/board-backup.sh [DEST]          (default DEST: build/board-backup/<date>)
# Large regenerable trees (our compiled vmfb/irpa experiments, Python venvs) are listed but skipped unless FULL=1.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BOARD="${BOARD:?set BOARD=root@<board-ip>}"
DEST="${1:-$HERE/build/board-backup/$(date +%Y%m%d-%H%M)}"
KEEP=(moon sl2610-examples demos torq2-stable torq2 settings cmd.wav)
SKIP=(ireetest ireetest-bf16 fcvenv venv-torqstable sl2610-voice-cc torq-v2.0.0-official)
mkdir -p "$DEST"

echo "==> inventory"
ssh "$BOARD" 'cd /home/root && du -sk * 2>/dev/null | sort -n' | tee "$DEST/inventory.txt"
ssh "$BOARD" 'cat /etc/build*; uname -a; modinfo syna_npu; tr -d "\0" < /proc/device-tree/model' > "$DEST/board-info.txt" 2>&1 || true

items=("${KEEP[@]}"); [ "${FULL:-0}" = 1 ] && items+=("${SKIP[@]}")
for item in "${items[@]}"; do
  if ssh "$BOARD" "[ -e '/home/root/$item' ]"; then
    echo "==> $item"
    ssh "$BOARD" "cd /home/root && tar cf - '$item'" | tar xf - -C "$DEST"
  fi
done
[ "${FULL:-0}" = 1 ] || echo "skipped (regenerable; FULL=1 to include): ${SKIP[*]}"
( cd "$DEST" && find . -type f ! -name SHA256SUMS -exec sha256sum {} + > SHA256SUMS )
echo "backup in $DEST ($(du -sh "$DEST" | cut -f1)); checksums in SHA256SUMS"
