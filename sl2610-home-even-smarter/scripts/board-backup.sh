#!/usr/bin/env bash
# Copies what a reflash would destroy (the eMMC image rewrites the whole /home partition): everything under /home
# except large trees that can be regenerated from the host or downloaded again. Read-only on the board; writes to DEST.
#   BOARD=root@<ip> scripts/board-backup.sh [DEST]          (default DEST: build/board-backup/<date>)
# The skipped trees are listed in the inventory; FULL=1 includes them too.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BOARD="${BOARD:?set BOARD=root@<board-ip>}"
DEST="${1:-$HERE/build/board-backup/$(date +%Y%m%d-%H%M)}"
# relative to /home: compiled experiments, Python venvs, downloadable releases, installed and test packs
SKIP=(root/ireetest root/ireetest-bf16 root/fcvenv root/venv-torqstable root/sl2610-voice-cc root/torq-v2.0.0-official
      root/hes-p3 root/hes-p4 root/.homeevensmarter root/.cache/pip lost+found)
mkdir -p "$DEST"

echo "==> inventory"
ssh "$BOARD" 'cd /home && du -sk * root/* root/.[a-z]* 2>/dev/null | sort -n' | tee "$DEST/inventory.txt"
ssh "$BOARD" 'cat /etc/build*; uname -a; modinfo syna_npu; tr -d "\0" < /proc/device-tree/model' > "$DEST/board-info.txt" 2>&1 || true

excludes=(); [ "${FULL:-0}" = 1 ] || for s in "${SKIP[@]}"; do excludes+=("--exclude=$s"); done
echo "==> /home"
ssh "$BOARD" "cd /home && tar cf - ${excludes[*]} ." | tar xf - -C "$DEST"
[ "${FULL:-0}" = 1 ] || echo "skipped (regenerable; FULL=1 to include): ${SKIP[*]}"
( cd "$DEST" && find . -type f ! -name SHA256SUMS ! -name inventory.txt ! -name board-info.txt -exec sha256sum {} + > SHA256SUMS )
echo "backup in $DEST ($(du -sh "$DEST" | cut -f1)); checksums in SHA256SUMS"
