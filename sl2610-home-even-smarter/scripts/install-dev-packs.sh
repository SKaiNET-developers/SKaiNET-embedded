#!/usr/bin/env bash
# Installs pack_dirs into the board app's cartridge store (~/.homeevensmarter/cartridges/<id>/), where hes-tui and
# hes-app-check find them. Only files that are missing on the board or differ in size are sent (BusyBox has no rsync),
# so re-running after a restage moves just what changed.
#   scripts/install-dev-packs.sh [pack-dir…]     (default: every pack under build/packs)
#   BOARD=root@192.168.3.26 STORE=/home/root/.homeevensmarter/cartridges
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BOARD="${BOARD:-root@192.168.3.26}"
STORE="${STORE:-/home/root/.homeevensmarter/cartridges}"
PACKS=("$@"); [ ${#PACKS[@]} -gt 0 ] || PACKS=("$HERE"/build/packs/*/)
for pack in "${PACKS[@]}"; do
  pack="${pack%/}"; id="$(basename "$pack")"
  remote="$(ssh "$BOARD" "mkdir -p '$STORE/$id' && cd '$STORE/$id' && find . -type f -exec stat -c '%s %n' {} +" || true)"
  sent=0
  while IFS= read -r rel; do
    size="$(stat -c %s "$pack/$rel")"
    if ! grep -qxF "$size ./$rel" <<<"$remote"; then
      ssh "$BOARD" "mkdir -p '$STORE/$id/$(dirname "$rel")' && cat > '$STORE/$id/$rel'" < "$pack/$rel"
      sent=$((sent + size))
    fi
  done < <(cd "$pack" && find . -type f | sed 's|^\./||' | sort)
  # files the pack no longer has
  for stale in $(comm -13 <(cd "$pack" && find . -type f | sort) <(sed 's/^[0-9]* //' <<<"$remote" | sort)); do
    ssh "$BOARD" "rm -f '$STORE/$id/$stale'"
  done
  echo "$id: sent $((sent / 1000)) KB"
done
