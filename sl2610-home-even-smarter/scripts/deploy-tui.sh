#!/usr/bin/env bash
# Cross-builds the terminal app for linuxArm64 and installs it on a dev board over ssh.
#   BOARD=root@<ip> scripts/deploy-tui.sh [--no-build] [--run [hes-tui args…]]
# The board has a BusyBox userland (no rsync/sftp), so the binary is streamed with `ssh cat`. Kotlin/Native links
# libcrypt.so.1 while the board ships libcrypt.so.2; a private lib/ directory with a symlink bridges that without
# touching system directories. Run it later with: ssh -t $BOARD $DEST/hes-tui [--mic plughw:1,0]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BOARD="${BOARD:?set BOARD=root@<board-ip>}"
DEST="${DEST:-/home/root/hes-tui}"
BUILD=1; RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --no-build) BUILD=0; shift ;;
    --run) RUN=1; shift; break ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
BIN="$HERE/tui/build/bin/linuxArm64/releaseExecutable/hes-tui.kexe"
if [ "$BUILD" = 1 ]; then (cd "$HERE" && ./gradlew -q :tui:linkReleaseExecutableLinuxArm64); fi
[ -f "$BIN" ] || { echo "no binary at $BIN"; exit 1; }

echo "==> $BOARD:$DEST ($(du -h "$BIN" | cut -f1))"
ssh "$BOARD" "mkdir -p '$DEST/lib' && cat > '$DEST/hes-tui.kexe.new' && chmod +x '$DEST/hes-tui.kexe.new' && mv '$DEST/hes-tui.kexe.new' '$DEST/hes-tui.kexe'" < "$BIN"
ssh "$BOARD" "set -e
  crypt=\$(ls /usr/lib/libcrypt.so.2 /lib/libcrypt.so.2 2>/dev/null | head -n 1)
  [ -n \"\$crypt\" ] && ln -sf \"\$crypt\" '$DEST/lib/libcrypt.so.1'
  printf '%s\n' '#!/bin/sh' 'export LD_LIBRARY_PATH=\"$DEST/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"' 'export TERM=\"\${TERM:-xterm-256color}\"' 'exec \"$DEST/hes-tui.kexe\" \"\$@\"' > '$DEST/hes-tui'
  chmod +x '$DEST/hes-tui'"
echo "installed: ssh -t $BOARD $DEST/hes-tui --help"
if [ "$RUN" = 1 ]; then exec ssh -t "$BOARD" "$DEST/hes-tui" "$@"; fi
