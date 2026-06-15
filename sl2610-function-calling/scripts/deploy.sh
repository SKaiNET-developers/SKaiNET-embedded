#!/usr/bin/env bash
# Push the linuxArm64 binary to the board and run it (from the HOST).
#   BOARD=root@<ip> sh scripts/deploy.sh [--run] [-- <app args>]
# Streams the binary over ssh (BusyBox board has no rsync/sftp; cat is enough).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BOARD=${BOARD:-root@192.168.3.26}
DEST=${DEST:-/home/root/voicecc-kt}
SSH_OPTS=${SSH_OPTS:--o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10}

# Prefer the release binary; fall back to debug.
BIN=${BIN:-$ROOT/build/bin/linuxArm64/releaseExecutable/sl2610-function-calling.kexe}
[ -f "$BIN" ] || BIN=$ROOT/build/bin/linuxArm64/debugExecutable/sl2610-function-calling.kexe
[ -f "$BIN" ] || { echo "no linuxArm64 binary; run ./gradlew linkReleaseExecutableLinuxArm64" >&2; exit 1; }

echo "[deploy] $BIN -> $BOARD:$DEST/voicecc"
ssh $SSH_OPTS "$BOARD" "mkdir -p '$DEST' && cat > '$DEST/voicecc' && chmod +x '$DEST/voicecc'" < "$BIN"

# Kotlin/Native links libcrypt.so.1 (glibc); the board has libxcrypt's
# libcrypt.so.2. Provide a soname-compat symlink in the deploy dir and run
# with LD_LIBRARY_PATH so nothing in the system tree is touched.
ssh $SSH_OPTS "$BOARD" "
  d='$DEST'
  if [ ! -e \"\$d/libcrypt.so.1\" ]; then
    src=\$(ls /usr/lib/libcrypt.so.2* /lib/libcrypt.so.2* 2>/dev/null | head -n1)
    [ -n \"\$src\" ] && ln -sf \"\$src\" \"\$d/libcrypt.so.1\" && echo \"[deploy] libcrypt.so.1 -> \$src\"
  fi
"
echo "[deploy] done."

if [ "${1:-}" = "--run" ]; then
    shift
    [ "${1:-}" = "--" ] && shift
    echo "[deploy] running on board:"
    ssh $SSH_OPTS "$BOARD" "LD_LIBRARY_PATH='$DEST' '$DEST/voicecc' $*"
fi
