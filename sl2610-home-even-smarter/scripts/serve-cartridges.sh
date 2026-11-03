#!/usr/bin/env bash
# Serve the materialized cartridges to the board over the network.  scripts/serve-cartridges.sh [dir] [port]
# Point the board app at it: hes-tui --server http://<host-ip>:<port>, then Tab → i (fetch) → d (download).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$(cd "${1:-$HERE/build/cartridges}" 2>/dev/null && pwd)" || { echo "no cartridges at ${1:-$HERE/build/cartridges}"; exit 1; }
PORT="${2:-8080}"
cd "$HERE" && CARTRIDGES_DIR="$DIR" PORT="$PORT" exec ./gradlew -q :server:run
