#!/usr/bin/env bash
# Builds every board ASR variant with the blueprint recipe and stages it, plus the NLU, into the directory the cartridge
# server serves (default build/cartridges), so the TUI can list, download and switch them.
#   scripts/stage-variants.sh [dest]         (ASR_TARGETS="torq-sl26xx sl26xx-cpu-int8 cpu-sl26xx" by default)
# Each ASR target has a profile in cartridges/profiles/asr-moonshine.<target>.en.json. The NLU pack is staged once
# with scripts/stage-nlu-pack.sh (its recipe run is listed in docs/BOARD-UPDATE.md, Step 4) and copied from build/packs.
# Then: scripts/serve-cartridges.sh build/cartridges 8089, and on the board hes-tui --server http://<host-ip>:8089
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$HERE/build/cartridges}"
BLU="${BLUEPRINTS_DIR:-$HERE/../../SKaiNET-cartridge-blueprints}"
A=:blueprints:asr-moonshine-v2-streaming-iree
mkdir -p "$DEST"
for T in ${ASR_TARGETS:-torq-sl26xx sl26xx-cpu-int8 cpu-sl26xx}; do
  echo "==> $T"
  # the recipe's work directory holds one target at a time: stage each right after its build
  (cd "$BLU" && ./gradlew -q $A:compileEnFrontend $A:compileEnFrontendStep $A:compileEnEncoder $A:compileEnAdapter \
    $A:compileEnPrefill $A:compileEnWithPast $A:weightsEn -Pprofile="$HERE/cartridges/profiles/asr-moonshine.$T.en.json")
  "$HERE/scripts/stage-asr-pack.sh" "$T" en "$DEST/asr-moonshine-v2-streaming-iree-$T-dev" | tail -n 1
done
NLU=nlu-functiongemma-270m-iree-cpu-sl26xx-dev
if [ -d "$HERE/build/packs/$NLU" ]; then
  [ -d "$DEST/$NLU" ] || cp -r "$HERE/build/packs/$NLU" "$DEST/"
else
  echo "no $NLU in build/packs: run scripts/stage-nlu-pack.sh first (docs/BOARD-UPDATE.md, Step 4)"
fi
ls "$DEST"
