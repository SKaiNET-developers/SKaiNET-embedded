#!/usr/bin/env bash
# Assembles an UNSIGNED development pack_dir for the Moonshine v2 streaming ASR from the blueprint recipe's outputs
# (export → parameters → compile for a board target), in the layout the blueprint's PackDir reads, for board checks
# before the signed materializeCartridge pack exists for that target.
#   scripts/stage-asr-pack.sh [target] [lang] [dest]
#     (defaults: cpu-sl26xx, en, build/packs/asr-moonshine-v2-streaming-iree-<target>-dev)
# Run the recipe first, in SKaiNET-cartridge-blueprints:
#   ./gradlew :blueprints:asr-moonshine-v2-streaming-iree:{compile,parameters}<Lang>... -Pprofile=<profile>
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-cpu-sl26xx}"
LANG_="${2:-en}"
ID="asr-moonshine-v2-streaming-iree-$TARGET-dev"
DEST="${3:-$HERE/build/packs/$ID}"
BP="${BLUEPRINTS_DIR:-$HERE/../../SKaiNET-cartridge-blueprints}/blueprints/asr-moonshine-v2-streaming-iree/build/blueprint/work"
M="$DEST/artifacts/model/$LANG_"; W="$DEST/artifacts/weights/$LANG_"; T="$DEST/artifacts/tokenizer/$LANG_"
mkdir -p "$M" "$W" "$T"
for g in frontend encoder adapter prefill step; do cp "$BP/compile/$TARGET/$LANG_/$g.vmfb" "$M/"; done
cp "$BP/parameters/$LANG_/params.irpa" "$W/"
cp "$BP/export/$LANG_/dec_embed.bin" "$W/"
cp "$BP/export/$LANG_/vocab.bin" "$T/"
cat > "$DEST/descriptor.json" <<JSON
{"id": "$ID", "version": "0.0.0-dev", "task": "asr", "target": {"hardware": "IREE-linux-aarch64 (SL26xx, Cortex-A55)", "abi": "linux-aarch64"}, "attributes": {"languages": ["$LANG_"]}, "note": "unsigned development pack from the $TARGET recipe outputs"}
JSON
du -sh "$DEST"; find "$DEST" -type f | sed "s|$DEST/||" | sort
