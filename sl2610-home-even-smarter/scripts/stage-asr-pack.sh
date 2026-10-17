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
# descriptor + manifest; the stage devices default to the CPU (override: ASR_DEVICES="encoder=torq ...")
python3 -I "$HERE/scripts/finish-dev-pack.py" "$DEST" "$BP/../../../blueprint.json" "$TARGET" "$ID" --lang "$LANG_" \
  $(for d in ${ASR_DEVICES:-frontend=local-task encoder=local-task adapter=local-task decoder=local-task}; do echo --device "$d"; done)
du -sh "$DEST"; find "$DEST" -type f | sed "s|$DEST/||" | sort
