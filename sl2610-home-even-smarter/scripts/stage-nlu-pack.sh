#!/usr/bin/env bash
# Assembles an UNSIGNED development pack_dir for the FunctionGemma NLU from the blueprint recipe's outputs
# (export → host-gather → parameters → compile for a board target), for board checks before the signed
# materializeCartridge pack exists for that target.
#   scripts/stage-nlu-pack.sh [target] [dest]     (defaults: cpu-sl26xx, build/packs/nlu-functiongemma-270m-iree-<target>-dev)
# Run the recipe first, in SKaiNET-cartridge-blueprints:
#   ./gradlew :blueprints:nlu-functiongemma-270m-iree:{compile,parameters}{WithPast,PrefillWithPast,PrefillAt} -Pprofile=<profile>
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-cpu-sl26xx}"
ID="nlu-functiongemma-270m-iree-$TARGET-dev"
DEST="${2:-$HERE/build/packs/$ID}"
BP="${BLUEPRINTS_DIR:-$HERE/../../SKaiNET-cartridge-blueprints}/blueprints/nlu-functiongemma-270m-iree/build/blueprint"
mkdir -p "$DEST/artifacts/model" "$DEST/artifacts/weights" "$DEST/artifacts/tokenizer" "$DEST/artifacts/other"
for g in with-past prefill-with-past prefill-at; do
  cp "$BP/work/compile/$TARGET/gemma-$g.vmfb" "$DEST/artifacts/model/"
  ln -f "$BP/work/parameters/gemma-$g.irpa" "$DEST/artifacts/weights/gemma-$g.irpa" 2>/dev/null || cp "$BP/work/parameters/gemma-$g.irpa" "$DEST/artifacts/weights/"
done
cp "$BP/sources/weights/functiongemma-270m-it-Q8_0.gguf" "$DEST/artifacts/tokenizer/"
cp "$HERE/cartridges/catalogs/home-tools.v1.json" "$DEST/artifacts/other/tool-catalog.json"
# descriptor + manifest (hashes 1.6 GB of parameters: ~10 s); the decoder device defaults to the CPU
python3 -I "$HERE/scripts/finish-dev-pack.py" "$DEST" "$BP/../../blueprint.json" "$TARGET" "$ID" \
  --device "decoder=${NLU_DEVICE:-local-task}"
du -sh "$DEST"; find "$DEST" -type f | sed "s|$DEST/||" | sort
