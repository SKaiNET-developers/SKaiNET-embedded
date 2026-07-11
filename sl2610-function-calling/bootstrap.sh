#!/usr/bin/env bash
# bootstrap.sh — one-time setup for the SKaiNET FunctionGemma demo.
#
#   ./bootstrap.sh
#
# Idempotent. Does NOT touch the board and does NOT download anything gated without asking.
# It: (1) creates demo.env from the template, (2) locates/links the Torq toolchain into
# .toolchain/, (3) checks the model files, (4) verifies host prerequisites — then prints the
# exact next commands. Anything it can't do for you (private Torq wheel, model download) it
# tells you how to do.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
say()  { printf '\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[bootstrap] WARN\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[bootstrap]  ok \033[0m %s\n' "$*"; }

# ── 1. demo.env ──────────────────────────────────────────────────────────────────
if [ -f "$ROOT/demo.env" ]; then
    ok "demo.env exists — leaving it untouched."
else
    cp "$ROOT/demo.env.example" "$ROOT/demo.env"
    say "created demo.env from demo.env.example — edit BOARD + GEMMA_GGUF for your setup."
fi
# shellcheck disable=SC1091
. "$ROOT/demo.env"

# ── 2. Torq toolchain (.toolchain/torqpkg) ───────────────────────────────────────
# The g165e12a Torq-fork iree-compile is a PRIVATE, non-PyPI wheel (see docs/TOOLCHAIN-PIN.md).
# We cannot fetch it for you. Resolution order:
#   a) TORQ_PKG already points at a valid unpacked package  -> use it (symlink into .toolchain).
#   b) a wheel at $TORQ_WHEEL (or ./torq_compiler-*.whl)     -> unpack into .toolchain/torqpkg.
#   c) neither                                               -> explain and continue (host-only still works).
TOOLCHAIN="$ROOT/.toolchain"; mkdir -p "$TOOLCHAIN"
resolve_torq() {
    if [ -x "${TORQ_PKG:-}/iree/compiler/_mlir_libs/iree-compile" ]; then
        ok "Torq compiler found at TORQ_PKG=$TORQ_PKG"
        [ -e "$TOOLCHAIN/torqpkg" ] || ln -sfn "$TORQ_PKG" "$TOOLCHAIN/torqpkg"
        return 0
    fi
    local whl="${TORQ_WHEEL:-$(ls "$ROOT"/torq_compiler-*.whl 2>/dev/null | head -n1 || true)}"
    if [ -n "$whl" ] && [ -f "$whl" ]; then
        say "unpacking Torq wheel $whl -> .toolchain/torqpkg"
        rm -rf "$TOOLCHAIN/torqpkg.tmp"; mkdir -p "$TOOLCHAIN/torqpkg.tmp"
        ( cd "$TOOLCHAIN/torqpkg.tmp" && python3 -m zipfile -e "$whl" . 2>/dev/null || unzip -q "$whl" )
        rm -rf "$TOOLCHAIN/torqpkg"; mv "$TOOLCHAIN/torqpkg.tmp" "$TOOLCHAIN/torqpkg"
        ok "Torq compiler unpacked."
        return 0
    fi
    return 1
}
if resolve_torq; then :; else
    warn "Torq toolchain not found. Board (aarch64/NPU) compiles need it."
    warn "  Get the g165e12a torq_compiler package (Synaptics AI Developer portal / your Coral build),"
    warn "  then either:  export TORQ_PKG=/path/to/unpacked/torqpkg   (in demo.env), or"
    warn "                drop torq_compiler-*.whl next to bootstrap.sh and re-run."
    warn "  Host-only (x64 llvm-cpu) validation via docker still works without it."
fi

# ── 3. Models ────────────────────────────────────────────────────────────────────
if [ -f "${GEMMA_GGUF:-}" ]; then
    ok "FunctionGemma GGUF: $GEMMA_GGUF"
else
    warn "FunctionGemma GGUF not found at GEMMA_GGUF='${GEMMA_GGUF:-}'."
    warn "  Set GEMMA_GGUF in demo.env to your functiongemma-*-Q5_K_M.gguf (or your finetuned one)."
fi
[ -d "$ROOT/weights" ] && ok "Moonshine weights/ present." || \
    warn "Moonshine weights/ absent — see docs (extract from the moonshine-tiny checkpoint) if self-compiling ASR."

# ── 4. Host prerequisites ────────────────────────────────────────────────────────
for tool in docker ssh; do
    command -v "$tool" >/dev/null 2>&1 && ok "$tool present" || warn "$tool not on PATH (needed for compile/deploy)."
done
[ -x "$ROOT/gradlew" ] && ok "gradlew present" || warn "gradlew missing?"

# ── Next steps ───────────────────────────────────────────────────────────────────
cat <<EOF

$(say "setup checked. Next:")
  1) edit demo.env            (BOARD ip, GEMMA_GGUF path)
  2) ./gradlew :jvmRun        # host smoke test (no board needed)
  3) GEMMA_GGUF=... scripts/compile-gemma.sh board   # self-compile FunctionGemma for the board
  4) BOARD=root@<ip> ./gradlew deployBoard           # push + run on the SL2610
  finetune on your own commands:  docs/FINETUNING.md
EOF
