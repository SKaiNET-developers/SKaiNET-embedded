#!/usr/bin/env bash
# Fails when any tracked or staged file mentions a term from the private word list .check-terms.local
# (one case-insensitive term per line; the list itself is git-ignored, so the terms never enter the repository).
# Run before every commit: scripts/check-terms.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LIST="${CHECK_TERMS_FILE:-$HERE/.check-terms.local}"
[ -f "$LIST" ] || { echo "no word list at $LIST — nothing checked"; exit 0; }
PATTERN="$(grep -v '^\s*$' "$LIST" | grep -v '^#' | paste -sd'|' -)"
[ -n "$PATTERN" ] || exit 0
cd "$HERE"
FILES="$( (git ls-files; git diff --cached --name-only --diff-filter=AM) | sort -u)"
HITS="$(echo "$FILES" | xargs -r grep -n -i -E -- "$PATTERN" 2>/dev/null || true)"
if [ -n "$HITS" ]; then echo "forbidden terms found:"; echo "$HITS"; exit 1; fi
echo "check-terms: clean ($(echo "$FILES" | wc -l) files)"
