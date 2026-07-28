#!/usr/bin/env bash
# run13_00_prepare_corpus.sh — Build a clean corpus dir WITHOUT touching
# 01_build_index.py: symlink the real PDFs into data/corpus_run13/,
# excluding Jupyter checkpoint duplicates and any duplicate filename.
#
# Root cause of the 42.3% index contamination: data/corpus/ contains
# .ipynb_checkpoints/ copies (and/or *-checkpoint files) that 01 globbed.
# Fix at the SOURCE (corpus dir), not in the code — fully additive.
#
# Usage:  bash run13_00_prepare_corpus.sh [SRC_DIR] [DST_DIR]
# Default: SRC=data/corpus  DST=data/corpus_run13
set -e

SRC="${1:-data/corpus}"
DST="${2:-data/corpus_run13}"

mkdir -p "$DST"
rm -f "$DST"/*.pdf 2>/dev/null || true

n=0
declare -A seen
while IFS= read -r -d '' f; do
    base="$(basename "$f")"
    # skip anything checkpoint-related
    case "$f" in
        *ipynb_checkpoints*|*-checkpoint*) continue ;;
    esac
    # skip duplicate basenames
    if [[ -n "${seen[$base]:-}" ]]; then continue; fi
    seen[$base]=1
    ln -s "$(readlink -f "$f")" "$DST/$base"
    n=$((n+1))
done < <(find "$SRC" -type f -iname '*.pdf' -print0)

echo "=============================================="
echo "run13 corpus prepared: $DST"
echo "PDFs linked : $n   (expected: 37)"
echo "=============================================="
if [ "$n" -ne 37 ]; then
    echo "WARNING: count != 37 — inspect before building the index:"
    ls "$DST"
fi