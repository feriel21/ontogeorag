#!/usr/bin/env python3
"""
rebuild_corpus_run13.py — Deterministic corpus rebuild from run11 doc ids.
==========================================================================
Guarantees run13 indexes EXACTLY the papers run11 indexed (comparability of
the contamination ablation), and eliminates filename pathologies (newlines,
trailing spaces) that break `ls | wc -l` checks.

Steps:
  1. Read the 37 normalized doc ids from run11 chunks.jsonl (ground truth).
  2. Inspect current data/corpus_run13 entries with repr() to expose any
     pathological name.
  3. Wipe and rebuild data/corpus_run13: for each run11 doc id, locate the
     matching PDF under data/full_corpus (recursive, checkpoints excluded,
     exact stem match, then whitespace-normalized fallback) and symlink it
     under a SANITIZED name (doc id + '.pdf').
  4. Verify final count via os.listdir.

NOTE: sanitized link names must still yield the same doc_id in 01_build_index
(doc_id = filename stem) -> we name links exactly '<run11_doc_id>.pdf', so
run13 doc_ids will match run11 doc_ids by construction.

USAGE (from ~/ontogeorag):
    python rebuild_corpus_run13.py
"""

import json
import os
import re
import sys
from pathlib import Path

CHUNKS = "output/step1/chunks.jsonl"
SRC = "data/full_corpus"
DST = Path("data/corpus_run13")


def normspace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main():
    # 1. ground truth: run11 doc ids
    docs = set()
    for line in open(CHUNKS, encoding="utf-8"):
        d = json.loads(line)["doc_id"]
        docs.add(re.sub(r"-checkpoint$", "", d))
    docs = sorted(docs)
    print(f"[1] run11 ground truth: {len(docs)} papers")

    # 2. inspect current corpus_run13 for pathological names
    if DST.exists():
        entries = os.listdir(DST)
        print(f"[2] current corpus_run13: {len(entries)} entries (os.listdir)")
        for e in entries:
            if ("\n" in e or "\r" in e or e != e.strip()
                    or not e.lower().endswith(".pdf")):
                print(f"    PATHOLOGICAL NAME: {e!r}")

    # 3. index source PDFs by stem
    src_by_stem = {}
    src_by_norm = {}
    for root, dirs, files in os.walk(SRC):
        dirs[:] = [d for d in dirs if "checkpoint" not in d.lower()]
        for f in files:
            if not f.lower().endswith(".pdf") or "checkpoint" in f.lower():
                continue
            stem = os.path.splitext(f)[0]
            p = os.path.abspath(os.path.join(root, f))
            src_by_stem[stem] = p
            src_by_norm[normspace(stem)] = p
    print(f"[3] source PDFs under {SRC}: {len(src_by_stem)}")

    # 4. rebuild
    if DST.exists():
        for e in os.listdir(DST):
            os.unlink(DST / e)
    DST.mkdir(parents=True, exist_ok=True)

    unmatched = []
    for d in docs:
        p = src_by_stem.get(d) or src_by_norm.get(normspace(d))
        if p is None:
            unmatched.append(d)
            continue
        os.symlink(p, DST / f"{d}.pdf")

    final = [e for e in os.listdir(DST)]
    print(f"[4] rebuilt: {len(final)} symlinks (expected {len(docs)})")
    if unmatched:
        print("    UNMATCHED run11 docs (no source PDF found):")
        for d in unmatched:
            print(f"      {d!r}")
        sys.exit(1)
    bad = [e for e in final if "\n" in e or e != e.strip()]
    assert not bad, f"pathological names remain: {bad!r}"
    print("[OK] corpus_run13 == run11 paper set, sanitized names. "
          f"Ready: N_PAPERS_EXPECTED={len(final)}")


if __name__ == "__main__":
    main()