#!/usr/bin/env python3
"""
run13_build_index_from_run11.py — Deduplicated index WITHOUT re-parsing PDFs.
=============================================================================
CONTEXT
    pipeline/01_build_index.py leaks memory in the current environment
    (~1.7 GB/3s on the first PDF, regardless of file — environment drift,
    since run11 built fine historically). Re-parsing is also scientifically
    unnecessary AND undesirable: the run11 chunks already exist, and reusing
    them makes the run11/run13 ablation perfectly controlled — chunking is
    bit-identical, ONLY the deduplication differs.

WHAT
    Reads output/step1/chunks.jsonl (run11), removes Jupyter checkpoint
    duplicates, writes output/run13/step1/chunks.jsonl in the exact same
    record format (02_extract_triples.py rebuilds BM25 from it unchanged).

    Rules, per normalized paper (doc_id minus '-checkpoint'):
      * prefer records whose doc_id is already the normalized form;
      * if a paper exists ONLY as a checkpoint copy, keep those records but
        rewrite doc_id (and the doc prefix inside chunk_id) to the
        normalized name, so the run13 namespace is checkpoint-free;
      * drop exact duplicate texts within a paper (hash on normalized text).

    Verifies at the end: 37 papers, 0 duplicate (doc,text) pairs, and prints
    the run11->run13 chunk count per paper for the ablation table.

USAGE (from ~/ontogeorag; CPU, seconds)
    python run13_build_index_from_run11.py
    # optional overrides:
    #   --src output/step1/chunks.jsonl --dst output/run13/step1/chunks.jsonl
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


def norm_doc(d: str) -> str:
    return re.sub(r"-checkpoint$", "", d)


def norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="output/step1/chunks.jsonl")
    ap.add_argument("--dst", default="output/run13/step1/chunks.jsonl")
    ap.add_argument("--expect-papers", type=int, default=37)
    args = ap.parse_args()

    by_doc = defaultdict(lambda: {"clean": [], "checkpoint": []})
    n_src = 0
    for line in open(args.src, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        n_src += 1
        d = rec["doc_id"]
        by_doc[norm_doc(d)][
            "checkpoint" if d.endswith("-checkpoint") else "clean"
        ].append(rec)

    out, per_paper = [], {}
    n_renamed_docs = 0
    for doc in sorted(by_doc):
        pools = by_doc[doc]
        if pools["clean"]:
            records = pools["clean"]
        else:  # paper existed only as its checkpoint copy: rename
            n_renamed_docs += 1
            records = []
            for rec in pools["checkpoint"]:
                rec = dict(rec)
                old = rec["doc_id"]
                rec["doc_id"] = doc
                if "chunk_id" in rec and isinstance(rec["chunk_id"], str):
                    rec["chunk_id"] = rec["chunk_id"].replace(old, doc)
                records.append(rec)
        seen = set()
        kept = []
        for rec in records:
            h = hashlib.md5(
                norm_text(rec.get("text", "")).encode()
            ).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            kept.append(rec)
        per_paper[doc] = (
            len(pools["clean"]) + len(pools["checkpoint"]),
            len(kept),
        )
        out.extend(kept)

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta = {
        "built_from": args.src,
        "method": "run11 chunks deduplicated "
        "(checkpoint removal + per-paper text-hash dedup); chunking "
        "bit-identical to run11 by construction",
        "n_source_records": n_src,
        "n_output_records": len(out),
        "n_papers": len(by_doc),
        "n_checkpoint_only_papers_renamed": n_renamed_docs,
    }
    with open(
        dst.parent / "index_meta_run13.json", "w", encoding="utf-8"
    ) as f:
        json.dump(meta, f, indent=2)

    # final integrity check (same as pipeline stage-2 check)
    papers, hashes, dup = set(), set(), 0
    for rec in out:
        papers.add(rec["doc_id"])
        h = hashlib.md5(
            (rec["doc_id"] + rec.get("text", "")).encode()
        ).hexdigest()
        dup += h in hashes
        hashes.add(h)

    print("=" * 62)
    print("RUN13 INDEX (from run11 chunks, deduplicated)")
    print("=" * 62)
    print(f"source records : {n_src}")
    print(f"output records : {len(out)}")
    print(f"papers         : {len(papers)} (expected {args.expect_papers})")
    print(f"dup (doc,text) : {dup} (expected 0)")
    print(f"checkpoint-only papers renamed: {n_renamed_docs}")
    print("chunks per paper (run11 total -> run13 kept), lowest 6:")
    for doc, (a, b) in sorted(per_paper.items(), key=lambda x: x[1][1])[:6]:
        print(f"  {a:4d} -> {b:4d}  {doc}")
    assert dup == 0 and len(papers) == args.expect_papers, "INTEGRITY FAIL"
    print(f"[OK] written: {dst}")
    print(
        "Next: FROM_STAGE=3 N_PAPERS_EXPECTED=37 "
        "bash analysis_suite/run13_pipeline_v2.sh"
    )


if __name__ == "__main__":
    main()
