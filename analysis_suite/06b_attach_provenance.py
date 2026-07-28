#!/usr/bin/env python3
"""
06b_attach_provenance.py — Re-attach native provenance lost at fusion.
======================================================================
CONTEXT (verified on cluster)
    Extraction stores `_provenance.selected_chunks` (format
    "docname::chunkN") and the field SURVIVES through 03/04/05
    (canonical_triples_v5.jsonl still has it). It is dropped only by
    06_tiered_fusion.py, which writes support_count=0. This script joins the
    fused KG back to the per-pass canonical files and re-attaches, without
    modifying 06.

WHAT each fused triple gains:
    retrieved_chunk_ids : union of selected_chunks across both passes
                          (RETRIEVAL provenance — the passages the LLM saw;
                          NOT a claim that each supports the triple)
    retrieved_papers    : distinct papers derived from those ids
                          ('-checkpoint' copies collapse onto the real paper)
    n_retrieved_chunks / n_retrieved_papers

    Join key: (subject, relation, object) normalized lowercase; both fused
    formats (run11 list-of-pairs and M4 {tier1,tier2,...}) are supported via
    kg_io. Unmatched fused triples are reported (should be ~0 since fusion
    inputs ARE these files).

    NOTE ON SEMANTICS (for the manuscript): retrieval provenance ≠ evidence
    provenance. The single supporting chunk per triple remains the one from
    evidence re-anchoring (08). 08 combines three channels: evidence anchor
    (support), retrieved chunks (context), co-occurrence (consensus bound).

USAGE (run11 retrofit, or run13 after fusion)
    python analysis_suite/06b_attach_provenance.py \
        --kg output/run11_kg/tiered_kg_run11.json \
        --pass-a output/run11_a/canonical_triples_v5.jsonl \
        --pass-b output/run11_b/canonical_triples_v5.jsonl \
        --out output/run11_kg/tiered_kg_run11_prov.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from kg_io import (load_kg, dump_kg, get_subject, get_object, get_relation,
                   get_chunk_ids, normalize_paper_id)


def key(s, r, o):
    n = lambda x: re.sub(r"\s+", " ", str(x).lower()).strip()
    return (n(s), n(r), n(o))


def paper_of(chunk_id: str) -> str:
    base = chunk_id.split("::")[0] if "::" in chunk_id else chunk_id
    return normalize_paper_id(base)


def load_pass(path):
    idx = defaultdict(list)
    n = 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)
        s = t.get("subject") or t.get("source")
        o = t.get("object") or t.get("target")
        r = t.get("relation")
        cids = get_chunk_ids(t)
        if s and o and r:
            idx[key(s, r, o)].extend(cids)
            n += 1
    return idx, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--pass-a", required=True)
    ap.add_argument("--pass-b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    kg = load_kg(args.kg)
    idx_a, na = load_pass(args.pass_a)
    idx_b, nb = load_pass(args.pass_b)
    print(f"pass A: {na} triples indexed | pass B: {nb} triples indexed")

    matched = unmatched = 0
    total_papers = set()
    for t in kg["triples"]:
        k = key(get_subject(t), get_relation(t), get_object(t))
        cids = []
        for c in idx_a.get(k, []) + idx_b.get(k, []):
            if c not in cids:
                cids.append(c)
        if cids:
            matched += 1
        else:
            unmatched += 1
        papers = []
        for c in cids:
            p = paper_of(c)
            if p not in papers:
                papers.append(p)
        total_papers.update(papers)
        t["retrieved_chunk_ids"] = cids
        t["retrieved_papers"] = papers
        t["n_retrieved_chunks"] = len(cids)
        t["n_retrieved_papers"] = len(papers)

    dump_kg(kg, args.out)
    print("=" * 60)
    print("NATIVE PROVENANCE ATTACHED")
    print("=" * 60)
    print(f"fused triples matched   : {matched}")
    print(f"fused triples unmatched : {unmatched}"
          + ("  <-- inspect (surface drift between fusion in/out?)"
             if unmatched else ""))
    print(f"distinct papers touched : {len(total_papers)}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()