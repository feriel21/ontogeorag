#!/usr/bin/env python3
"""Standalone recall verification for OntoGeoRAG run11 (76.5% = 26/34)."""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable


RELATION_MAP = {
    "hasdescriptor": "hasDescriptor",
    "occursin":      "occursIn",
    "partof":        "partOf",
    "formedby":      "formedBy",
    "causedby":      "causedBy",
    "triggers":      "triggers",
    "causes":        "causes",
    "controls":      "controls",
    "overlies":      "overlies",
    "underlies":     "underlies",
    "associatedwith": "associatedWith",
}


def normalize_relation(rel: str) -> str:
    if not rel:
        return ""
    key = rel.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    return RELATION_MAP.get(key, rel.strip())


def norm_text(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").lower().strip())
    return s.rstrip(".,;:")


def ref_matches_triple(ref_edge: dict, triple: dict) -> bool:
    rs = norm_text(ref_edge.get("subject", ""))
    ro = norm_text(ref_edge.get("object", ""))
    rr = normalize_relation(ref_edge.get("relation", ""))
    ts = norm_text(triple.get("subject", ""))
    to = norm_text(triple.get("object", ""))
    tr = normalize_relation(triple.get("relation", ""))
    return (rs in ts) and (ro in to) and (rr == tr)


def count_hits(ref_edges: Iterable[dict], triples: Iterable[dict]):
    triples = list(triples)
    matched_pairs = []
    unmatched = []
    for r in ref_edges:
        match_triple = next(
            (t for t in triples if ref_matches_triple(r, t)),
            None,
        )
        if match_triple is not None:
            matched_pairs.append((r, match_triple))
        else:
            unmatched.append(r)
    return len(matched_pairs), matched_pairs, unmatched


def load_json_or_die(path: Path):
    if not path.exists():
        sys.exit(f"ERROR: file not found: {path}")
    with path.open() as f:
        return json.load(f)


def extract_triples(kg_obj) -> list[dict]:
    if isinstance(kg_obj, dict):
        return kg_obj.get("triples", []) or kg_obj.get("edges", [])
    return list(kg_obj)


def extract_ref_edges(ref_obj) -> list[dict]:
    if isinstance(ref_obj, dict):
        return ref_obj.get("edges", []) or ref_obj.get("reference_edges", [])
    return list(ref_obj)


def fmt_edge(e: dict) -> str:
    s = e.get("subject", "")
    r = normalize_relation(e.get("relation", ""))
    o = e.get("object", "")
    return f"{s:<30s} --[{r}]--> {o}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg",  default="output/run11_kg/tiered_kg_run11.json")
    ap.add_argument("--ref", default="configs/lb_reference_edges.json")
    ap.add_argument("--expected-recall", type=int, default=26)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    kg_path = Path(args.kg)
    ref_path = Path(args.ref)

    kg_obj = load_json_or_die(kg_path)
    ref_obj = load_json_or_die(ref_path)

    triples = extract_triples(kg_obj)
    ref_edges = extract_ref_edges(ref_obj)

    tier1 = [t for t in triples if t.get("tier") == 1]
    tier12 = triples
    n_ref = len(ref_edges)

    t1_hits,  t1_pairs,  t1_unmatched  = count_hits(ref_edges, tier1)
    t12_hits, t12_pairs, t12_unmatched = count_hits(ref_edges, tier12)

    print("=" * 70)
    print(" OntoGeoRAG run11 recall verification")
    print("=" * 70)
    print(f"  KG file        : {kg_path}")
    print(f"  Benchmark file : {ref_path}")
    print(f"  Total triples  : {len(triples)}  (Tier 1 = {len(tier1)}, Tier 2 = {len(triples) - len(tier1)})")
    print(f"  Reference edges: {n_ref}")
    print()
    print("  Tier-1 recall:")
    print(f"    vs {n_ref}-edge benchmark: {t1_hits}/{n_ref} = {t1_hits/n_ref*100:.1f}%")
    print(f"    vs 26-edge denominator  : {t1_hits}/26 = {t1_hits/26*100:.1f}%")
    print()
    print("  Tier-1+2 recall (HEADLINE):")
    print(f"    vs {n_ref}-edge benchmark: {t12_hits}/{n_ref} = {t12_hits/n_ref*100:.1f}%   <-- paper / README")
    print(f"    vs 26-edge denominator  : {t12_hits}/26 = {t12_hits/26*100:.1f}%")
    print()

    if args.verbose:
        print("-" * 70)
        print(f"  MATCHED reference edges (Tier 1+2): {len(t12_pairs)}")
        print("-" * 70)
        for ref_e, trip in t12_pairs:
            print(f"  OK  {fmt_edge(ref_e)}")
            print(f"        matched by: {fmt_edge(trip)}")
        print()
        print("-" * 70)
        print(f"  UNMATCHED reference edges (Tier 1+2): {len(t12_unmatched)}")
        print("-" * 70)
        for ref_e in t12_unmatched:
            print(f"  MISS  {fmt_edge(ref_e)}")
        print()

    expected = args.expected_recall
    if t12_hits == expected:
        print(f"  ASSERTION PASSED: Tier-1+2 hits = {t12_hits} (expected {expected})")
        if n_ref == 34:
            print(f"  -> headline recall = {t12_hits}/34 = {t12_hits/34*100:.1f}% matches paper (76.5%)")
        return 0
    else:
        print(f"  ASSERTION FAILED: Tier-1+2 hits = {t12_hits}, expected {expected}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
