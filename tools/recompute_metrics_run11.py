#!/usr/bin/env python3
"""
recompute_metrics_run11.py
==========================
Recompute run11 recall under three matchers, side by side:
  1. substring_normRel  -- SLURM-inline / paper headline matcher
  2. exact_normRel      -- exact entity equality + relation map (intermediate)
  3. exact_strict       -- exact equality, no relation normalization
                           (reproduces 07_final_metrics.py recall() logic)

Reads : output/run11_kg/tiered_kg_run11.json
        configs/lb_reference_edges.json    (34-edge benchmark)
Writes: output/run11_kg/metrics_run11_unified.json   (NEW file, additive)

Does NOT modify metrics_run11.json or any pipeline code.
"""
import argparse
import json
import re
import sys
from pathlib import Path

RELATION_MAP = {
    "hasdescriptor": "hasDescriptor", "occursin": "occursIn",
    "partof": "partOf", "formedby": "formedBy", "causedby": "causedBy",
    "triggers": "triggers", "causes": "causes", "controls": "controls",
    "overlies": "overlies", "underlies": "underlies",
    "associatedwith": "associatedWith",
}

def normalize_relation(rel):
    if not rel: return ""
    key = rel.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    return RELATION_MAP.get(key, rel.strip())

def norm_text(s):
    return re.sub(r"\s+", " ", (s or "").lower().strip()).rstrip(".,;:")

def match_substring(ref, triple):
    """Headline matcher: bidirectional substring + relation normalization."""
    rs, ro = norm_text(ref.get("subject","")), norm_text(ref.get("object",""))
    rr = normalize_relation(ref.get("relation",""))
    ts, to = norm_text(triple.get("subject","")), norm_text(triple.get("object",""))
    tr = normalize_relation(triple.get("relation",""))
    return ((ts == rs or ts in rs or rs in ts) and
            (to == ro or to in ro or ro in to) and
            tr == rr)

def match_exact_normRel(ref, triple):
    """Exact entity equality + normalized relation."""
    return (norm_text(ref.get("subject","")) == norm_text(triple.get("subject","")) and
            norm_text(ref.get("object","")) == norm_text(triple.get("object","")) and
            normalize_relation(ref.get("relation","")) == normalize_relation(triple.get("relation","")))

def match_exact_strict(ref, triple):
    """Pure exact equality, no relation map (mimics 07_final_metrics.py)."""
    return (norm_text(ref.get("subject","")) == norm_text(triple.get("subject","")) and
            norm_text(ref.get("object","")) == norm_text(triple.get("object","")) and
            norm_text(ref.get("relation","")) == norm_text(triple.get("relation","")))

def count_hits(ref_edges, triples, matcher):
    triples = list(triples)
    hits, miss = [], []
    for r in ref_edges:
        (hits if any(matcher(r, t) for t in triples) else miss).append(r)
    return hits, miss

def fmt_edge(e):
    return f"{e.get('subject',''):<28s} --[{normalize_relation(e.get('relation',''))}]--> {e.get('object','')}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg",  default="output/run11_kg/tiered_kg_run11.json")
    ap.add_argument("--ref", default="configs/lb_reference_edges.json")
    ap.add_argument("--out", default="output/run11_kg/metrics_run11_unified.json")
    ap.add_argument("--show-diffs", action="store_true",
                    help="Print which edges are gained by substring vs exact-normRel")
    args = ap.parse_args()

    kg_obj  = json.load(open(args.kg))
    ref_obj = json.load(open(args.ref))
    triples = kg_obj.get("triples", kg_obj) if isinstance(kg_obj, dict) else list(kg_obj)
    ref     = ref_obj.get("edges", ref_obj) if isinstance(ref_obj, dict) else list(ref_obj)

    tier1  = [t for t in triples if t.get("tier") == 1]
    tier12 = triples
    n = len(ref)

    matchers = [
        ("substring_normRel", match_substring,       "<-- HEADLINE (paper / SLURM)"),
        ("exact_normRel",     match_exact_normRel,   "    (entity-exact, rel-normalized)"),
        ("exact_strict",      match_exact_strict,    "    (mimics 07_final_metrics.py)"),
    ]

    print("=" * 78)
    print(" Unified recall recomputation for run11")
    print("=" * 78)
    print(f"  KG       : {args.kg}  ({len(triples)} triples, T1={len(tier1)})")
    print(f"  Benchmark: {args.ref}  ({n} edges)")
    print()
    print(f"  {'Matcher':<22s} {'Tier-1':>16s} {'Tier-1+2':>16s}")
    print(f"  {'-'*22} {'-'*16} {'-'*16}")
    results = {}
    for label, matcher, note in matchers:
        t1h, _   = count_hits(ref, tier1,  matcher)
        t12h, _  = count_hits(ref, tier12, matcher)
        results[label] = {
            "tier1_hits":  len(t1h),  "tier1_recall":  len(t1h)/n,
            "tier12_hits": len(t12h), "tier12_recall": len(t12h)/n,
            "denominator": n,
            "matched_tier12": [f"{e.get('subject')}|{normalize_relation(e.get('relation',''))}|{e.get('object')}"
                               for e in t12h],
        }
        t1s  = f"{len(t1h):>3d}/{n} ({100*len(t1h)/n:>5.1f}%)"
        t12s = f"{len(t12h):>3d}/{n} ({100*len(t12h)/n:>5.1f}%)"
        print(f"  {label:<22s} {t1s:>16s} {t12s:>16s}  {note}")
    print()

    # Compare to existing metrics_run11.json (read-only)
    existing = Path("output/run11_kg/metrics_run11.json")
    if existing.exists():
        try:
            old = json.load(open(existing))
            # try common shapes
            r = old.get("recall")
            if isinstance(r, dict):
                hits = r.get("hits"); tot = r.get("total_reference"); val = r.get("recall")
                print(f"  Existing metrics_run11.json -> recall.hits = {hits} / {tot} "
                      f"= {100*val:.1f}%" if val is not None else f"  Existing metrics_run11.json structure unrecognized")
            else:
                print(f"  Existing metrics_run11.json top-level keys: {list(old.keys())}")
        except Exception as e:
            print(f"  Could not parse metrics_run11.json: {e}")
        print()

    # Show which edges substring picks up that exact_normRel misses
    if args.show_diffs:
        sub = set(results["substring_normRel"]["matched_tier12"])
        ex  = set(results["exact_normRel"]["matched_tier12"])
        gained = sub - ex
        print("-" * 78)
        print(f"  Edges gained by substring vs exact_normRel ({len(gained)}):")
        for g in sorted(gained):
            s, r, o = g.split("|")
            print(f"    + {s:<28s} --[{r}]--> {o}")
        print()

    # Write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kg":           args.kg,
        "benchmark":    args.ref,
        "n_reference":  n,
        "n_triples":    len(triples),
        "n_tier1":      len(tier1),
        "n_tier2":      len(triples) - len(tier1),
        "results":      results,
        "headline":     {"matcher": "substring_normRel", **{k: v for k, v in results["substring_normRel"].items() if k != "matched_tier12"}},
        "lower_bound":  {"matcher": "exact_normRel",     **{k: v for k, v in results["exact_normRel"].items()     if k != "matched_tier12"}},
        "strict_floor": {"matcher": "exact_strict",      **{k: v for k, v in results["exact_strict"].items()      if k != "matched_tier12"}},
        "notes": (
            "headline = matcher used in slurm/run11_gpu.sh inline block (paper Table 6). "
            "lower_bound = same matcher, exact entity equality (no substring). "
            "strict_floor = pure exact match without relation normalization "
            "(reproduces pipeline/07_final_metrics.py recall() behaviour modulo benchmark file)."
        ),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Wrote: {out_path}")

if __name__ == "__main__":
    sys.exit(main())
