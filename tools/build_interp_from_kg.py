#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_interp_from_kg.py

Reads the Part I OntoGeoRAG knowledge graph (tiered_kg_run11.json) and emits the
verified KG-descriptor block used to ground the Part II MTD-candidate interpretation.

Design principles (do not change without reason):
  - tier and evidence are READ FROM THE KG, never hardcoded. If `chaotic` is Tier-2
    in run11, this script writes tier=2 — automatically and verifiably.
  - descriptors are copied VERBATIM from the KG (no paraphrase of evidence text).
  - the requested descriptor set is explicit and auditable (TARGET_DESCRIPTORS below);
    the script reports FOUND / MISSING / non-Tier-1 for every requested item.
  - geometry (per-body area, etc.) is the DATA side of the Fig.1 frontier and is kept
    in a separate `measured_attributes` block, never mixed into kg_descriptors.

Stdlib only. CPU only. Python 3.8+.

Usage
-----
# 1) Core: verify + extract the KG descriptor block (always runnable):
python3 build_interp_from_kg.py \
    --kg /home/talbi/ontogeorag/output/run11_kg/tiered_kg_run11.json \
    --out-descriptors kg_descriptors_run11.json

# 2) Optional: also assemble a per-line interp.json by merging body geometry.
#    --bodies-stats expects a JSON list like:
#       [{"id":1,"area_px":176300}, {"id":2,"area_px":146667}, {"id":3,"area_px":12398}]
python3 build_interp_from_kg.py \
    --kg /home/talbi/ontogeorag/output/run11_kg/tiered_kg_run11.json \
    --line Inline_2240 \
    --bodies-stats bodies_Inline_2240.json \
    --out-interp interp_Inline_2240.json
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — audit these. This is the descriptor set grounded in the Part II v3
# figure. Matching is EXACT on (subject, relation, object) after whitespace/case
# normalization, so no unrelated triple can be picked up by accident.
# ---------------------------------------------------------------------------
TARGET_DESCRIPTORS = [
    {"subject": "mass transport deposit", "relation": "hasDescriptor", "object": "transparent"},
    {"subject": "mass transport deposit", "relation": "hasDescriptor", "object": "chaotic"},
    {"subject": "mass transport deposit", "relation": "hasDescriptor", "object": "hummocky"},
    {"subject": "mass transport deposit", "relation": "partOf",        "object": "basal shear surface"},
]

# Analyst-level caveats (YOUR scientific statement, not a KG field). Keyed by object.
# Kept separate from KG provenance on purpose. Edit/extend as you see fit.
ANALYST_CAVEATS = {
    "chaotic": "non-diagnostic: implication is one-way (MTD -> chaotic), not (chaotic -> MTD); "
               "chaotic also describes megaslide and debris-flow deposits.",
}

# Fields copied verbatim from each KG triple into the interp record.
KEEP_FIELDS = ["subject", "relation", "object", "subject_type", "object_type",
               "tier", "tier_label", "verdict", "origin", "evidence",
               "query", "strategy", "support_count", "supporting_papers"]

SCOPE_BOUND = {"faults": "out of scope — no fault triple in KG (not the same as 'faults absent')"}


def norm(s):
    return " ".join(str(s).lower().split())


def load_kg_triples(kg_path):
    """tiered_kg_run11.json is serialized as a list of [key, value] pairs;
    dict() reconstructs the top-level mapping, then ['triples'] is the triple list."""
    raw = json.loads(Path(kg_path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        d = dict(raw)
    elif isinstance(raw, dict):
        d = raw
    else:
        sys.exit(f"[FATAL] Unexpected KG top-level type: {type(raw).__name__}")
    if "triples" not in d:
        sys.exit(f"[FATAL] No 'triples' key in KG. Top-level keys: {list(d.keys())[:20]}")
    return d["triples"]


def find_triple(triples, target):
    ts, tr, to = norm(target["subject"]), norm(target["relation"]), norm(target["object"])
    for t in triples:
        if norm(t.get("subject")) == ts and norm(t.get("relation")) == tr and norm(t.get("object")) == to:
            return t
    return None


def build_descriptor_record(t):
    rec = {k: t.get(k) for k in KEEP_FIELDS if k in t}
    obj = norm(t.get("object"))
    tier = t.get("tier")
    # honest, KG-derived confidence note
    if isinstance(tier, int) and tier >= 2:
        rec["confidence_note"] = (f"Tier-{tier}: supported in one extraction pass only "
                                  f"(less consistent than Tier-1).")
    # analyst-level caveat, clearly separated from KG provenance
    if obj in ANALYST_CAVEATS:
        rec["analyst_caveat"] = ANALYST_CAVEATS[obj]
    rec["kg_match"] = "exact"
    return rec


def main():
    ap = argparse.ArgumentParser(description="Extract verified MTD descriptors from the OntoGeoRAG KG.")
    ap.add_argument("--kg", required=True, help="Path to tiered_kg_run11.json")
    ap.add_argument("--out-descriptors", default="kg_descriptors_run11.json",
                    help="Output path for the shared KG-descriptor block")
    ap.add_argument("--line", default=None, help="Line name (e.g. Inline_2240) for the interp.json")
    ap.add_argument("--bodies-stats", default=None,
                    help="Optional JSON list of bodies with at least {'id','area_px'}")
    ap.add_argument("--out-interp", default=None, help="Output path for the per-line interp.json")
    args = ap.parse_args()

    triples = load_kg_triples(args.kg)
    print(f"[info] KG loaded: {len(triples)} triples from {args.kg}\n")

    kg_descriptors = []
    integrity_flags = []
    print("=== descriptor verification report ===")
    for tgt in TARGET_DESCRIPTORS:
        label = f'({tgt["subject"]}) {tgt["relation"]} -> {tgt["object"]}'
        t = find_triple(triples, tgt)
        if t is None:
            print(f"  [MISSING] {label}")
            integrity_flags.append(f"MISSING: {label}")
            continue
        tier = t.get("tier")
        tag = "OK(T1)" if tier == 1 else f"WARNING(T{tier})"
        print(f"  [{tag}] {label}   verdict={t.get('verdict')}  papers={t.get('supporting_papers')}")
        if tier != 1:
            integrity_flags.append(f"NON-TIER-1 (tier={tier}): {label}")
        kg_descriptors.append(build_descriptor_record(t))

    print("\n=== integrity summary ===")
    if integrity_flags:
        for f in integrity_flags:
            print(f"  ! {f}")
        print("  -> Reflect these tiers/labels on the figure and in the manuscript wording.")
    else:
        print("  All requested descriptors found at Tier-1.")
    # provenance reality check
    if all(not d.get("supporting_papers") for d in kg_descriptors):
        print("  ! supporting_papers empty on all descriptors -> available provenance is "
              "PASSAGE-LEVEL (evidence quote), not paper-ID. Adjust legend wording accordingly.")

    Path(args.out_descriptors).write_text(
        json.dumps({"kg_source": Path(args.kg).name, "descriptors": kg_descriptors},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[ok] wrote {args.out_descriptors}  ({len(kg_descriptors)} descriptors)")

    # ---- optional: assemble per-line interp.json ----
    if args.out_interp:
        if not args.line:
            sys.exit("[FATAL] --out-interp requires --line")
        bodies = []
        if args.bodies_stats:
            bs = json.loads(Path(args.bodies_stats).read_text(encoding="utf-8"))
            if isinstance(bs, dict):
                bs = bs.get("bodies", [])
            for b in bs:
                bodies.append({
                    "id": b.get("id"),
                    "area_px": b.get("area_px", b.get("area")),
                    "measured_attributes": {}   # DATA side — filled later (morpho/topo)
                })
        else:
            print("[note] no --bodies-stats given: 'bodies' left empty (fill from mask pipeline).")

        interp = {
            "line": args.line,
            "kg_source": Path(args.kg).name,
            "generated_from": "build_interp_from_kg.py",
            "kg_descriptors": kg_descriptors,   # KNOWLEDGE side — shared, literature-grounded
            "bodies": bodies,                   # DATA side — geometry + measured_attributes
            "scope_bound": SCOPE_BOUND,
        }
        Path(args.out_interp).write_text(
            json.dumps(interp, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[ok] wrote {args.out_interp}  ({len(bodies)} bodies)")


if __name__ == "__main__":
    main()