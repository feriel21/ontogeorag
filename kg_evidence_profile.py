#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kg_evidence_profile.py

From the Part I OntoGeoRAG graph (tiered_kg_run11.json, 153 triples), build the
KG evidence profile for the Part II MTD-candidate interpretation:

  - KG-specificity of each OBSERVED descriptor  = 1 / (number of distinct canonical
    objects in the KG that carry this descriptor via the same relation).
    High specificity (few bearers) = high discriminating power. MEASURED, not a weight.
  - contrast-set = the other geological objects in the KG that also carry the
    observed descriptors (the alternatives the descriptor does NOT rule out).
  - not-observed KG-expected = descriptors/relations the KG attaches to MTD that are
    NOT in the observed set (these BOUND the claim; absence is information).

Hard design rules (do not relax):
  - The script reads ALL triples to build the profile, but only the OBSERVED set
    (edit below — these are YOUR signal-side observations) is attached to the body.
    A KG triple (MTD, hasDescriptor, X) means "an MTD may show X", NOT "this body shows X".
  - No aggregate P(MTD), no evidence summation. Power comes from graph STRUCTURE
    (specificity, contrast, context, absence), not from a scalar.
  - Variant normalization: generic string hygiene (case, whitespace, hyphen->space)
    + optional lexicon.json (your Part I canonicalization). No invented domain map.

Stdlib only. CPU only. Python 3.8+.

Usage
-----
python3 kg_evidence_profile.py \
    --kg /home/talbi/ontogeorag/output/run11_kg/tiered_kg_run11.json \
    --out evidence_profile_MTD.json
    [--lexicon /path/to/lexicon.json]
"""

import argparse
import collections
import json
import sys
from pathlib import Path

# --- CONFIG: YOUR observations (signal-derived). Edit to match what you SEE. ----
MTD_CANON = "mass transport deposit"
MTD_ALIASES = {"mtd", "mass transport deposits", "mass-transport deposit",
               "mass-transport deposits", "mass transport deposit"}

OBSERVED = [
    {"relation": "hasDescriptor", "object": "transparent"},
    {"relation": "hasDescriptor", "object": "chaotic"},
    {"relation": "hasDescriptor", "object": "hummocky"},
    {"relation": "partOf",        "object": "basal shear surface"},
]

# Relations treated as "descriptor-like" vs "structural/contextual" for the report.
DESCRIPTOR_RELATIONS = {"hasdescriptor"}
CONTEXT_RELATIONS = {"occursin"}
# -------------------------------------------------------------------------------


def pre(s):
    """generic normalization: lower, unicode/ascii hyphen -> space, collapse ws."""
    x = str(s).lower().strip()
    for h in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "-"):
        x = x.replace(h, " ")
    return " ".join(x.split())


def load_alias_map(lexicon_path):
    """optional: build alias->canonical from Part I lexicon.json (same idea as schema.py)."""
    amap = {}
    if not lexicon_path:
        return amap
    p = Path(lexicon_path)
    if not p.exists():
        print(f"[note] lexicon not found ({lexicon_path}); using generic normalization only.")
        return amap
    data = json.loads(p.read_text(encoding="utf-8"))
    for entry in data:
        canon = pre(entry.get("concept", ""))
        if not canon:
            continue
        amap[canon] = canon
        for al in entry.get("aliases", []) or []:
            amap[pre(al)] = canon
    print(f"[info] lexicon loaded: {len(amap)} alias->canonical mappings.")
    return amap


def make_canon(amap):
    mtd_alias_pre = {pre(a) for a in MTD_ALIASES}
    mtd_canon_pre = pre(MTD_CANON)

    def canon(term):
        t = pre(term)
        t = amap.get(t, t)
        if t in mtd_alias_pre:
            t = mtd_canon_pre
        return t
    return canon, mtd_canon_pre


def load_triples(kg_path):
    raw = json.loads(Path(kg_path).read_text(encoding="utf-8"))
    d = dict(raw) if isinstance(raw, list) else raw
    if "triples" not in d:
        sys.exit(f"[FATAL] no 'triples' key. keys={list(d.keys())[:20]}")
    return d["triples"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--lexicon", default=None)
    ap.add_argument("--out", default="evidence_profile_MTD.json")
    args = ap.parse_args()

    triples = load_triples(args.kg)
    amap = load_alias_map(args.lexicon)
    canon, MTD = make_canon(amap)
    print(f"[info] {len(triples)} triples; MTD canonical = '{MTD}'\n")

    # --- indexes over the WHOLE graph ---
    # bearers[(relation_canon, object_canon)] = set of subject_canon
    bearers = collections.defaultdict(set)
    # surface forms merged, for transparency
    merged = collections.defaultdict(set)
    # MTD's own outgoing triples, keep the richest representative per (rel,obj)
    mtd_out = {}     # (rel_canon, obj_canon) -> triple
    for t in triples:
        rel = pre(t.get("relation"))
        subj_c = canon(t.get("subject"))
        obj_c = canon(t.get("object"))
        bearers[(rel, obj_c)].add(subj_c)
        merged[obj_c].add(str(t.get("subject")).strip())
        if subj_c == MTD:
            key = (rel, obj_c)
            # prefer a Tier-1 / STRONG triple as representative if duplicates exist
            cur = mtd_out.get(key)
            if cur is None or (t.get("tier", 9) < cur.get("tier", 9)):
                mtd_out[key] = t

    # --- observed descriptors: specificity + contrast ---
    observed_keys = {(pre(o["relation"]), canon(o["object"])) for o in OBSERVED}
    observed_records = []
    contrast_counter = collections.Counter()  # alt object -> #observed descriptors shared

    print("=== observed descriptors: KG-specificity & contrast ===")
    for o in OBSERVED:
        rel, obj_c = pre(o["relation"]), canon(o["object"])
        bset = bearers.get((rel, obj_c), set())
        rep = mtd_out.get((rel, obj_c))
        if not bset or MTD not in bset:
            print(f"  [NO KG SUPPORT] ({MTD}) {rel} -> {o['object']}  -> not attachable from KG")
            observed_records.append({
                "relation": o["relation"], "object": o["object"],
                "kg_support": False, "kg_specificity": None, "contrast_set": []
            })
            continue
        contrast = sorted(bset - {MTD})
        spec = round(1.0 / len(bset), 4)
        for alt in contrast:
            contrast_counter[alt] += 1
        disc = (len(bset) == 1)
        tag = "DISCRIMINATING" if disc else f"shared with {len(contrast)}"
        print(f"  ({MTD}) {rel} -> {o['object']:22s} | bearers={len(bset)} "
              f"spec={spec} | tier={rep.get('tier')} {rep.get('verdict')} | {tag}")
        if contrast:
            print(f"        contrast: {contrast}")
        observed_records.append({
            "relation": o["relation"], "object": o["object"],
            "kg_support": True,
            "tier": rep.get("tier"), "tier_label": rep.get("tier_label"),
            "verdict": rep.get("verdict"),
            "evidence": rep.get("evidence"),
            "bearer_count": len(bset),
            "kg_specificity": spec,
            "discriminating": disc,
            "contrast_set": contrast,
        })

    # --- global contrast set (who else looks like this, and how much) ---
    global_contrast = [{"object": alt, "shared_observed_descriptors": n}
                       for alt, n in contrast_counter.most_common()]
    discriminators = [r["object"] for r in observed_records if r.get("discriminating")]

    print("\n=== global contrast set (alternatives sharing observed descriptors) ===")
    for g in global_contrast:
        print(f"  {g['object']:24s} shares {g['shared_observed_descriptors']} observed descriptor(s)")
    print(f"\n=== discriminating features (specificity = 1, KG-unique to MTD) ===\n  {discriminators or 'none'}")

    # --- not-observed KG-expected (bounds the claim) ---
    not_observed = collections.defaultdict(list)
    context_rels = []
    for (rel, obj_c), t in sorted(mtd_out.items()):
        entry = {"object": t.get("object"), "tier": t.get("tier"),
                 "verdict": t.get("verdict"), "evidence": t.get("evidence")}
        if rel in CONTEXT_RELATIONS:
            context_rels.append({"relation": t.get("relation"), **entry})
        if (rel, obj_c) in observed_keys:
            continue
        not_observed[t.get("relation")].append(entry)

    print("\n=== MTD descriptors/relations in KG but NOT observed (claim bounds) ===")
    for rel, items in sorted(not_observed.items()):
        objs = [it["object"] for it in items]
        print(f"  {rel}: {objs}")
    print("\n=== KG context for MTD (occursIn) ===")
    for c in context_rels:
        print(f"  {c['relation']} -> {c['object']}  (tier {c['tier']})")

    # --- variant-merge transparency (which surface forms collapsed) ---
    merges = {k: sorted(v) for k, v in merged.items() if len(v) > 1}

    out = {
        "kg_source": Path(args.kg).name,
        "mtd_canonical": MTD,
        "n_triples": len(triples),
        "method_notes": [
            "kg_specificity = 1 / (distinct canonical objects bearing the descriptor via the same relation); measured from the KG, not a learned weight.",
            "A KG triple (MTD, rel, X) means an MTD MAY show X; attachment to a body requires SIGNAL OBSERVATION (OBSERVED config).",
            "No aggregate MTD score is produced by design; power is structural (specificity, contrast, context, absence).",
            "Variant normalization: generic (case/whitespace/hyphen) + optional lexicon.json.",
        ],
        "observed_descriptors": observed_records,
        "global_contrast_set": global_contrast,
        "discriminating_features": discriminators,
        "not_observed_kg_expected": dict(not_observed),
        "context_relations": context_rels,
        "variant_merges": merges,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[ok] wrote {args.out}")
    if merges:
        print(f"[note] merged surface-form variants: {merges}")


if __name__ == "__main__":
    main()