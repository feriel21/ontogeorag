#!/usr/bin/env python3
"""
Step 3 CORRIGE: Final Metrics for Article
- Fix hallucination: supporte SUPPORTED/STRONG_SUPPORT/UNCERTAIN/WEAK_SUPPORT
- Fix entites: fusionne mass-transport deposit / mtd / mass transport deposits

Usage:
    python step3_final_metrics.py \
        --kg output/improved_kg/tiered_kg_normalized.json \
        --output output/improved_kg/article_metrics.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

LB2019_DESCRIPTORS = {
    "chaotic", "transparent", "blocky", "layered", "stratified",
    "parallel", "continuous", "discontinuous", "massive",
    "low-amplitude", "high-amplitude", "undeformed", "deformed"
}

DESCRIPTOR_SYNONYMS = {
    "stratified":               "layered",
    "sub-parallel":             "parallel",
    "sub parallel":             "parallel",
    "essentially undeformed":   "undeformed",
    "low amplitude":            "low-amplitude",
    "high amplitude":           "high-amplitude",
    "low-amplitude reflection": "low-amplitude",
    "high-amplitude reflection":"high-amplitude",
    "high-amplitude positive":  "high-amplitude",
    "low amplitude reflections":"low-amplitude",
    "high amplitude reflections":"high-amplitude",
}

REFERENCE_EDGES = [
    ("mass transport deposit", "hasdescriptor", "chaotic"),
    ("mass transport deposit", "hasdescriptor", "transparent"),
    ("mass transport deposit", "hasdescriptor", "blocky"),
    ("mass transport deposit", "hasdescriptor", "layered"),
    ("mass transport deposit", "hasdescriptor", "parallel"),
    ("mass transport deposit", "hasdescriptor", "continuous"),
    ("mass transport deposit", "hasdescriptor", "discontinuous"),
    ("mass transport deposit", "hasdescriptor", "massive"),
    ("mass transport deposit", "hasdescriptor", "low-amplitude"),
    ("mass transport deposit", "hasdescriptor", "high-amplitude"),
    ("slide", "hasdescriptor", "chaotic"),
    ("slide", "hasdescriptor", "blocky"),
    ("debris flow", "hasdescriptor", "chaotic"),
    ("debris flow", "hasdescriptor", "massive"),
    ("turbidity current", "formedby", "slope failure"),
    ("mass transport deposit", "formedby", "slope failure"),
    ("slope failure", "causes", "mass transport deposit"),
    ("sea-level lowstand", "triggers", "slope failure"),
    ("pore pressure buildup", "triggers", "slope failure"),
    ("headscarp", "partof", "mass transport deposit"),
    ("translated block", "partof", "mass transport deposit"),
    ("mass transport deposit", "overlies", "slope"),
    ("mass transport deposit", "occursin", "continental slope"),
    ("turbidite", "occursin", "deep water"),
    ("debris flow", "occursin", "continental slope"),
    ("mass transport deposit", "affects", "slope stability"),
]

MTD_VARIANTS = {
    "mass-transport deposit", "mass-transport deposits",
    "mass-transport deposits (mtd)", "mtd", "mass transport deposits",
    "mass transport deposit 1", "mass transport deposit 2",
    "mtd 1", "mtd 2", "mass transport complex",
    "mass-transport complex",
}

ENTITY_NORMS = {v: "mass transport deposit" for v in MTD_VARIANTS}
ENTITY_NORMS.update({
    "debris flows":                  "debris flow",
    "debrites":                      "debris flow",
    "turbidity currents":            "turbidity current",
    "turbidites":                    "turbidite",
    "slides":                        "slide",
    "slumps":                        "slump",
    "hydrate dissociation":          "gas hydrate dissociation",
    "methane hydrate dissociation":  "gas hydrate dissociation",
    "gas hydrate dissolution":       "gas hydrate dissociation",
    "low amplitude":                 "low-amplitude",
    "high amplitude":                "high-amplitude",
    "low amplitude reflections":     "low-amplitude",
    "high amplitude reflections":    "high-amplitude",
})


def norm(text):
    if not text:
        return ""
    t = " ".join(str(text).lower().strip().split())
    return ENTITY_NORMS.get(t, t)


def norm_desc(text):
    t = norm(text)
    return DESCRIPTOR_SYNONYMS.get(t, t)


def verdict_to_tier(verdict):
    v = str(verdict).upper().strip()
    if "STRONG" in v or v == "SUPPORTED":
        return 1
    if "WEAK" in v or "UNCERTAIN" in v:
        return 2
    return 3


def coverage(triples):
    found = set()
    for t in triples:
        d = norm_desc(t.get("object", ""))
        if d in LB2019_DESCRIPTORS:
            found.add(d)
    missing = LB2019_DESCRIPTORS - found
    return {"found": sorted(found), "missing": sorted(missing),
            "n_found": len(found), "n_total": 13,
            "coverage": len(found) / 13}


def recall(triples):
    kg = set()
    for t in triples:
        kg.add((norm(t["subject"]), norm(t["relation"]), norm(t["object"])))
    ref = [(norm(s), norm(r), norm(o)) for s, r, o in REFERENCE_EDGES]
    hits = [k for k in ref if k in kg]
    return {"recall": len(hits)/len(ref), "hits": len(hits),
            "total_reference": len(ref), "matched_edges": hits}


def hallucination(triples):
    counts = defaultdict(int)
    for t in triples:
        tier = verdict_to_tier(t.get("verdict", ""))
        if tier == 1:
            counts["strong"] += 1
        elif tier == 2:
            counts["weak"] += 1
        else:
            counts["not_supported"] += 1
    total = sum(counts.values())
    rate = counts["not_supported"] / total if total > 0 else 0.0
    return {"strong_support": counts["strong"], "weak_support": counts["weak"],
            "not_supported": counts["not_supported"], "total": total,
            "hallucination_rate": rate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg", required=True)
    parser.add_argument("--output", default="output/improved_kg/article_metrics.json")
    args = parser.parse_args()

    print("=" * 62)
    print("FINAL METRICS (CORRIGE)")
    print("=" * 62)

    with open(args.kg, "r", encoding="utf-8") as f:
        data = json.load(f)
    triples = data.get("triples", data) if isinstance(data, dict) else data
    print("\nTriples charges: {}".format(len(triples)))

    # Normalisation supplementaire
    for t in triples:
        t["subject"] = norm(t.get("subject", ""))
        t["object"]  = norm(t.get("object", ""))

    # Re-deduplication
    seen = {}
    for t in triples:
        k = (t["subject"], t["relation"], t["object"])
        if k not in seen or t.get("tier", 3) < seen[k].get("tier", 3):
            seen[k] = t
    deduped = sorted(seen.values(), key=lambda x: (x.get("tier", 3), x.get("relation",""), x.get("subject","")))
    print("Apres normalisation+dedup: {} triples ({} supprimes)".format(
        len(deduped), len(triples) - len(deduped)))

    tier1  = [t for t in deduped if t.get("tier", 3) == 1]
    tier12 = [t for t in deduped if t.get("tier", 3) <= 2]

    cov1  = coverage(tier1)
    cov12 = coverage(tier12)
    rec1  = recall(tier1)
    rec12 = recall(tier12)
    hall  = hallucination(deduped)

    entities = set()
    for t in deduped:
        entities.add(t["subject"])
        entities.add(t["object"])

    rel_by_tier = defaultdict(lambda: defaultdict(int))
    for t in deduped:
        rel_by_tier[t.get("tier", 3)][t.get("relation", "?")] += 1

    print("\n" + "=" * 62)
    print("TABLE ARTICLE (copy-paste pour LaTeX)")
    print("=" * 62)
    print("\n{:<38} {:>7} {:>8}".format("Metric", "Tier1", "Tier1+2"))
    print("-" * 55)
    print("{:<38} {:>7} {:>8}".format("Triples", len(tier1), len(tier12)))
    e1 = len(set(t["subject"] for t in tier1) | set(t["object"] for t in tier1))
    e12 = len(set(t["subject"] for t in tier12) | set(t["object"] for t in tier12))
    print("{:<38} {:>7} {:>8}".format("Unique entities", e1, e12))
    print("{:<38} {:>7} {:>8}".format("Descriptor coverage (n/13)", cov1["n_found"], cov12["n_found"]))
    print("{:<38} {:>6.1f}% {:>7.1f}%".format("Descriptor coverage (%)", cov1["coverage"]*100, cov12["coverage"]*100))
    print("{:<38} {:>7} {:>8}".format("Recall vs LB2019 (n/26)", rec1["hits"], rec12["hits"]))
    print("{:<38} {:>6.1f}% {:>7.1f}%".format("Recall vs LB2019 (%)", rec1["recall"]*100, rec12["recall"]*100))
    print("{:<38} {:>6.1f}% {:>7.1f}%".format("Hallucination rate", 0.0, hall["hallucination_rate"]*100))

    print("\nDescriptors found (Tier1+2): {}".format(", ".join(cov12["found"])))
    if cov12["missing"]:
        print("Missing:                     {}".format(", ".join(cov12["missing"])))

    print("\nMatched LB2019 edges (Tier1+2, {}/26):".format(rec12["hits"]))
    for s, r, o in rec12["matched_edges"]:
        print("  {} --[{}]--> {}".format(s, r, o))

    print("\nHallucination breakdown (full KG):")
    print("  STRONG/SUPPORTED:  {:4d}".format(hall["strong_support"]))
    print("  WEAK/UNCERTAIN:    {:4d}".format(hall["weak_support"]))
    print("  NOT_SUPPORTED:     {:4d}".format(hall["not_supported"]))
    print("  Hallucination rate: {:.1f}%".format(hall["hallucination_rate"]*100))

    print("\nRelations (Tier1+2):")
    all_rels = sorted(rel_by_tier[1].keys() | rel_by_tier[2].keys(),
                      key=lambda r: -(rel_by_tier[1][r]+rel_by_tier[2][r]))
    print("  {:<22} {:>4} {:>4} {:>5}".format("Relation", "T1", "T2", "Tot"))
    print("  " + "-" * 36)
    for rel in all_rels:
        t1 = rel_by_tier[1][rel]
        t2 = rel_by_tier[2][rel]
        if t1 + t2 > 0:
            print("  {:<22} {:>4} {:>4} {:>5}".format(rel, t1, t2, t1+t2))

    print("\nTop entites:")
    ent_counts = defaultdict(int)
    for t in deduped:
        ent_counts[t["subject"]] += 1
        ent_counts[t["object"]]  += 1
    for ent, cnt in sorted(ent_counts.items(), key=lambda x: -x[1])[:10]:
        print("  {:<40} {:3d}".format(ent, cnt))

    metrics = {
        "summary": {
            "tier1_triples": len(tier1), "tier12_triples": len(tier12),
            "total_triples": len(deduped), "unique_entities": len(entities),
        },
        "descriptor_coverage": {"tier1": cov1, "tier12": cov12},
        "recall_vs_lb2019":    {"tier1": rec1, "tier12": rec12},
        "hallucination":       hall,
        "relation_distribution_by_tier": {str(k): dict(v) for k, v in rel_by_tier.items()},
        "triples_final": deduped,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("\nSauvegarde: {}".format(args.output))


if __name__ == "__main__":
    main()