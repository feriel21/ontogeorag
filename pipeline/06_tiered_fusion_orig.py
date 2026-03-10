#!/usr/bin/env python3
"""
Step 1: Tiered KG Fusion
Gere les deux formats:
  - Iter4: _verdict (racine) + _provenance.best_chunk_text
  - Iter7: _verification.verdict (imbrique) + _verification.evidence

Usage:
    python step1_tiered_fusion.py \
        --iter4 output/step4/verified_triples_v4.jsonl \
        --iter7 output/step7/verified_triples_v7.jsonl \
        --output output/improved_kg/tiered_kg.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def get_verdict(t):
    if "_verification" in t and isinstance(t["_verification"], dict):
        return t["_verification"].get("verdict", "")
    return t.get("_verdict", "")


def get_evidence(t):
    if "_verification" in t and isinstance(t["_verification"], dict):
        ev = t["_verification"].get("evidence", "")
        if ev:
            return ev[:400]
    prov = t.get("_provenance", {})
    return prov.get("best_chunk_text", "")[:300]


def get_provenance(t):
    prov = t.get("_provenance", {})
    verif = t.get("_verification", {}) if isinstance(t.get("_verification"), dict) else {}
    return {
        "query":         prov.get("query", ""),
        "strategy":      prov.get("strategy", ""),
        "best_chunk_id": prov.get("best_chunk_id", ""),
        "model":         verif.get("model", ""),
        "reasoning":     (verif.get("reasoning") or "")[:200],
    }


def verdict_to_tier(verdict):
    v = str(verdict).upper().strip()
    if "STRONG" in v:
        return 1
    if "WEAK" in v or "UNCERTAIN" in v:
        return 2
    return 3


def load_jsonl(path):
    triples = []
    errors = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                triples.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors += 1
                if errors <= 3:
                    print("   Warning ligne {}: {}".format(i+1, e))
    if errors:
        print("   Total lignes ignorees: {}".format(errors))
    return triples


def normalize(text):
    if not text:
        return ""
    return " ".join(str(text).lower().strip().split())


def triple_key(t):
    return (
        normalize(t.get("source", t.get("subject", ""))),
        normalize(t.get("relation", "")),
        normalize(t.get("target", t.get("object", ""))),
    )


def to_standard(t, tier, in_iter7, origin):
    prov = get_provenance(t)
    tier_labels = {1: "verified", 2: "literature-implied", 3: "parametric-flagged"}
    return {
        "subject":       normalize(t.get("source", t.get("subject", ""))),
        "relation":      normalize(t.get("relation", "")),
        "object":        normalize(t.get("target", t.get("object", ""))),
        "subject_type":  t.get("source_type", ""),
        "object_type":   t.get("target_type", ""),
        "tier":          tier,
        "tier_label":    tier_labels[tier],
        "verdict":       get_verdict(t),
        "in_iter7":      in_iter7,
        "origin":        origin,
        "evidence":      get_evidence(t),
        "query":         prov["query"],
        "strategy":      prov["strategy"],
        "best_chunk_id": prov["best_chunk_id"],
        "reasoning":     prov["reasoning"],
    }


def build_index(triples):
    index = {}
    for t in triples:
        k = triple_key(t)
        tier = verdict_to_tier(get_verdict(t))
        if k not in index or tier < index[k]["tier"]:
            index[k] = {"tier": tier, "raw": t}
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter4", required=True)
    parser.add_argument("--iter7", required=True)
    parser.add_argument("--output", default="output/improved_kg/tiered_kg.json")
    parser.add_argument("--include_tier3", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("TIERED KG FUSION")
    print("=" * 60)

    print("\nIter4: {}".format(args.iter4))
    iter4_raw = load_jsonl(args.iter4)
    print("  -> {} triples bruts".format(len(iter4_raw)))

    print("Iter7: {}".format(args.iter7))
    iter7_raw = load_jsonl(args.iter7)
    print("  -> {} triples bruts".format(len(iter7_raw)))

    # Detecter formats
    if iter4_raw:
        v4_key = "_verification.verdict" if "_verification" in iter4_raw[0] else "_verdict"
        print("\nFormat Iter4: verdict via '{}'  exemple='{}'".format(v4_key, get_verdict(iter4_raw[0])))
    if iter7_raw:
        v7_key = "_verification.verdict" if "_verification" in iter7_raw[0] else "_verdict"
        print("Format Iter7: verdict via '{}'  exemple='{}'".format(v7_key, get_verdict(iter7_raw[0])))

    iter4_index = build_index(iter4_raw)
    iter7_index = build_index(iter7_raw)
    iter7_keys = set(iter7_index.keys())

    tiered = []

    # Triples Iter4
    for k, entry in iter4_index.items():
        in_iter7 = k in iter7_keys
        tier = entry["tier"]
        if in_iter7:
            tier = min(tier, iter7_index[k]["tier"])
            origin = "both"
        else:
            origin = "iter4"
        if tier == 3 and not args.include_tier3:
            continue
        tiered.append(to_standard(entry["raw"], tier, in_iter7, origin))

    # Triples Iter7 absents de Iter4
    for k, entry in iter7_index.items():
        if k not in iter4_index:
            tier = entry["tier"]
            if tier == 3 and not args.include_tier3:
                continue
            tiered.append(to_standard(entry["raw"], tier, True, "iter7_only"))

    tiered.sort(key=lambda x: (x["tier"], x["relation"], x["subject"]))

    tier_counts   = defaultdict(int)
    rel_counts    = defaultdict(int)
    origin_counts = defaultdict(int)
    v4_verdicts   = defaultdict(int)
    v7_verdicts   = defaultdict(int)

    for t in tiered:
        tier_counts[t["tier"]] += 1
        rel_counts[t["relation"]] += 1
        origin_counts[t["origin"]] += 1
    for t in iter4_raw:
        v4_verdicts[get_verdict(t)] += 1
    for t in iter7_raw:
        v7_verdicts[get_verdict(t)] += 1

    print("\n" + "=" * 60)
    print("RESULTATS")
    print("=" * 60)
    print("\nKG final: {} triples".format(len(tiered)))
    print("\nPar Tier:")
    print("  Tier 1 - verified:            {:4d}  (STRONG_SUPPORT)".format(tier_counts[1]))
    print("  Tier 2 - literature-implied:  {:4d}  (WEAK / UNCERTAIN)".format(tier_counts[2]))
    print("  Tier 3 - parametric:          {:4d}  (NOT_SUPPORTED)".format(tier_counts[3]))
    print("\nOrigine:")
    print("  Iter4 & Iter7 (both):    {:4d}".format(origin_counts["both"]))
    print("  Iter4 seulement:         {:4d}".format(origin_counts["iter4"]))
    print("  Iter7 seulement:         {:4d}".format(origin_counts["iter7_only"]))
    print("\nRelations:")
    for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print("  {:<22} {:4d}".format(rel, count))
    print("\nVerdicts Iter4 ({} total):".format(len(iter4_raw)))
    for v, c in sorted(v4_verdicts.items(), key=lambda x: -x[1]):
        print("  {:<25} {:4d}".format(str(v), c))
    print("\nVerdicts Iter7 ({} total):".format(len(iter7_raw)))
    for v, c in sorted(v7_verdicts.items(), key=lambda x: -x[1]):
        print("  {:<25} {:4d}".format(str(v), c))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out = {
        "metadata": {
            "iter4_file":            str(args.iter4),
            "iter7_file":            str(args.iter7),
            "iter4_raw_count":       len(iter4_raw),
            "iter7_raw_count":       len(iter7_raw),
            "total_triples":         len(tiered),
            "tier_distribution":     dict(tier_counts),
            "relation_distribution": dict(rel_counts),
            "origin_distribution":   dict(origin_counts),
            "include_tier3":         args.include_tier3,
        },
        "triples": tiered,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\nSauvegarde: {}".format(args.output))
    print("\nSuite:")
    print("  python step2_entity_normalization.py \\")
    print("    --input {} \\".format(args.output))
    print("    --output output/improved_kg/tiered_kg_normalized.json")


if __name__ == "__main__":
    main()