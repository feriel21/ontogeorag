#!/usr/bin/env python3
"""
m4_enrich_quotes.py — Join verifier evidence quotes back into the KG
=====================================================================

Context: the final tiered KG (run11) does not carry the evidence quotes.
The quotes exist upstream, in the per-pass verification audits
(run11_a/verification_audit.jsonl, run11_b/verification_audit.jsonl):
each record holds the Qwen verifier's quoted evidence sentence,
reasoning, and verdict for one raw triple.

This script joins those audits to the tiered KG by canonicalized triple
key (applying canonical_map_v5.json so pre-canonicalization audit names
match post-canonicalization KG names) and writes a NEW enriched KG —
the frozen run11 artefact is never modified.

Output triple gains:
  "evidence": {"quote": ..., "reasoning": ..., "verdict": ..., "pass": "A"|"B"}
(if both passes have an audit record, pass A wins; pass B kept under
 "evidence_pass_b")

Usage:
    python m4_enrich_quotes.py \
        --kg      ~/ontogeorag/output/run11_kg/tiered_kg_run11.json \
        --pass-a  ~/ontogeorag/output/run11_a \
        --pass-b  ~/ontogeorag/output/run11_b \
        --output  ~/ontogeorag/output/m4/tiered_kg_run11_enriched.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from m4_verify import load_triples, triple_fields


def load_canonical_map(run_dir: Path) -> dict:
    """Load `run_dir`/canonical_map_v5.json (lowercased key/value) if present; returns {} otherwise, no side effects."""
    p = run_dir / "canonical_map_v5.json"
    if p.exists():
        return {k.strip().lower(): v.strip().lower()
                for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
    return {}


def canon(name: str, cmap: dict) -> str:
    """Lowercase/strip `name` and apply `cmap`'s canonicalization if present; no side effects."""
    n = name.strip().lower()
    return cmap.get(n, n)


def load_audit(run_dir: Path, cmap: dict, tag: str) -> dict:
    """Map canonicalized (s, r, o) -> audit record."""
    out = {}
    p = run_dir / "verification_audit.jsonl"
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a = json.loads(line)
            key = (canon(a.get("subject", ""), cmap),
                   a.get("relation", "").strip(),
                   canon(a.get("object", ""), cmap))
            rec = {"quote": a.get("evidence", ""),
                   "reasoning": a.get("reasoning", ""),
                   "verdict": a.get("verdict", ""),
                   "pass": tag}
            # keep the strongest verdict if duplicated within a pass
            if key not in out or (rec["verdict"] == "STRONG_SUPPORT"
                                  and out[key]["verdict"] != "STRONG_SUPPORT"):
                out[key] = rec
    return out


def main():
    """CLI entry point: joins --pass-a/--pass-b verification audits onto --kg's triples by canonicalized key, and writes the enriched KG (with evidence quotes + join stats) to --output; the source KG is never modified."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--pass-a", required=True)
    ap.add_argument("--pass-b", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    kg_path = Path(args.kg).expanduser()
    dir_a = Path(args.pass_a).expanduser()
    dir_b = Path(args.pass_b).expanduser()

    cmap_a = load_canonical_map(dir_a)
    cmap_b = load_canonical_map(dir_b)
    audit_a = load_audit(dir_a, cmap_a, "A")
    audit_b = load_audit(dir_b, cmap_b, "B")
    print(f"Audit records: pass A = {len(audit_a)}, pass B = {len(audit_b)}")

    triples = load_triples(kg_path)
    stats = Counter()
    for t in triples:
        s, r, o = triple_fields(t)
        key = (s.strip().lower(), r.strip(), o.strip().lower())
        rec_a = audit_a.get(key)
        rec_b = audit_b.get(key)
        primary = rec_a or rec_b
        if primary:
            t["evidence"] = dict(primary)
            if rec_a and rec_b:
                t["evidence_pass_b"] = dict(rec_b)
                stats["both_passes"] += 1
            else:
                stats[f"pass_{primary['pass']}_only"] += 1
        else:
            stats["no_audit_match"] += 1

    # preserve tier structure in output
    tier1 = [t for t in triples if t.get("tier") == 1]
    tier2 = [t for t in triples if t.get("tier") != 1]
    out = {"meta": {"source_kg": str(kg_path),
                    "enrichment": "verifier evidence quotes from "
                                  "run11_a/run11_b verification audits",
                    "join_stats": dict(stats)},
           "tier1": tier1, "tier2": tier2}

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(json.dumps(dict(stats), indent=2))
    print(f"Enriched KG: {out_path}")
    if stats["no_audit_match"]:
        print(f"WARNING: {stats['no_audit_match']} triples without audit "
              f"match — inspect entity normalization if this is high.")


if __name__ == "__main__":
    main()