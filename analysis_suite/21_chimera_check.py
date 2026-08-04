#!/usr/bin/env python3
"""
21_chimera_check.py — Are triples composed across passages?
===========================================================
WHY (attacks the "not measured" weakness)
    Extraction feeds the model FIVE concatenated passages at once. Nothing
    prevents it from combining an entity seen in passage 1 with an entity
    seen in passage 4 into a single triple that appears in neither — a
    "chimera". This is listed as an uncontrolled failure mode, and until now
    it was unmeasured, which is worse than being large: a reviewer can
    assume the worst.

    It is measurable from data already on disk. Each triple stores the ids
    of the passages it was retrieved with; the corpus stores the passages.
    A triple is *co-located* if some single retrieved passage contains both
    its subject and its object. If no single passage does, the triple was
    composed across passages and is a chimera candidate.

WHAT
    For every triple, classifies:
      CO_LOCATED     both entities occur in one and the same passage
      SPLIT          each entity occurs, but never in the same passage
                     -> chimera candidate
      PARTIAL        only one entity is found in the retrieved passages
      NOT_FOUND      neither entity found (usually canonicalized surface)
    Cross-tabulates against tier and panel decision: if SPLIT triples are
    accepted at the same rate as CO_LOCATED ones, cross-passage composition
    is not being caught by verification — which is the number that matters.

    Matching is on normalized surface forms and is therefore conservative:
    an entity renamed by canonicalization will not be found in the raw text,
    which inflates PARTIAL/NOT_FOUND rather than SPLIT. The chimera rate
    reported is thus a LOWER bound, and stated as such.

OUTPUTS (in --outdir)
    chimera_report.json     rates overall, by tier, by panel decision
    chimera_candidates.csv  the SPLIT triples, for inspection

USAGE
    python analysis_suite/21_chimera_check.py \
        --kg output/run13/analysis/kg_with_provenance.json \
        --chunks output/run13/step1/chunks.jsonl \
        --decisions output/run13/m4_panel/m4_panel_decisions.jsonl \
        --outdir output/run13/analysis
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kg_io import (get_chunk_ids, get_object, get_relation, get_subject,
                   load_chunk_records, load_kg)


def norm(s):
    s = str(s or "").lower()
    for bad, good in (("\ufb01", "fi"), ("\ufb02", "fl"), ("-", " ")):
        s = s.replace(bad, good)
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--decisions", default=None)
    ap.add_argument("--outdir", default="output/run13/analysis")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    kg = load_kg(args.kg)
    triples = [t for t in kg["triples"] if t.get("_status") != "quarantine"]
    chunks = {c["chunk_id"]: norm(c["text"])
              for c in load_chunk_records(args.chunks)}

    panel = {}
    if args.decisions:
        for line in open(args.decisions, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                panel[(norm(d.get("subject")), str(d.get("relation")).strip(),
                       norm(d.get("object")))] = str(
                    d.get("m4_decision") or "").upper()

    status = Counter()
    by_tier = defaultdict(Counter)
    by_decision = defaultdict(Counter)
    splits, no_prov = [], 0

    for t in triples:
        s, r, o = get_subject(t), get_relation(t), get_object(t)
        sn, on = norm(s), norm(o)
        ids = (t.get("retrieved_chunk_ids") or get_chunk_ids(t)
               or t.get("cooc_chunk_ids") or [])
        texts = [chunks[i] for i in ids if i in chunks]
        if not texts:
            no_prov += 1
            continue
        both = any(sn in txt and on in txt for txt in texts)
        has_s = any(sn in txt for txt in texts)
        has_o = any(on in txt for txt in texts)
        if both:
            st = "CO_LOCATED"
        elif has_s and has_o:
            st = "SPLIT"
        elif has_s or has_o:
            st = "PARTIAL"
        else:
            st = "NOT_FOUND"
        status[st] += 1
        tier = t.get("_tier", 2)
        by_tier[tier][st] += 1
        dec = panel.get((sn, r.strip(), on), "")
        if dec:
            by_decision[dec][st] += 1
        if st == "SPLIT":
            splits.append({
                "subject": s, "relation": r, "object": o, "tier": tier,
                "panel_decision": dec, "n_passages": len(texts),
                "confidence": t.get("confidence", ""),
                "evidence_paper": t.get("evidence_paper", "")})

    n = sum(status.values())
    rate = {k: round(v / max(n, 1), 4) for k, v in status.items()}

    # the number that matters: are SPLIT triples accepted as often?
    acc = {}
    for st in ("CO_LOCATED", "SPLIT"):
        tot = sum(by_decision[d][st] for d in by_decision)
        ok = by_decision.get("ACCEPT", Counter())[st]
        acc[st] = {"n_judged": tot,
                   "accept_rate": round(ok / tot, 4) if tot else None}

    report = {
        "n_triples_evaluated": n,
        "n_without_usable_provenance": no_prov,
        "status_counts": dict(status), "status_rates": rate,
        "by_tier": {str(k): dict(v) for k, v in by_tier.items()},
        "by_panel_decision": {k: dict(v) for k, v in by_decision.items()},
        "acceptance_by_colocation": acc,
        "caveat": ("Surface matching is conservative: entities renamed by "
                   "canonicalization are not found in the raw passages, "
                   "which inflates PARTIAL/NOT_FOUND rather than SPLIT. The "
                   "SPLIT rate is therefore a LOWER bound on cross-passage "
                   "composition."),
        "power_warning": ("The acceptance comparison between SPLIT and "
                          "CO_LOCATED is only interpretable with a few tens "
                          "of SPLIT triples; below that, report the SPLIT "
                          "rate alone."),
        "reading": ("SPLIT triples are candidates for cross-passage "
                    "composition: both entities were in the retrieved "
                    "context but never in one and the same passage. If "
                    "their acceptance rate matches that of CO_LOCATED "
                    "triples, verification does not distinguish them."),
    }
    with open(outdir / "chimera_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if splits:
        with open(outdir / "chimera_candidates.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(splits[0].keys()))
            w.writeheader()
            w.writerows(splits)

    print("=" * 62)
    print("CROSS-PASSAGE COMPOSITION (CHIMERA) CHECK")
    print("=" * 62)
    print(f"triples evaluated        : {n} "
          f"({no_prov} without usable provenance)")
    for st in ("CO_LOCATED", "SPLIT", "PARTIAL", "NOT_FOUND"):
        if st in status:
            print(f"  {st:12s} {status[st]:4d}  ({100*rate[st]:.1f} %)")
    print(f"\nchimera candidates (SPLIT): {status.get('SPLIT', 0)} "
          "— LOWER bound, see caveat")
    for st, d in acc.items():
        if d["accept_rate"] is not None:
            print(f"  panel accepts {st:11s}: {100*d['accept_rate']:.1f} % "
                  f"(n={d['n_judged']})")
    n_split = acc.get("SPLIT", {}).get("n_judged", 0)
    if (acc.get("SPLIT", {}).get("accept_rate") is not None
            and acc.get("CO_LOCATED", {}).get("accept_rate") is not None):
        gap = acc["CO_LOCATED"]["accept_rate"] - acc["SPLIT"]["accept_rate"]
        print(f"\n  -> gap = {100*gap:+.1f} pp")
        if n_split < 20:
            print(f"     n={n_split} SPLIT triples: the comparison has no "
                  "statistical power. One triple changing verdict moves the "
                  f"rate by {100/max(n_split,1):.0f} pp. Report the SPLIT "
                  "RATE, not this comparison.")
        elif gap > 0.1:
            print("     verification does discriminate them")
        else:
            print("     verification does not appear to discriminate them")
    print(f"\noutputs in: {outdir}")


if __name__ == "__main__":
    main()