#!/usr/bin/env python3
"""
08_rebuild_provenance.py (v2) — Provenance via evidence re-anchoring.
=====================================================================
CONTEXT (verified on the cluster, run11)
    run11 triples store NO chunk ids. They store the raw `evidence` text
    quoted at verification time, plus support_count=0 / supporting_papers=[].
    The chunk index (chunks.jsonl) additionally contains Jupyter
    '-checkpoint' duplicates of every paper (64 doc_ids for ~32 real papers),
    which this script neutralizes (paper-id normalization + text-hash dedup).

METHOD — two complementary provenance channels, both text-grounded:
    1. EVIDENCE ANCHOR: the stored evidence text is matched back to the
       corpus (exact normalized substring, then token-shingle fuzzy
       fallback). Yields evidence_chunk_id + evidence_paper: the passage
       the verifier actually saw.
    2. CO-OCCURRENCE SUPPORT: number of distinct chunks / papers whose text
       contains BOTH the subject and the object surface forms. This is the
       same construction already used for the 8 extended benchmark edges
       (co-occurrence counts 4-74 chunks), generalized to every triple.
       It is the inter-article consensus signal: support_papers >= 2 means
       at least two independent papers state both terms together.
       (Co-occurrence is necessary-but-weaker evidence than assertion; the
       report labels it as such — it upper-bounds, not proves, consensus.)

CONFIDENCE (unchanged structure, consensus now from co-occurrence papers):
    confidence = w_tier * w_m4 * w_consensus, components always exported.
    w_tier: T1=1.0, T2=0.6, quarantine=0.2
    w_m4  : PREFERRED source is the cross-family panel decision, passed
            with --decisions (ACCEPT=1.0, UNCERTAIN=0.7, REJECT=0.3). This
            is the channel the score is supposed to encode: the panel is
            the independent check, whereas the in-KG `verdict` field holds
            the same-model self-verification that M4 was built to replace.
            Fallback when --decisions is absent: STRONG=1.0 / WEAK=0.7 /
            missing=0.5 from the `verdict` field. A constant w_m4 (e.g. all
            verdicts missing) collapses the score onto w_consensus and is
            reported as a warning, because it silently makes the composite
            untestable.
    w_consensus: log2(1+n_papers)/log2(1+4), capped at 1.

OUTPUTS
    kg_with_provenance.json — same structural format as input, each triple
        gains: evidence_chunk_id, evidence_paper, evidence_match,
        cooc_chunk_ids (capped list), support_chunks, support_papers,
        paper_ids, confidence, conf_components
    provenance_report.csv   — per-triple table sorted by confidence
    provenance_unmatched.csv — triples whose evidence could not be anchored
        (inspection list)

USAGE
    python 08_rebuild_provenance.py \
        --kg ../output/run11_kg/tiered_kg_run11.json \
        --chunks ../output/step1/chunks.jsonl \
        --outdir ../output/analysis
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

from kg_io import (load_kg, dump_kg, load_chunk_records, get_subject,
                   get_object, get_relation, get_m4_verdict)

P_REF = 4
W_TIER = {1: 1.00, 2: 0.60, 0: 0.20}
MAX_COOC_IDS = 30  # cap stored id list to keep the JSON readable


def norm_text(s: str) -> str:
    s = s.lower().replace('"', " ").replace("'", " ")
    return re.sub(r"\s+", " ", s).strip()


def shingles(s: str, k: int = 5) -> set:
    toks = norm_text(s).split()
    return {" ".join(toks[i:i + k]) for i in range(max(len(toks) - k + 1, 1))}


PANEL_W = {"ACCEPT": 1.00, "UNCERTAIN": 0.70, "REJECT": 0.30}


def m4_weight(verdict: str) -> float:
    if "STRONG" in verdict:
        return 1.00
    if "WEAK" in verdict:
        return 0.70
    return 0.50


def load_panel(path):
    """Return {(subject, relation, object) -> decision} from a panel or M4
    decisions jsonl."""
    idx = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        k = (norm_text(d.get("subject", "")), str(d.get("relation", "")).strip(),
             norm_text(d.get("object", "")))
        idx[k] = str(d.get("m4_decision") or d.get("decision") or "").upper()
    print(f"[panel] {len(idx)} decisions loaded from {path}")
    return idx


def consensus_weight(n_papers: int) -> float:
    if n_papers <= 0:
        return 0.0
    return min(1.0, math.log2(1 + n_papers) / math.log2(1 + P_REF))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--outdir", default="output/analysis")
    ap.add_argument("--decisions", default=None,
                    help="m4_panel_decisions.jsonl — cross-family panel "
                         "decisions used as the w_m4 channel (recommended)")
    ap.add_argument("--fuzzy-thr", type=float, default=0.5,
                    help="min shingle containment for fuzzy anchor")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    kg = load_kg(args.kg)
    chunks = load_chunk_records(args.chunks)
    panel = load_panel(args.decisions) if args.decisions else {}
    chunk_norm = [norm_text(c["text"]) for c in chunks]
    chunk_shingles = None  # lazy — only built if fuzzy matching is needed

    n_exact = n_fuzzy = n_unmatched = 0
    rows, unmatched = [], []

    for t in kg["triples"]:
        s, r, o = get_subject(t), get_relation(t), get_object(t)
        sn, on = norm_text(s), norm_text(o)

        # ── channel 1: evidence anchoring ─────────────────────────────
        ev = norm_text(str(t.get("evidence", "")))
        ev_chunk, ev_paper, ev_match = "", "", "none"
        if len(ev) >= 30:
            # exact: mid-slice survives truncated/quoted edges
            probe = ev[8:158] if len(ev) > 60 else ev
            hit = next((i for i, ct in enumerate(chunk_norm)
                        if probe in ct), None)
            if hit is not None:
                ev_chunk, ev_paper, ev_match = (chunks[hit]["chunk_id"],
                                                chunks[hit]["paper"],
                                                "exact")
                n_exact += 1
            else:
                if chunk_shingles is None:
                    chunk_shingles = [shingles(c["text"]) for c in chunks]
                ev_sh = shingles(ev)
                best_i, best_j = None, 0.0
                for i, csh in enumerate(chunk_shingles):
                    inter = len(ev_sh & csh)
                    if not inter:
                        continue
                    j = inter / max(len(ev_sh), 1)  # containment
                    if j > best_j:
                        best_i, best_j = i, j
                if best_i is not None and best_j >= args.fuzzy_thr:
                    ev_chunk, ev_paper = (chunks[best_i]["chunk_id"],
                                          chunks[best_i]["paper"])
                    ev_match = f"fuzzy({best_j:.2f})"
                    n_fuzzy += 1
                else:
                    n_unmatched += 1
                    unmatched.append({"subject": s, "relation": r,
                                      "object": o,
                                      "evidence_head": ev[:120]})
        else:
            n_unmatched += 1
            unmatched.append({"subject": s, "relation": r, "object": o,
                              "evidence_head": ev[:120]})

        # ── channel 2: co-occurrence support ──────────────────────────
        cooc_ids, cooc_papers = [], []
        if sn and on:
            for i, ct in enumerate(chunk_norm):
                if sn in ct and on in ct:
                    cooc_ids.append(chunks[i]["chunk_id"])
                    p = chunks[i]["paper"]
                    if p not in cooc_papers:
                        cooc_papers.append(p)
        if ev_paper and ev_paper not in cooc_papers:
            cooc_papers.append(ev_paper)  # evidence paper always counts

        verdict = get_m4_verdict(t)
        w_t = W_TIER.get(t["_tier"], 0.20)
        pdec = panel.get((sn, r.strip(), on)) if panel else None
        if pdec in PANEL_W:
            w_m = PANEL_W[pdec]
            verdict = verdict or pdec
        else:
            w_m = m4_weight(verdict)
        w_c = consensus_weight(len(cooc_papers))
        conf = round(w_t * w_m * w_c, 4)

        t["evidence_chunk_id"] = ev_chunk
        t["evidence_paper"] = ev_paper
        t["evidence_match"] = ev_match
        t["cooc_chunk_ids"] = cooc_ids[:MAX_COOC_IDS]
        t["support_chunks"] = len(cooc_ids)
        t["support_papers"] = len(cooc_papers)
        t["paper_ids"] = cooc_papers
        t["confidence"] = conf
        t["conf_components"] = {"w_tier": w_t, "w_m4": w_m,
                                "w_consensus": round(w_c, 4)}

        rows.append({"subject": s, "relation": r, "object": o,
                     "tier": t["_tier"], "status": t["_status"],
                     "verdict": verdict or "NA",
                     "evidence_paper": ev_paper,
                     "evidence_match": ev_match,
                     "support_chunks": len(cooc_ids),
                     "support_papers": len(cooc_papers),
                     "confidence": conf,
                     "paper_ids": ";".join(cooc_papers)})

    dump_kg(kg, outdir / "kg_with_provenance.json")
    rows.sort(key=lambda r: -r["confidence"])
    with open(outdir / "provenance_report.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if unmatched:
        with open(outdir / "provenance_unmatched.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(unmatched[0].keys()))
            w.writeheader()
            w.writerows(unmatched)

    n = len(rows)
    multi = sum(1 for r in rows if r["support_papers"] >= 2)
    from collections import Counter as _C
    wm_vals = _C(t["conf_components"]["w_m4"] for t in kg["triples"])
    wt_vals = _C(t["conf_components"]["w_tier"] for t in kg["triples"])
    print("=" * 60)
    print("PROVENANCE REBUILD (v2 — evidence re-anchoring) SUMMARY")
    print("=" * 60)
    print(f"triples processed          : {n}")
    print(f"evidence anchored exact    : {n_exact}")
    print(f"evidence anchored fuzzy    : {n_fuzzy}")
    print(f"evidence unmatched         : {n_unmatched}"
          + ("  -> see provenance_unmatched.csv" if n_unmatched else ""))
    print(f"triples with >=2 papers    : {multi} ({100*multi/max(n,1):.1f}%)"
          "  (co-occurrence consensus, upper bound)")
    for label, vals in (("w_m4", wm_vals), ("w_tier", wt_vals)):
        if len(vals) == 1:
            print(f"!! {label} is CONSTANT ({list(vals)[0]}) across all "
                  f"triples: the composite confidence collapses onto the "
                  f"remaining channels and cannot be validated as a whole."
                  + ("  Pass --decisions <m4_panel_decisions.jsonl> to "
                     "activate this channel." if label == "w_m4" else ""))
        else:
            print(f"{label} distribution: {dict(vals)}")
    print(f"outputs in: {outdir}")


if __name__ == "__main__":
    main()