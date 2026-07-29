#!/usr/bin/env python3
"""
09_analyze_vocabulary.py — Full vocabulary audit of the KG.
===========================================================
WHY
    Residual entity fragmentation (SciBERT threshold 0.06 is deliberately
    ultra-conservative) inflates node counts and corrupts every centrality
    metric downstream. This script measures fragmentation instead of
    guessing it, and produces the ablation material for the P0 item
    "canonicalization protection ablation".

WHAT
    1. vocabulary_report.csv      — per concept: occurrences, papers,
                                    relations, neighbors, degree centrality
    2. synonyms_report.csv        — candidate synonym pairs from THREE
                                    independent detectors:
         rule      : deterministic normalization (hyphens, plurals,
                     spacing, abbreviation expansion)
         string    : character-level similarity (difflib >= --string-thr)
         scibert   : cosine distance in [--merge-thr, --gray-thr]
                     (the "gray zone" the 0.06 threshold refuses to decide)
                     — optional, needs sentence-transformers + model access
    3. canonicalization_report.csv — union of detectors with an ACTION
       column: AUTO_MERGE (rule-level identity), LLM_JUDGE (gray zone,
       to be adjudicated by Llama — cross-family, never Qwen), or
       EXPERT (low-agreement pairs -> Antoine's inspection list)
    4. llm_judgment_pairs.jsonl   — ready-to-run prompts for the LLM judge
       (kept as a separate GPU step so this script stays CPU-only)

    Comparison of approaches is explicit: each pair records WHICH detector(s)
    found it, so rule vs embedding vs string coverage can be reported as a
    table in the paper.

WHERE
    After 08_rebuild_provenance.py. Feeds a possible re-canonicalization and
    the manuscript's vocabulary section.

USAGE
    python 09_analyze_vocabulary.py \
        --kg output/analysis/kg_with_provenance.json \
        --outdir output/analysis [--scibert]
"""

import argparse
import csv
import difflib
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

from kg_io import get_object, get_relation, get_subject, load_kg

# Minimal, documented abbreviation expansions used ONLY for analysis
# (never injected into pipeline construction — no-lexicon rule preserved).
ABBREV = {
    "mtd": "mass transport deposit",
    "mtds": "mass transport deposit",
    "mtc": "mass transport complex",
}


def norm_form(e: str) -> str:
    """Deterministic normal form: lowercase, hyphens->space, squeeze spaces,
    expand known abbreviations, strip trailing plural 's' per token
    (conservative: only for tokens > 3 chars, keeps 'gas', 'toe' safe... note
    'toes' -> 'toe')."""
    x = e.lower().strip()
    x = x.replace("-", " ").replace("_", " ")
    x = re.sub(r"[^\w\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    if x in ABBREV:
        x = ABBREV[x]
    toks = []
    for tok in x.split():
        if (
            len(tok) > 3
            and tok.endswith("s")
            and not tok.endswith(("ss", "is", "us"))
        ):
            tok = tok[:-1]  # keeps debris, hiatus, loess intact
        toks.append(tok)
    return " ".join(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--outdir", default="output/analysis")
    ap.add_argument(
        "--scibert",
        action="store_true",
        help="also run SciBERT gray-zone detection (needs GPU/"
        "model cache; uses allenai/scibert_scivocab_uncased)",
    )
    ap.add_argument("--merge-thr", type=float, default=0.06)
    ap.add_argument("--gray-thr", type=float, default=0.20)
    ap.add_argument("--string-thr", type=float, default=0.85)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    kg = load_kg(args.kg)
    active = [t for t in kg["triples"] if t["_status"] == "active"]

    # ── per-concept table ─────────────────────────────────────────────
    occ = defaultdict(int)
    papers = defaultdict(set)
    rels = defaultdict(set)
    neigh = defaultdict(set)
    for t in active:
        s, o, r = get_subject(t), get_object(t), get_relation(t)
        for e in (s, o):
            occ[e] += 1
            rels[e].add(r)
            for p in t.get("paper_ids", []):
                papers[e].add(p)
        neigh[s].add(o)
        neigh[o].add(s)

    concepts = sorted(occ)
    n_nodes = len(concepts)
    with open(
        outdir / "vocabulary_report.csv", "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "concept",
                "normal_form",
                "n_triples",
                "n_papers",
                "n_relations",
                "n_neighbors",
                "degree_centrality",
            ]
        )
        for c in sorted(concepts, key=lambda c: -occ[c]):
            w.writerow(
                [
                    c,
                    norm_form(c),
                    occ[c],
                    len(papers[c]),
                    len(rels[c]),
                    len(neigh[c]),
                    round(len(neigh[c]) / max(n_nodes - 1, 1), 4),
                ]
            )

    # ── detector 1: deterministic rules ───────────────────────────────
    by_norm = defaultdict(list)
    for c in concepts:
        by_norm[norm_form(c)].append(c)
    rule_pairs = set()
    for group in by_norm.values():
        for a, b in itertools.combinations(sorted(group), 2):
            rule_pairs.add((a, b))

    # ── detector 2: string similarity ─────────────────────────────────
    string_pairs = {}
    for a, b in itertools.combinations(concepts, 2):
        r = difflib.SequenceMatcher(None, a, b).ratio()
        if r >= args.string_thr:
            string_pairs[(a, b)] = round(r, 3)

    # ── detector 3: SciBERT gray zone (optional) ──────────────────────
    scibert_pairs = {}
    if args.scibert:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("allenai/scibert_scivocab_uncased")
            emb = model.encode(concepts, normalize_embeddings=True)
            sims = emb @ emb.T
            for i, j in itertools.combinations(range(len(concepts)), 2):
                dist = 1.0 - float(sims[i, j])
                if args.merge_thr <= dist <= args.gray_thr:
                    scibert_pairs[(concepts[i], concepts[j])] = round(dist, 4)
        except Exception as e:  # keep the CPU path alive on the cluster
            print(f"[WARN] SciBERT detection skipped: {e}")

    # ── merged reports ────────────────────────────────────────────────
    all_pairs = set(rule_pairs) | set(string_pairs) | set(scibert_pairs)
    with open(
        outdir / "synonyms_report.csv", "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "concept_a",
                "concept_b",
                "rule",
                "string_sim",
                "scibert_dist",
                "n_detectors",
            ]
        )
        for a, b in sorted(all_pairs):
            nd = (
                int((a, b) in rule_pairs)
                + int((a, b) in string_pairs)
                + int((a, b) in scibert_pairs)
            )
            w.writerow(
                [
                    a,
                    b,
                    int((a, b) in rule_pairs),
                    string_pairs.get((a, b), ""),
                    scibert_pairs.get((a, b), ""),
                    nd,
                ]
            )

    llm_pairs = []
    with open(
        outdir / "canonicalization_report.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow(["concept_a", "concept_b", "action", "reason"])
        for a, b in sorted(all_pairs):
            if (a, b) in rule_pairs:
                w.writerow(
                    [
                        a,
                        b,
                        "AUTO_MERGE",
                        "identical after deterministic normalization",
                    ]
                )
            elif ((a, b) in string_pairs) and ((a, b) in scibert_pairs):
                w.writerow(
                    [a, b, "LLM_JUDGE", "gray zone, two detectors agree"]
                )
                llm_pairs.append((a, b))
            else:
                w.writerow(
                    [
                        a,
                        b,
                        "LLM_JUDGE" if (a, b) in scibert_pairs else "EXPERT",
                        "single-detector candidate",
                    ]
                )
                if (a, b) in scibert_pairs:
                    llm_pairs.append((a, b))

    with open(outdir / "llm_judgment_pairs.jsonl", "w", encoding="utf-8") as f:
        for a, b in llm_pairs:
            prompt = (
                "You are a geological terminology expert. Answer strictly.\n"
                f'Do "{a}" and "{b}" denote the SAME geological concept '
                "(mere lexical variants), or DIFFERENT concepts?\n"
                "Answer with exactly one line: SAME or DIFFERENT, then a "
                "one-sentence justification."
            )
            f.write(
                json.dumps({"concept_a": a, "concept_b": b, "prompt": prompt})
                + "\n"
            )

    print("=" * 60)
    print("VOCABULARY AUDIT SUMMARY")
    print("=" * 60)
    print(f"unique concepts          : {len(concepts)}")
    print(f"rule-identical pairs     : {len(rule_pairs)}  (AUTO_MERGE)")
    print(f"string-similar pairs     : {len(string_pairs)}")
    print(
        f"scibert gray-zone pairs  : {len(scibert_pairs)}"
        + ("" if args.scibert else "  (detector disabled, use --scibert)")
    )
    print(
        f"pairs sent to LLM judge  : {len(llm_pairs)} "
        f"(llm_judgment_pairs.jsonl — run with Llama, NOT Qwen)"
    )
    print(f"outputs in: {outdir}")


if __name__ == "__main__":
    main()
