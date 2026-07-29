#!/usr/bin/env python3
"""
m4_quote_verify.py — V1: Deterministic quote verification
===========================================================

Model-free, fully reproducible audit of grounding traceability.
Two independent checks:

  Check A — PIPELINE quotes: for every triple in the KG, does its
            evidence quote (produced at extraction time) actually exist
            in its provenance passage (and, fallback, anywhere in the
            corpus chunk index)?
  Check B — M4 JUDGE quotes: for every evidence-pass verdict, does the
            QUOTE copied by the judge actually exist in the passage it
            was shown? (audits the verifier itself)

Matching ladder (applied in order):
  EXACT      normalized quote is a substring of the normalized target
  NEAR       best difflib similarity >= --near-threshold (default 0.85)
             (tolerates OCR artefacts, dehyphenation, ellipses)
  NOT_FOUND  neither

Normalization: lowercase, collapse whitespace, strip quotation marks and
bracketed ellipses. Purely deterministic; no model, no seed.

Outputs:
  quote_verification.jsonl   per-item result
  quote_verification_summary.json

Usage:
    python m4_quote_verify.py \
        --kg       ~/ontogeorag/output/run11_kg/tiered_kg_run11.json \
        --verdicts ~/ontogeorag/output/m4/m4_verdicts.jsonl \
        --index    ~/ontogeorag/output/step1 \
        --output   ~/ontogeorag/output/m4
"""

import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path

from m4_verify import (
    evidence_from_provenance,
    load_chunk_index,
    load_triples,
    triple_fields,
)


def normalize(text: str) -> str:
    """Lowercase `text`, fix known OCR ligature artefacts, strip bracketed ellipses/quotation marks, and collapse to alphanumeric tokens; no side effects."""
    t = text.lower()
    # OCR ligature artefacts present in the corpus PDFs
    t = t.replace("¢", "fi").replace("£", "fl")
    t = t.replace("ﬁ", "fi").replace("ﬂ", "fl")
    t = re.sub(r"\[\s*(\.\.\.|…)\s*\]", " ", t)  # bracketed ellipses
    t = re.sub(r"[\"'“”‘’«»]", "", t)  # quotation marks
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()  # punct/hyphen variance
    return t


def pipeline_quote(t: dict) -> str:
    """Extract the extraction-time evidence quote string from triple `t`'s `evidence` field, if present; no side effects."""
    ev = t.get("evidence", {})
    if isinstance(ev, dict):
        return str(ev.get("quote", "") or "")
    return ""


def best_similarity(quote: str, target: str) -> float:
    """Best difflib ratio of quote vs sliding windows of target."""
    if not quote or not target:
        return 0.0
    lq = len(quote)
    if lq >= len(target):
        return difflib.SequenceMatcher(None, quote, target).ratio()
    best = 0.0
    step = max(1, lq // 4)
    for start in range(0, len(target) - lq + 1, step):
        window = target[start : start + lq + step]
        r = difflib.SequenceMatcher(None, quote, window).ratio()
        if r > best:
            best = r
            if best > 0.98:
                break
    return best


def classify(quote: str, target: str, near: float) -> tuple:
    """Classify `quote` against `target` as EXACT/EXACT_FRAGMENTED/NEAR (>= `near` similarity)/NOT_FOUND/EMPTY_QUOTE/NO_TARGET; returns (status, similarity_score), no side effects."""
    nq, nt = normalize(quote), normalize(target)
    if not nq:
        return "EMPTY_QUOTE", 0.0
    if not nt:
        return "NO_TARGET", 0.0
    if nq in nt:
        return "EXACT", 1.0
    # split on ellipses: all fragments must be found for EXACT-fragmented
    frags = [f.strip() for f in re.split(r"\.\.\.|…", nq) if f.strip()]
    if len(frags) > 1 and all(f in nt for f in frags):
        return "EXACT_FRAGMENTED", 1.0
    sim = best_similarity(nq, nt)
    if sim >= near:
        return "NEAR", round(sim, 3)
    return "NOT_FOUND", round(sim, 3)


def load_jsonl(path):
    """Read `path` as one JSON object per line; no side effects."""
    out = []
    with open(Path(path).expanduser(), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    """CLI entry point: audits --kg pipeline quotes against their provenance passage (Check A) and, if --verdicts given, M4 judge quotes against their shown passage (Check B); writes quote_verification.jsonl + quote_verification_summary.json to --output."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument(
        "--verdicts", default=None, help="m4_verdicts.jsonl (enables Check B)"
    )
    ap.add_argument(
        "--index",
        default=None,
        help="Chunk index dir (corpus-wide fallback for Check A)",
    )
    ap.add_argument("--near-threshold", type=float, default=0.85)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    triples = load_triples(Path(args.kg).expanduser())
    chunks = (
        load_chunk_index(Path(args.index).expanduser()) if args.index else []
    )
    chunk_norm = [
        normalize(c.get("text", "") if isinstance(c, dict) else str(c))
        for c in chunks
    ]

    results = []
    counts_a = Counter()
    counts_b = Counter()

    # ── Check A: pipeline quotes vs provenance passage (+corpus fallback)
    for i, t in enumerate(triples):
        subj, rel, obj = triple_fields(t)
        quote = pipeline_quote(t)
        passage = evidence_from_provenance(t)
        status, score = classify(quote, passage, args.near_threshold)

        corpus_status = None
        if status == "NOT_FOUND" and chunks:
            nq = normalize(quote)
            frags = [
                f.strip() for f in re.split(r"\.\.\.|…", nq) if f.strip()
            ] or [nq]
            found = any(all(f in cn for f in frags) for cn in chunk_norm)
            corpus_status = (
                "FOUND_ELSEWHERE_IN_CORPUS" if found else "NOT_IN_CORPUS"
            )

        counts_a[status] += 1
        if corpus_status:
            counts_a[corpus_status] += 1
        results.append(
            {
                "check": "A_pipeline_quote",
                "index": i,
                "subject": subj,
                "relation": rel,
                "object": obj,
                "status": status,
                "similarity": score,
                "corpus_fallback": corpus_status,
            }
        )

    # ── Check B: M4 judge quotes vs the passage it was shown
    if args.verdicts:
        verdicts = load_jsonl(args.verdicts)
        for v in verdicts:
            ev = v.get("evidence", {})
            if not ev.get("passage_found"):
                continue
            quote = str(ev.get("quote", "") or "")
            if normalize(quote) in ("", "no evidence found"):
                counts_b["NO_EVIDENCE_DECLARED"] += 1
                continue
            idx = v["m4_index"]
            passage = (
                evidence_from_provenance(triples[idx])
                if idx < len(triples)
                else ""
            )
            status, score = classify(quote, passage, args.near_threshold)
            counts_b[status] += 1
            results.append(
                {
                    "check": "B_judge_quote",
                    "index": idx,
                    "subject": v["subject"],
                    "relation": v["relation"],
                    "object": v["object"],
                    "evidence_verdict": ev.get("verdict"),
                    "status": status,
                    "similarity": score,
                }
            )

    out_path = out_dir / "quote_verification.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def rates(c):
        tot = sum(
            v
            for k, v in c.items()
            if k
            in (
                "EXACT",
                "EXACT_FRAGMENTED",
                "NEAR",
                "NOT_FOUND",
                "EMPTY_QUOTE",
                "NO_TARGET",
            )
        )
        return {
            k: {"n": v, "rate": round(v / tot, 4) if tot else None}
            for k, v in sorted(c.items())
        }

    summary = {
        "near_threshold": args.near_threshold,
        "check_A_pipeline_quotes": rates(counts_a),
        "check_B_judge_quotes": rates(counts_b) if args.verdicts else None,
        "note": (
            "Check A audits extraction-time grounding; Check B audits "
            "the verifier's own quoting. Both are deterministic and "
            "model-free."
        ),
    }
    (out_dir / "quote_verification_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"\nDetails: {out_path}")


if __name__ == "__main__":
    main()
