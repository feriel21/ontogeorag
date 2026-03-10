#!/usr/bin/env python3
"""
EXP-A: Corpus Recoverability Analysis
======================================
For each of the 26 LB2019 reference edges, determines whether the relevant
passage is present in the corpus (BM25-retrievable) or not.

Classifies each edge as:
  - CORPUS_PRESENT   : BM25 retrieves ≥1 chunk with score ≥ threshold
  - CORPUS_MARGINAL  : BM25 retrieves chunks but score is low (0 < score < threshold)
  - CORPUS_ABSENT    : No relevant chunk found → pipeline cannot recover this edge

This answers the key reviewer question:
  "Is the 34.6% recall a pipeline failure or a corpus coverage limitation?"

Runtime: ~10–15 minutes (pure Python, no GPU needed)

Usage:
  python exp_a_corpus_recoverability.py \
      --bm25   ~/kg_test/output/step2/bm25_index.json \
      --ref    ~/mtd-kg-pipeline/mtd-kg-pipeline/configs/lb_reference_edges.json \
      --out    ~/kg_test/output/exp_a/
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


# ── BM25 implementation (pure Python, mirrors what the pipeline uses) ──

def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())

class BM25:
    def __init__(self, corpus_docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = corpus_docs
        self.N = len(corpus_docs)
        self.avgdl = sum(len(d) for d in corpus_docs) / max(self.N, 1)
        self.idf = {}
        self.tf = []
        df = defaultdict(int)
        for doc in corpus_docs:
            seen = set()
            freq = defaultdict(int)
            for t in doc:
                freq[t] += 1
                if t not in seen:
                    df[t] += 1
                    seen.add(t)
            self.tf.append(freq)
        for term, n in df.items():
            self.idf[term] = math.log((self.N - n + 0.5) / (n + 0.5) + 1)

    def score(self, query_tokens, doc_idx):
        s = 0.0
        dl = len(self.docs[doc_idx])
        tf_map = self.tf[doc_idx]
        for t in query_tokens:
            if t not in tf_map:
                continue
            idf = self.idf.get(t, 0)
            tf = tf_map[t]
            s += idf * (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
        return s

    def top_k(self, query_tokens, k=5):
        scores = [(i, self.score(query_tokens, i)) for i in range(self.N)]
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


# ── Query generation for each reference edge ──────────────────────────

def make_queries_for_edge(edge):
    """Generate 3–5 diverse BM25 queries for a reference edge."""
    s, r, o = edge["subject"], edge["relation"], edge["object"]
    queries = []

    # Direct mention query
    queries.append(f"{s} {o}")

    # Relation-specific expansions
    if r == "hasDescriptor":
        queries.append(f"{s} seismic character {o}")
        queries.append(f"{o} reflection {s}")
        queries.append(f"{s} appears {o} on seismic")
    elif r == "occursIn":
        queries.append(f"{s} found in {o}")
        queries.append(f"{o} {s} environment setting")
    elif r in ("causes", "triggers", "controls"):
        queries.append(f"{s} {r} {o}")
        queries.append(f"{o} caused triggered by {s}")
    elif r == "formedBy":
        queries.append(f"{s} formed by {o}")
        queries.append(f"{o} generates produces {s}")
    elif r in ("overlies", "underlies"):
        queries.append(f"{s} above below {o} stratigraphy")
    else:
        queries.append(f"{s} related to {o}")

    return queries


# ── Main logic ────────────────────────────────────────────────────────

def load_bm25_index(bm25_path):
    """Load the pre-built BM25 index JSON.
    Expected format: list of {"id": ..., "text": ..., "chunk_id": ...}
    or dict with "chunks" key.
    """
    with open(bm25_path) as f:
        raw = json.load(f)

    # Handle multiple possible formats
    if isinstance(raw, list):
        chunks = raw
    elif isinstance(raw, dict):
        chunks = raw.get("chunks", raw.get("documents", raw.get("index", [])))
        if not chunks:
            # Maybe it's {"chunk_id": text, ...}
            chunks = [{"id": k, "text": v if isinstance(v, str) else v.get("text", "")}
                      for k, v in raw.items()]
    else:
        raise ValueError(f"Unexpected BM25 index format: {type(raw)}")

    # Normalise to list of dicts with 'text'
    normalized = []
    for c in chunks:
        if isinstance(c, str):
            normalized.append({"id": len(normalized), "text": c})
        elif isinstance(c, dict):
            text = c.get("text", c.get("content", c.get("chunk", "")))
            normalized.append({"id": c.get("id", c.get("chunk_id", len(normalized))), "text": text})

    return normalized


def run_exp_a(bm25_path, ref_path, out_dir, top_k=10, threshold=2.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load reference edges
    with open(ref_path) as f:
        ref_raw = json.load(f)
    if isinstance(ref_raw, dict):
        edges = ref_raw.get("edges", [])
    else:
        edges = ref_raw

    print(f"[EXP-A] Loaded {len(edges)} reference edges")

    # Load corpus chunks
    print(f"[EXP-A] Loading BM25 index from {bm25_path} ...")
    chunks = load_bm25_index(bm25_path)
    print(f"[EXP-A] Corpus: {len(chunks)} chunks")

    # Build BM25
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25(tokenized_corpus)

    # --- Analyse each edge ---
    results = []
    category_counts = {"CORPUS_PRESENT": 0, "CORPUS_MARGINAL": 0, "CORPUS_ABSENT": 0}

    for edge in edges:
        s, r, o = edge["subject"], edge["relation"], edge["object"]
        queries = make_queries_for_edge(edge)

        best_score = 0.0
        best_chunk_id = None
        best_chunk_text = ""
        best_query = ""

        for q in queries:
            q_tokens = tokenize(q)
            if not q_tokens:
                continue
            hits = bm25.top_k(q_tokens, k=top_k)
            if hits and hits[0][1] > best_score:
                best_score = hits[0][1]
                best_chunk_id = chunks[hits[0][0]]["id"]
                best_chunk_text = chunks[hits[0][0]]["text"][:300]
                best_query = q

        # Classify
        if best_score >= threshold:
            category = "CORPUS_PRESENT"
        elif best_score > 0:
            category = "CORPUS_MARGINAL"
        else:
            category = "CORPUS_ABSENT"

        category_counts[category] += 1

        result = {
            "subject": s,
            "relation": r,
            "object": o,
            "edge_key": f"{s} --{r}--> {o}",
            "category": category,
            "best_bm25_score": round(best_score, 4),
            "best_query": best_query,
            "best_chunk_id": best_chunk_id,
            "best_chunk_preview": best_chunk_text,
        }
        results.append(result)

        symbol = {"CORPUS_PRESENT": "✓", "CORPUS_MARGINAL": "~", "CORPUS_ABSENT": "✗"}[category]
        print(f"  {symbol} [{category:18s}] score={best_score:5.2f}  {s} --{r}--> {o}")

    # --- Summary stats ---
    n = len(edges)
    present = category_counts["CORPUS_PRESENT"]
    marginal = category_counts["CORPUS_MARGINAL"]
    absent = category_counts["CORPUS_ABSENT"]
    recoverable = present + marginal

    print("\n" + "="*60)
    print("EXP-A RESULTS — Corpus Recoverability")
    print("="*60)
    print(f"  Total reference edges  : {n}")
    print(f"  CORPUS_PRESENT  (≥{threshold:.1f}) : {present:2d}  ({100*present/n:.1f}%)")
    print(f"  CORPUS_MARGINAL (>0)  : {marginal:2d}  ({100*marginal/n:.1f}%)")
    print(f"  CORPUS_ABSENT         : {absent:2d}  ({100*absent/n:.1f}%)")
    print(f"  --- Recoverable total : {recoverable:2d}  ({100*recoverable/n:.1f}%)")
    print()

    # Compute corrected recall ceiling
    # Pipeline found 9 edges at Tier1+2 (34.6% of 26)
    pipeline_found = 9
    if recoverable > 0:
        corrected_recall = pipeline_found / recoverable * 100
    else:
        corrected_recall = 0
    print(f"  Pipeline recall vs ALL   edges : {pipeline_found}/{n} = {100*pipeline_found/n:.1f}%")
    print(f"  Pipeline recall vs RECOVERABLE : {pipeline_found}/{recoverable} = {corrected_recall:.1f}%")
    print()

    # Absent edges — these are the key finding
    absent_edges = [r for r in results if r["category"] == "CORPUS_ABSENT"]
    if absent_edges:
        print("  ABSENT edges (pipeline cannot recover — corpus limitation, not pipeline failure):")
        for e in absent_edges:
            print(f"    - {e['edge_key']}")
    print("="*60)

    # --- Save outputs ---
    # Full per-edge JSON
    out_json = out_dir / "exp_a_recoverability_details.json"
    with open(out_json, "w") as f:
        json.dump({
            "description": "EXP-A Corpus Recoverability Analysis",
            "bm25_threshold": threshold,
            "n_chunks": len(chunks),
            "n_reference_edges": n,
            "summary": {
                "CORPUS_PRESENT": present,
                "CORPUS_MARGINAL": marginal,
                "CORPUS_ABSENT": absent,
                "recoverable_total": recoverable,
                "pipeline_found": pipeline_found,
                "recall_vs_all": round(pipeline_found/n, 4),
                "recall_vs_recoverable": round(pipeline_found/max(recoverable,1), 4),
            },
            "edges": results
        }, f, indent=2)
    print(f"\n[EXP-A] Full details saved → {out_json}")

    # Paper-ready summary CSV
    import csv
    out_csv = out_dir / "exp_a_recoverability_table.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "subject", "relation", "object", "category", "best_bm25_score"
        ])
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in ["subject", "relation", "object", "category", "best_bm25_score"]})
    print(f"[EXP-A] Paper table saved  → {out_csv}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25",  default=os.path.expanduser(
        "~/kg_test/output/step2/bm25_index.json"))
    parser.add_argument("--ref",   default=os.path.expanduser(
        "~/mtd-kg-pipeline/mtd-kg-pipeline/configs/lb_reference_edges.json"))
    parser.add_argument("--out",   default=os.path.expanduser(
        "~/kg_test/output/exp_a/"))
    parser.add_argument("--threshold", type=float, default=2.0,
        help="BM25 score threshold for CORPUS_PRESENT classification")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    run_exp_a(args.bm25, args.ref, args.out, top_k=args.topk, threshold=args.threshold)