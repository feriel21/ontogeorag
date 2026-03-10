#!/usr/bin/env python3
"""
experiments/exp_e_llama_extraction.py — Llama-3.1-8B vs Qwen-7B Extraction Comparison

Runs the same BM25-RAG extraction pipeline with Llama-3.1-8B and compares
output quality against the Qwen-7B Run-7 results.

Comparison dimensions:
  1. Extraction yield (triples per query)
  2. Relation distribution (does Llama over/under-produce certain relations?)
  3. LB2019 descriptor coverage (how many of the 13 descriptors recovered?)
  4. Hallucination rate after Qwen-7B verification (cross-model check)
  5. Entity quality (vague, blacklisted, self-loops)

This is EXP-E — designed after EXP-D showed Llama is permissive as a VERIFIER.
The question here is: is Llama better or worse as an EXTRACTOR?

Runtime: ~8–12 hours on CPU (same corpus, same queries)
Submit as SLURM job: sbatch slurm/run_exp_e.sh

Usage:
    python experiments/exp_e_llama_extraction.py \\
        --index-dir output/step1/ \\
        --schema    configs/ontology_schema.json \\
        --queries   configs/descriptor_queries.jsonl \\
        --qwen-ref  output/step7/raw_triples_v7.jsonl \\
        --out       output/exp_e/ \\
        --model     meta-llama/Llama-3.1-8B-Instruct \\
        --backend   hf
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from pipeline.rag.constants import (
    ALLOWED_RELATIONS, LB2019_DESCRIPTORS, LB2019_REFERENCE_EDGES,
    normalize_entity, normalize_relation, normalize_descriptor,
)


# ── Shared helpers ────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    triples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                triples.append(json.loads(line))
    return triples


def triple_key(t: dict) -> tuple:
    return (
        normalize_entity(t.get("source", t.get("subject", ""))),
        normalize_relation(t.get("relation", "")),
        normalize_entity(t.get("target", t.get("object", ""))),
    )


# ── Metrics ───────────────────────────────────────────────────────────

def compute_descriptor_coverage(triples: list[dict]) -> dict:
    found = set()
    for t in triples:
        r = normalize_relation(t.get("relation", ""))
        if r == "hasDescriptor":
            d = normalize_descriptor(t.get("target", t.get("object", "")))
            if d in LB2019_DESCRIPTORS:
                found.add(d)
    missing = LB2019_DESCRIPTORS - found
    return {
        "found":    sorted(found),
        "missing":  sorted(missing),
        "n_found":  len(found),
        "n_total":  len(LB2019_DESCRIPTORS),
        "coverage": len(found) / len(LB2019_DESCRIPTORS),
    }


def compute_lb_recall(triples: list[dict]) -> dict:
    kg = set()
    for t in triples:
        kg.add((
            normalize_entity(t.get("source", t.get("subject", ""))),
            normalize_relation(t.get("relation", "")),
            normalize_entity(t.get("target", t.get("object", ""))),
        ))
    ref = [(normalize_entity(s), normalize_relation(r), normalize_entity(o))
           for s, r, o in LB2019_REFERENCE_EDGES]
    hits  = [k for k in ref if k in kg]
    return {
        "hits":   len(hits),
        "total":  len(ref),
        "recall": len(hits) / len(ref),
        "matched_edges": [list(h) for h in hits],
    }


def compute_relation_dist(triples: list[dict]) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for t in triples:
        r = normalize_relation(t.get("relation", ""))
        counts[r] += 1
    total = sum(counts.values())
    return {
        "counts": dict(counts),
        "total":  total,
        "rates":  {r: round(c / total, 4) for r, c in counts.items()} if total > 0 else {},
    }


def dedup(triples: list[dict]) -> list[dict]:
    seen, out = set(), []
    for t in triples:
        k = triple_key(t)
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


# ── Extraction (same as pipeline/02) ─────────────────────────────────

def load_bm25(index_dir: str):
    from rank_bm25 import BM25Okapi
    chunks_path = Path(index_dir) / "chunks.jsonl"
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    corpus = [(c.get("text", "")).lower().split() for c in chunks]
    bm25   = BM25Okapi(corpus)

    def retrieve(query: str, top_n: int = 25) -> list[dict]:
        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)
        top_i  = scores.argsort()[-top_n:][::-1]
        return [
            {"chunk_id": chunks[i].get("chunk_id", f"chunk_{i}"),
             "text":     chunks[i].get("text", ""),
             "score":    float(scores[i]),
             "source_file": chunks[i].get("source_file", "?")}
            for i in top_i
        ]
    return retrieve


EXTRACTION_PROMPT = """You are a geological knowledge extraction system.
Given this text excerpt from a scientific paper about mass-transport deposits:

---
{chunk}
---

Question: {query}

Extract geological triples as JSON. Allowed relations: {relations}

Respond with a JSON array:
[{{"source": "...", "source_type": "SeismicObject", "relation": "...", "target": "...", "target_type": "..."}}]

If nothing to extract: []
"""


def run_llama_extraction(args) -> list[dict]:
    """Run full extraction with Llama model, return raw triples."""
    from pipeline.rag.llm_hf import make_hf_fn
    import re

    generate = make_hf_fn(args.model)

    queries = []
    with open(args.queries, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    print(f"  Queries: {len(queries)}")

    schema = json.loads(Path(args.schema).read_text())
    relations = [r["name"] if isinstance(r, dict) else r for r in schema.get("relations", [])]

    retrieve = load_bm25(args.index_dir)

    triples, t0 = [], time.time()

    for qi, q in enumerate(queries):
        query_text = (q.get("query", q.get("text", "")) or "").strip()
        if not query_text:
            continue

        candidates = retrieve(query_text, top_n=25)
        if not candidates or candidates[0]["score"] < 2.0:
            continue

        selected = candidates[:3]
        context  = "\n---\n".join(c["text"][:800] for c in selected)[:2500]

        prompt = EXTRACTION_PROMPT.format(
            chunk=context,
            query=query_text,
            relations=", ".join(relations),
        )

        try:
            response = generate("", prompt)
            m = re.search(r"\[.*\]", response, re.DOTALL)
            if m:
                items = json.loads(m.group())
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    rel_raw  = item.get("relation", "")
                    rel_norm = normalize_relation(rel_raw)
                    if rel_norm not in ALLOWED_RELATIONS:
                        continue
                    item["relation"] = rel_norm
                    item["_provenance"] = {
                        "query": query_text,
                        "model": args.model,
                        "context_preview": context[:800],
                        "best_bm25": candidates[0]["score"],
                    }
                    triples.append(item)
        except Exception as e:
            pass

        if (qi + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{qi+1}/{len(queries)}] triples={len(triples)} ({(qi+1)/elapsed:.2f} q/s)")

    return triples


# ── Comparison ────────────────────────────────────────────────────────

def compare(llama_triples: list[dict], qwen_triples: list[dict]) -> dict:
    llama_d = dedup(llama_triples)
    qwen_d  = dedup(qwen_triples)

    llama_keys = set(triple_key(t) for t in llama_d)
    qwen_keys  = set(triple_key(t) for t in qwen_d)

    overlap = llama_keys & qwen_keys
    only_llama = llama_keys - qwen_keys
    only_qwen  = qwen_keys  - llama_keys

    return {
        "llama_total":     len(llama_d),
        "qwen_total":      len(qwen_d),
        "overlap":         len(overlap),
        "only_llama":      len(only_llama),
        "only_qwen":       len(only_qwen),
        "jaccard":         len(overlap) / len(llama_keys | qwen_keys) if (llama_keys | qwen_keys) else 0,
        "llama_coverage":  compute_descriptor_coverage(llama_d),
        "qwen_coverage":   compute_descriptor_coverage(qwen_d),
        "llama_recall":    compute_lb_recall(llama_d),
        "qwen_recall":     compute_lb_recall(qwen_d),
        "llama_relations": compute_relation_dist(llama_d),
        "qwen_relations":  compute_relation_dist(qwen_d),
    }


def main():
    parser = argparse.ArgumentParser(description="EXP-E: Llama vs Qwen extraction comparison")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--schema",    required=True)
    parser.add_argument("--queries",   required=True)
    parser.add_argument("--qwen-ref",  required=True, help="Qwen Run-7 raw triples JSONL")
    parser.add_argument("--out",       default="output/exp_e/")
    parser.add_argument("--model",     default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--backend",   default="hf")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXP-E: Llama-3.1-8B vs Qwen-7B Extraction Comparison")
    print("=" * 60)
    print(f"\n  Llama model: {args.model}")

    # Run Llama extraction
    print("\n[Phase 1] Running Llama extraction...")
    llama_triples = run_llama_extraction(args)
    llama_out = out_dir / "llama_raw_triples.jsonl"
    with open(llama_out, "w", encoding="utf-8") as f:
        for t in llama_triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"  Llama extracted: {len(llama_triples)} triples → {llama_out}")

    # Load Qwen reference
    print(f"\n[Phase 2] Loading Qwen reference: {args.qwen_ref}")
    qwen_triples = load_jsonl(args.qwen_ref)
    print(f"  Qwen triples loaded: {len(qwen_triples)}")

    # Compare
    print("\n[Phase 3] Comparing...")
    comp = compare(llama_triples, qwen_triples)

    print("\n" + "=" * 60)
    print("EXP-E RESULTS")
    print("=" * 60)
    print(f"\n  {'Metric':<40} {'Llama':>8} {'Qwen':>8}")
    print(f"  {'-'*56}")
    print(f"  {'Total unique triples':<40} {comp['llama_total']:>8} {comp['qwen_total']:>8}")
    print(f"  {'Descriptor coverage (n/13)':<40} {comp['llama_coverage']['n_found']:>8} {comp['qwen_coverage']['n_found']:>8}")
    print(f"  {'Descriptor coverage (%)':<40} {comp['llama_coverage']['coverage']*100:>7.1f}% {comp['qwen_coverage']['coverage']*100:>7.1f}%")
    print(f"  {'LB2019 recall (n/26)':<40} {comp['llama_recall']['hits']:>8} {comp['qwen_recall']['hits']:>8}")
    print(f"  {'LB2019 recall (%)':<40} {comp['llama_recall']['recall']*100:>7.1f}% {comp['qwen_recall']['recall']*100:>7.1f}%")
    print(f"\n  Triple overlap: {comp['overlap']} | Jaccard: {comp['jaccard']:.3f}")
    print(f"  Only Llama: {comp['only_llama']} | Only Qwen: {comp['only_qwen']}")

    print(f"\n  Llama descriptor coverage: {comp['llama_coverage']['found']}")
    print(f"  Llama missing:             {comp['llama_coverage']['missing']}")

    stats_out = out_dir / "exp_e_stats.json"
    Path(stats_out).write_text(json.dumps(comp, indent=2))
    print(f"\n  Stats saved: {stats_out}")


if __name__ == "__main__":
    main()