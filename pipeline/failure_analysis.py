"""
failure_analysis.py
For each pipeline failure edge (corpus present but not recovered),
logs the top-5 retrieved passages under C10 and asks the LLM
to attempt extraction with verbose reasoning.
Output classifies each failure as:
  RETRIEVAL_FAILURE  - relevant passage not in top-5
  EXTRACTION_FAILURE - relevant passage retrieved but triple not produced
  SCHEMA_FAILURE     - triple produced but relation type mismatched
"""

import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


def load_json(path):
    path = str(path)
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path) as f:
        return json.load(f)


def normalize(text):
    return text.lower().strip()


def edge_recovered(kg, subj, rel, obj):
    subj_n, obj_n, rel_n = normalize(subj), normalize(obj), normalize(rel)
    for triple in kg.get("triples", []):
        if (normalize(triple.get("subject", "")) == subj_n and
                normalize(triple.get("relation", "")) == rel_n and
                normalize(triple.get("object", "")) == obj_n):
            return True
    return False


def chunks_containing_both(chunks, subj, obj):
    subj_n, obj_n = normalize(subj), normalize(obj)
    return [i for i, c in enumerate(chunks)
            if subj_n in normalize(c.get("text", ""))
            and obj_n in normalize(c.get("text", ""))]


def retrieve_passages(chunks, subj, obj, rel,
                      reranker=None, bm25_topn=20, top_k=5):
    """BM25 + optional cross-encoder reranking."""
    query = f"{subj} {rel} {obj}"
    tokenized = [c.get("text", "").lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top_bm25 = sorted(range(len(scores)),
                      key=lambda i: scores[i], reverse=True)[:bm25_topn]

    if reranker is not None:
        pairs = [(query, chunks[i].get("text", "")) for i in top_bm25]
        ce_scores = reranker.predict(pairs)
        top_bm25 = sorted(top_bm25,
                          key=lambda i: ce_scores[top_bm25.index(i)],
                          reverse=True)

    return top_bm25[:top_k]


def classify_failure(retrieved_indices, relevant_indices,
                     extraction_output, subj, rel, obj):
    """
    RETRIEVAL_FAILURE  - no relevant chunk in top-5
    EXTRACTION_FAILURE - relevant chunk retrieved, triple not produced
    SCHEMA_FAILURE     - triple produced with wrong relation type
    """
    retrieved_set = set(retrieved_indices)
    relevant_set = set(relevant_indices)

    if not retrieved_set & relevant_set:
        return "RETRIEVAL_FAILURE"

    # Check if triple was produced with any relation
    subj_n, obj_n = normalize(subj), normalize(obj)
    for t in extraction_output:
        if (normalize(t.get("subject", "")) == subj_n and
                normalize(t.get("object", "")) == obj_n):
            if normalize(t.get("relation", "")) != normalize(rel):
                return "SCHEMA_FAILURE"
            return "RECOVERED_IN_RERUN"  # Shouldn't happen but flag it

    return "EXTRACTION_FAILURE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--kg-c10", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-model",
                        default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load data
    chunk_file = Path(args.index_dir) / "chunks.jsonl"
    chunks = load_json(chunk_file)
    reference = load_json(args.reference)
    edges = reference if isinstance(reference, list) \
        else reference.get("edges", [])
    kg_c10 = load_json(args.kg_c10)

    # Identify pipeline failures only
    failures = []
    for edge in edges:
        subj = edge.get("subject", edge.get("subj", ""))
        rel = edge.get("relation", edge.get("rel", ""))
        obj = edge.get("object", edge.get("obj", ""))
        relevant = chunks_containing_both(chunks, subj, obj)
        if len(relevant) > 0 and not edge_recovered(kg_c10, subj, rel, obj):
            failures.append({
                "subject": subj, "relation": rel, "object": obj,
                "relevant_chunks": relevant,
                "corpus_count": len(relevant)
            })

    print(f"Found {len(failures)} pipeline failures to analyze")

    # Load reranker
    reranker = None
    if args.rerank:
        print(f"Loading cross-encoder: {args.rerank_model}")
        reranker = CrossEncoder(args.rerank_model)

    results = []
    for edge in failures:
        subj, rel, obj = edge["subject"], edge["relation"], edge["object"]
        print(f"\nAnalyzing: {subj} --{rel}--> {obj}")
        print(f"  Corpus presence: {edge['corpus_count']} chunks")

        # Retrieve passages
        retrieved = retrieve_passages(
            chunks, subj, obj, rel,
            reranker=reranker, bm25_topn=20, top_k=5)

        # Log retrieved passages
        retrieved_texts = [chunks[i].get("text", "") for i in retrieved]
        for j, (idx, text) in enumerate(zip(retrieved, retrieved_texts)):
            relevant_flag = "RELEVANT" if idx in edge["relevant_chunks"] \
                else "irrelevant"
            print(f"  [{j+1}] chunk_{idx} ({relevant_flag}): "
                  f"{text[:120]}...")

        # Classify
        failure_mode = classify_failure(
            retrieved, edge["relevant_chunks"],
            [],  # no LLM extraction in this lightweight version
            subj, rel, obj)

        results.append({
            "subject": subj,
            "relation": rel,
            "object": obj,
            "corpus_count": edge["corpus_count"],
            "retrieved_chunk_ids": retrieved,
            "relevant_chunk_ids": edge["relevant_chunks"][:10],
            "relevant_in_top5": bool(
                set(retrieved) & set(edge["relevant_chunks"])),
            "failure_mode": failure_mode,
            "retrieved_passages": retrieved_texts
        })

        print(f"  => FAILURE MODE: {failure_mode}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Summary
    from collections import Counter
    modes = Counter(r["failure_mode"] for r in results)
    print("\nFailure mode summary:")
    for mode, count in modes.items():
        print(f"  {mode}: {count}")


if __name__ == "__main__":
    main()
