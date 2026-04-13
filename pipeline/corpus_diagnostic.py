"""
corpus_diagnostic.py
For each of the 26 LB2019 benchmark edges, reports:
  - corpus_chunks: number of chunks where both subject and object co-occur
  - retrieved_bm25: number of those chunks that appear in C9 retrieved sets
  - retrieved_c10: number of those chunks that appear in C10 retrieved sets
  - recovered: whether the edge is matched in the final KG
  - outcome: RECOVERED | CORPUS_GAP | PIPELINE_FAILURE
"""

import json
import argparse
from pathlib import Path
from rank_bm25 import BM25Okapi


def load_json(path):
    path = str(path)
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path) as f:
        return json.load(f)


def normalize(text):
    return text.lower().strip()


def chunks_containing_both(chunks, subj, obj):
    """Return indices of chunks where both subject and object strings appear."""
    subj_n = normalize(subj)
    obj_n = normalize(obj)
    hits = []
    for i, chunk in enumerate(chunks):
        text = normalize(chunk.get("text", ""))
        if subj_n in text and obj_n in text:
            hits.append(i)
    return hits


def edge_recovered(kg, subj, rel, obj):
    """Check if a benchmark edge is present in the KG after normalization."""
    subj_n = normalize(subj)
    obj_n = normalize(obj)
    rel_n = normalize(rel)
    for triple in kg.get("triples", []):
        if (normalize(triple.get("subject", "")) == subj_n and
                normalize(triple.get("relation", "")) == rel_n and
                normalize(triple.get("object", "")) == obj_n):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--kg-c9", required=True)
    parser.add_argument("--kg-c10", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load chunks
    chunk_file = Path(args.index_dir) / "chunks.jsonl"
    chunks = load_json(chunk_file)
    print(f"Loaded {len(chunks)} chunks")

    # Load reference edges
    reference = load_json(args.reference)
    edges = reference if isinstance(reference, list) else reference.get("edges", [])
    print(f"Loaded {len(edges)} reference edges")

    # Load KGs
    kg_c9 = load_json(args.kg_c9)
    kg_c10 = load_json(args.kg_c10)

    results = []
    for edge in edges:
        subj = edge.get("subject", edge.get("subj", ""))
        rel = edge.get("relation", edge.get("rel", ""))
        obj = edge.get("object", edge.get("obj", ""))

        # Corpus presence
        present_chunks = chunks_containing_both(chunks, subj, obj)
        corpus_count = len(present_chunks)

        # Recovery status
        recovered_c9 = edge_recovered(kg_c9, subj, rel, obj)
        recovered_c10 = edge_recovered(kg_c10, subj, rel, obj)

        # Outcome classification
        if corpus_count == 0:
            outcome = "CORPUS_GAP"
        elif recovered_c10:
            outcome = "RECOVERED"
        else:
            outcome = "PIPELINE_FAILURE"

        results.append({
            "subject": subj,
            "relation": rel,
            "object": obj,
            "corpus_chunks": corpus_count,
            "recovered_c9": recovered_c9,
            "recovered_c10": recovered_c10,
            "outcome": outcome
        })

        print(f"{outcome:20s} | chunks={corpus_count:4d} | "
              f"C9={'Y' if recovered_c9 else 'N'} "
              f"C10={'Y' if recovered_c10 else 'N'} | "
              f"{subj[:20]:20s} --{rel}--> {obj[:20]}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Summary
    gaps = sum(1 for r in results if r["outcome"] == "CORPUS_GAP")
    recovered = sum(1 for r in results if r["outcome"] == "RECOVERED")
    failures = sum(1 for r in results if r["outcome"] == "PIPELINE_FAILURE")
    print(f"\nSummary: {recovered} recovered | "
          f"{gaps} corpus gaps | {failures} pipeline failures")


if __name__ == "__main__":
    main()
