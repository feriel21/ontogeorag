#!/usr/bin/env python3
"""
pipeline/02_extract_triples.py — RAG Triple Extraction

BM25 retrieval → multi-chunk context → LLM extraction.
Supports Qwen, Llama, Mistral (any chat-template HF model) via --model.

Key design decisions (documented for reviewers):
  - Top-k chunk concatenation (k=3) gives the LLM the same context seen during human reading
  - Per-strategy BM25 gating thresholds prevent low-signal queries from consuming LLM calls
  - Brace escaping ({{ }}) in prompt templates prevents Python .format() KeyError on JSON examples
  - Salience levels (typical/common/occasional) are required for hasDescriptor triples

Usage:
    python pipeline/02_extract_triples.py \\
        --index-dir output/step1/ \\
        --schema    configs/ontology_schema.json \\
        --queries   configs/descriptor_queries.jsonl \\
        --output    output/step2/raw_triples.jsonl \\
        --model     Qwen/Qwen2.5-7B-Instruct \\
        --backend   hf

    # Llama comparison (EXP-E):
    python pipeline/02_extract_triples.py \\
        --index-dir output/step1/ \\
        --schema    configs/ontology_schema.json \\
        --queries   configs/descriptor_queries.jsonl \\
        --output    output/exp_e/raw_triples_llama.jsonl \\
        --model     meta-llama/Llama-3.1-8B-Instruct \\
        --backend   hf
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline.rag.constants import ALLOWED_RELATIONS, normalize_relation


# ── Prompt templates ──────────────────────────────────────────────────
# IMPORTANT: literal JSON braces in .format() templates must be {{ }}

DESCRIPTOR_PROMPT = """You are a geological knowledge extraction system.

Given the following text excerpt from a scientific paper about mass-transport deposits (MTDs):

---
{chunk_text}
---

Question: {query}

Extract geological relationships as JSON triples. For DESCRIPTOR relationships, indicate salience:
- "typical": DIAGNOSTIC — almost always present, experts list it first
- "common": FREQUENT — often present but not defining
- "occasional": POSSIBLE — occurs in some cases but not characteristic

CRITICAL RULES:
1. Only extract relationships EXPLICITLY stated or DIRECTLY implied by this text
2. Use ONLY these relations: {relations}
3. Use ONLY these entity types: {types}

Respond with a JSON array. Each object must have:
  "source", "source_type", "relation", "target", "target_type"
  "salience": "typical" | "common" | "occasional"  (REQUIRED for hasDescriptor)

Example:
[
  {{"source": "mass transport deposit", "source_type": "SeismicObject", "relation": "hasDescriptor", "target": "chaotic", "target_type": "Descriptor", "salience": "typical"}},
  {{"source": "turbidite", "source_type": "SeismicObject", "relation": "hasDescriptor", "target": "massive", "target_type": "Descriptor", "salience": "occasional"}}
]

If no valid triples can be extracted, respond with: []
"""

CAUSAL_PROMPT = """You are a geological knowledge extraction system.

Given the following text excerpt from a scientific paper about mass-transport deposits (MTDs):

---
{chunk_text}
---

Question: {query}

Extract CAUSAL and PROCESS relationships as JSON triples.

CRITICAL RULES:
1. Only extract relationships EXPLICITLY stated in this text
2. Use precise relations: "triggers" for initiating events, "causes" for direct causation,
   "controls" for modulating factors, "affects" for general influence
3. Do NOT infer causation from co-occurrence alone
4. Use ONLY these relations: {relations}
5. Use ONLY these entity types: {types}

Respond with a JSON array. Each object must have:
  "source", "source_type", "relation", "target", "target_type"

If no valid triples can be extracted, respond with: []
"""

CONTEXT_PROMPT = """You are a geological knowledge extraction system.

Given the following text excerpt from a scientific paper about mass-transport deposits (MTDs):

---
{chunk_text}
---

Question: {query}

Extract SPATIAL and STRATIGRAPHIC relationships as JSON triples.

CRITICAL RULES:
1. Only extract relationships EXPLICITLY stated in this text
2. Use precise relations: "occursIn" for spatial location, "overlies"/"underlies" for stratigraphy,
   "partOf" for compositional hierarchy
3. Use ONLY these relations: {relations}
4. Use ONLY these entity types: {types}

Respond with a JSON array. Each object must have:
  "source", "source_type", "relation", "target", "target_type"

If no valid triples can be extracted, respond with: []
"""

# NOTE: all literal braces are doubled {{ }} — REQUIRED to avoid Python .format() KeyError
PROFILE_PROMPT = """You are a geological knowledge extraction system.

Given the following text excerpt from a scientific paper about mass-transport deposits:

---
{chunk_text}
---

For the geological object "{focus_object}", extract its COMPLETE descriptor profile from this text.

List ALL seismic/geological descriptors mentioned for this object, with their salience:
- "typical": diagnostic, almost always present
- "common": frequently present
- "occasional": sometimes present, context-dependent

Respond with a JSON object:
{{
  "object": "{focus_object}",
  "object_type": "SeismicObject",
  "descriptors": [
    {{"descriptor": "chaotic", "salience": "typical", "evidence": "quoted text fragment"}},
    {{"descriptor": "transparent", "salience": "common", "evidence": "quoted text fragment"}}
  ]
}}

Only include descriptors EXPLICITLY mentioned in the text for this specific object.
If no descriptors are found, respond with: {{"object": "{focus_object}", "descriptors": []}}
"""

PROMPTS = {
    "descriptor": DESCRIPTOR_PROMPT,
    "causal":     CAUSAL_PROMPT,
    "context":    CONTEXT_PROMPT,
    "profile":    PROFILE_PROMPT,
}


def get_prompt(strategy: str, **kwargs: Any) -> str:
    template = PROMPTS.get(strategy, DESCRIPTOR_PROMPT)
    return template.format(**kwargs)


def parse_response(response: str, strategy: str) -> list[dict]:
    """Parse LLM response into list of triple dicts."""
    text = (response or "").strip()

    # Try JSON array first
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            items = json.loads(m.group())
            if isinstance(items, list):
                return [x for x in items if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass

    # Profile strategy: JSON object → hasDescriptor triples
    if strategy == "profile":
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group())
                if isinstance(obj, dict) and isinstance(obj.get("descriptors"), list):
                    return [
                        {
                            "source":       obj.get("object", ""),
                            "source_type":  obj.get("object_type", "SeismicObject"),
                            "relation":     "hasDescriptor",
                            "target":       d.get("descriptor", ""),
                            "target_type":  "Descriptor",
                            "salience":     d.get("salience", "common"),
                            "_evidence":    d.get("evidence", ""),
                        }
                        for d in obj["descriptors"]
                        if isinstance(d, dict)
                    ]
            except json.JSONDecodeError:
                pass

    return []


def load_bm25(index_dir: str):
    """Load BM25 index from chunks.jsonl and return retrieve(query, top_n) function."""
    from rank_bm25 import BM25Okapi

    chunks_path = Path(index_dir) / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks.jsonl in {index_dir}")

    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    corpus = [(c.get("text", "")).lower().split() for c in chunks]
    bm25 = BM25Okapi(corpus)
    print(f"  BM25 index loaded: {len(chunks)} chunks")

    def retrieve(query: str, top_n: int = 25) -> list[dict]:
        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)
        top_idx = scores.argsort()[-top_n:][::-1]
        return [
            {
                "chunk_id":    chunks[i].get("chunk_id", f"chunk_{i}"),
                "text":        chunks[i].get("text", ""),
                "score":       float(scores[i]),
                "source_file": chunks[i].get("source_file", "unknown"),
            }
            for i in top_idx
        ]

    return retrieve


def concat_chunks(chunks: list[dict], max_chars: int = 2800) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        txt = (c.get("text", "")).strip()
        if txt:
            parts.append(f"[CHUNK {i} | {c.get('source_file','?')}]\n{txt}\n")
    return "\n---\n".join(parts)[:max_chars]


def main():
    parser = argparse.ArgumentParser(description="RAG triple extraction")
    parser.add_argument("--index-dir",  required=True)
    parser.add_argument("--schema",     required=True)
    parser.add_argument("--queries",    required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--model",      default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--backend",    default="hf", choices=["hf", "ollama"])
    parser.add_argument("--top-k",      type=int,   default=3)
    parser.add_argument("--bm25-topn",  type=int,   default=25)
    parser.add_argument("--max-chars",  type=int,   default=2800)
    parser.add_argument("--min-bm25",   type=float, default=0.0,
                        help="Default BM25 gating threshold")
    parser.add_argument("--min-bm25-desc",    type=float, default=None)
    parser.add_argument("--min-bm25-causal",  type=float, default=None)
    parser.add_argument("--min-bm25-context", type=float, default=None)
    parser.add_argument("--min-bm25-profile", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens",  type=int,   default=1024)
    parser.add_argument("--max-queries", type=int,   default=None)
    args = parser.parse_args()

    thresholds = {
        "default":    args.min_bm25,
        "descriptor": args.min_bm25_desc,
        "causal":     args.min_bm25_causal,
        "context":    args.min_bm25_context,
        "profile":    args.min_bm25_profile,
    }

    schema = json.loads(Path(args.schema).read_text())
    relations = [r["name"] if isinstance(r, dict) else r for r in schema.get("relations", [])]
    types     = [t["name"] if isinstance(t, dict) else t for t in schema.get("types", [])]

    queries = []
    with open(args.queries, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    if args.max_queries:
        queries = queries[:args.max_queries]
    print(f"  Loaded {len(queries)} queries")

    # Load LLM
    if args.backend == "hf":
        from pipeline.rag.llm_hf import make_hf_fn
        _gen = make_hf_fn(args.model, max_new_tokens=args.max_tokens)
        generate = lambda prompt: _gen("", prompt, temperature=args.temperature)
    else:
        import requests
        def generate(prompt: str) -> str:
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": args.model, "prompt": prompt,
                      "options": {"temperature": args.temperature}, "stream": False},
                timeout=600,
            )
            return r.json().get("response", "")

    retrieve = load_bm25(args.index_dir)
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)
    t0 = time.time()

    with open(outpath, "w", encoding="utf-8") as fout:
        for qi, q in enumerate(queries):
            query_text = (q.get("query", q.get("text", "")) or "").strip()
            if not query_text:
                continue

            strategy = q.get("strategy", "descriptor")
            focus    = q.get("focus", "")

            candidates = retrieve(query_text, top_n=args.bm25_topn)
            if not candidates:
                stats["skipped_no_chunks"] += 1
                continue

            best_score = candidates[0]["score"]
            thr = thresholds.get(strategy) or thresholds["default"]
            if thr > 0 and best_score < thr:
                stats["skipped_low_bm25"] += 1
                continue

            selected = candidates[:args.top_k]
            context  = concat_chunks(selected, max_chars=args.max_chars)

            prompt_kwargs = dict(
                chunk_text=context,
                query=query_text,
                relations=", ".join(relations),
                types=", ".join(types),
            )
            if strategy == "profile":
                focus_obj = focus.split("→")[0].strip() if "→" in focus else focus.strip()
                prompt_kwargs["focus_object"] = focus_obj or "mass transport deposit"

            try:
                prompt   = get_prompt(strategy, **prompt_kwargs)
                response = generate(prompt)
                triples  = parse_response(response, strategy)
            except Exception as e:
                print(f"  ERROR at query {qi+1}: {e}")
                stats["errors"] += 1
                continue

            for t in triples:
                if not isinstance(t, dict):
                    continue
                # Normalize relation
                raw_rel  = t.get("relation", "")
                norm_rel = normalize_relation(raw_rel)
                if norm_rel not in ALLOWED_RELATIONS:
                    continue
                t["relation"] = norm_rel
                if norm_rel == "hasDescriptor" and "salience" not in t:
                    t["salience"] = "common"

                t["_provenance"] = {
                    "query":           query_text,
                    "strategy":        strategy,
                    "focus":           focus,
                    "model":           args.model,
                    "source_files":    [c["source_file"] for c in selected],
                    "best_bm25":       best_score,
                    "bm25_threshold":  thr,
                    "context_preview": context[:1500],
                    "selected_chunks": [c["chunk_id"] for c in selected],
                }

                fout.write(json.dumps(t, ensure_ascii=False) + "\n")
                stats["total_triples"] += 1
                stats[f"strategy_{strategy}"] += 1

            if (qi + 1) % 25 == 0:
                elapsed = time.time() - t0
                print(f"  [{qi+1}/{len(queries)}] triples={stats['total_triples']} "
                      f"({(qi+1)/elapsed:.2f} q/s)")

    elapsed = time.time() - t0
    print(f"\n  Done: {stats['total_triples']} triples in {elapsed:.0f}s")
    print(f"  Model: {args.model}")
    print(f"  Skipped (low BM25): {stats['skipped_low_bm25']}")
    print(f"  Errors: {stats['errors']}")

    stats_path = outpath.parent / "raw_triples_stats.json"
    Path(stats_path).write_text(json.dumps(dict(stats), indent=2))
    print(f"  Stats: {stats_path}")


if __name__ == "__main__":
    main()