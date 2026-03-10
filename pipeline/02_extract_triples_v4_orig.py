#!/usr/bin/env python3
"""
src_/02_rag_extract_triples_v4.py — Run5-style extraction (salience + profile) with:
  - TOP-K chunk context (multi-chunk concatenation)
  - GATING on low BM25 score (skip weak retrieval => fewer hallucinations)
  - OPTIONAL rerank (BM25 top-N -> cosine similarity embeddings) if sentence-transformers installed
  - JSONL-safe loading for chunks.jsonl
  - FIXES:
      (1) Profile prompt braces escaped => no KeyError '"object"'
      (2) Stats counters (attempted / processed / total_queries) are consistent

Usage example:
  python -u src_/02_rag_extract_triples_v4.py \
      --index-dir output/step1 \
      --schema schema_seed_output/schema_step1.json \
      --queries schema_seed_output/queries_v4.jsonl \
      --output output/step5/raw_triples_v5.jsonl \
      --backend hf --model Qwen/Qwen2.5-7B-Instruct \
      --top-k 3 --bm25-topn 25 --max-chars 2800 \
      --min-bm25-desc 15.2 --min-bm25-profile 15.2 \
      --min-bm25-causal 16.8 --min-bm25-context 16.8

SLURM --wrap (IMPORTANT: keep everything inside the quotes):
  sbatch --partition=convergence --gres=gpu:a100_3g.40gb:1 --mem=32G --time=02:00:00 \
    --wrap="cd ~/kg_test && . venv/bin/activate && python -u src_/02_rag_extract_triples_v4.py \
      --index-dir output/step1 \
      --schema schema_seed_output/schema_step1.json \
      --queries schema_seed_output/queries_v4.jsonl \
      --output output/step5/raw_triples_v5.jsonl \
      --backend hf --model Qwen/Qwen2.5-7B-Instruct \
      --top-k 3 --bm25-topn 25 --max-chars 2800 \
      --min-bm25-desc 15.2 --min-bm25-profile 15.2 \
      --min-bm25-causal 16.8 --min-bm25-context 16.8"
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates (Run5-style: salience + profile)
# NOTE: any literal JSON braces inside a .format() template must be doubled: {{ }}
# ──────────────────────────────────────────────────────────────────────────────

DESCRIPTOR_PROMPT_V5 = """You are a geological knowledge extraction system.

Given the following text excerpt from a scientific paper about mass-transport deposits (MTDs):

---
{chunk_text}
---

Question: {query}

Extract geological relationships as JSON triples. For DESCRIPTOR relationships, you MUST indicate the salience level:
- "typical": This descriptor is DIAGNOSTIC — almost always present, experts would list it first
- "common": This descriptor is FREQUENT — often present but not defining
- "occasional": This descriptor is POSSIBLE — occurs in some cases but not characteristic

CRITICAL RULES:
1. Only extract relationships that are EXPLICITLY stated or DIRECTLY implied by this text
2. Do NOT assign descriptors that are merely possible — they must be described in the text
3. If the text says an object "can be" or "sometimes shows" a descriptor, mark it as "occasional"
4. If the text says an object "is characterized by" or "typically shows", mark it as "typical"
5. Use ONLY these relations: {relations}
6. Use ONLY these entity types: {types}

Respond with a JSON array of objects. Each object must have:
  "source": entity name (lowercase)
  "source_type": one of {types}
  "relation": one of {relations}
  "target": entity name (lowercase)
  "target_type": one of {types}
  "salience": "typical" | "common" | "occasional" (REQUIRED for hasDescriptor, optional otherwise)

Example:
[
  {{"source": "mass transport deposit", "source_type": "SeismicObject", "relation": "hasDescriptor", "target": "chaotic", "target_type": "Descriptor", "salience": "typical"}},
  {{"source": "turbidite", "source_type": "SeismicObject", "relation": "hasDescriptor", "target": "massive", "target_type": "Descriptor", "salience": "occasional"}}
]

If no valid triples can be extracted, respond with: []
"""

CAUSAL_PROMPT_V5 = """You are a geological knowledge extraction system.

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
3. Do NOT invent causal links — if the text only describes co-occurrence, do not assume causation
4. Use ONLY these relations: {relations}
5. Use ONLY these entity types: {types}

Respond with a JSON array. Each object must have:
  "source", "source_type", "relation", "target", "target_type"

If no valid triples can be extracted, respond with: []
"""

CONTEXT_PROMPT_V5 = """You are a geological knowledge extraction system.

Given the following text excerpt from a scientific paper about mass-transport deposits (MTDs):

---
{chunk_text}
---

Question: {query}

Extract SPATIAL and STRATIGRAPHIC relationships as JSON triples.

CRITICAL RULES:
1. Only extract relationships EXPLICITLY stated in this text
2. Use precise relations: "occursIn" for spatial location, "overlies"/"underlies" for stratigraphy,
   "partOf" for compositional hierarchy, "contains" for inclusion
3. Use ONLY these relations: {relations}
4. Use ONLY these entity types: {types}

Respond with a JSON array. Each object must have:
  "source", "source_type", "relation", "target", "target_type"

If no valid triples can be extracted, respond with: []
"""

# IMPORTANT: all literal braces are doubled {{ }} so .format() won't treat them as placeholders.
PROFILE_PROMPT_V5 = """You are a geological knowledge extraction system.

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


def get_prompt(strategy: str, **kwargs: Any) -> str:
    if strategy == "descriptor":
        return DESCRIPTOR_PROMPT_V5.format(**kwargs)
    if strategy == "causal":
        return CAUSAL_PROMPT_V5.format(**kwargs)
    if strategy == "context":
        return CONTEXT_PROMPT_V5.format(**kwargs)
    if strategy == "profile":
        return PROFILE_PROMPT_V5.format(**kwargs)
    raise ValueError(f"Unknown strategy: {strategy}")


def parse_llm_response(response_text: str, strategy: str = "descriptor") -> List[Dict[str, Any]]:
    """
    Parse model output into list of triples.
    - descriptor/causal/context: JSON array
    - profile: JSON object -> converted to hasDescriptor triples
    """
    import re

    text = (response_text or "").strip()

    # Try array first
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            items = json.loads(m.group())
            if isinstance(items, list):
                return [x for x in items if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass

    # Profile object
    if strategy == "profile":
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group())
                if isinstance(obj, dict) and isinstance(obj.get("descriptors", None), list):
                    triples: List[Dict[str, Any]] = []
                    for d in obj["descriptors"]:
                        if not isinstance(d, dict):
                            continue
                        triples.append({
                            "source": obj.get("object", ""),
                            "source_type": obj.get("object_type", "SeismicObject"),
                            "relation": "hasDescriptor",
                            "target": d.get("descriptor", ""),
                            "target_type": "Descriptor",
                            "salience": d.get("salience", "common"),
                            "_evidence_quote": d.get("evidence", ""),
                        })
                    return triples
            except json.JSONDecodeError:
                pass

    return []


# ──────────────────────────────────────────────────────────────────────────────
# Backend loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_backend(backend: str, model_name: str):
    if backend == "hf":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"  [HF] Loading model {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Prefer bf16 on A100; fallback to fp16/32
        if torch.cuda.is_available():
            try:
                major, _minor = torch.cuda.get_device_capability(0)
            except Exception:
                major = 8
            dtype = torch.bfloat16 if major >= 8 else torch.float16
            device_map = "auto"
        else:
            dtype = torch.float32
            device_map = "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,          # avoids torch_dtype deprecation warning
            device_map=device_map,
            trust_remote_code=True,
        )
        print(f"  [HF] Model loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}.")

        def generate(prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            do_sample = temperature > 0.0
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=0.9 if do_sample else None,
                )
            return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return generate

    if backend == "ollama":
        import requests

        def generate(prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                    "stream": False,
                },
                timeout=600,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")

        return generate

    raise ValueError(f"Unknown backend: {backend}")


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval: BM25 + optional rerank
# ──────────────────────────────────────────────────────────────────────────────

def _try_load_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception:
        return None


def _cosine_rerank(query: str, candidates: List[Dict[str, Any]], embedder, top_k: int) -> List[Dict[str, Any]]:
    if embedder is None or not candidates:
        return candidates[:top_k]

    texts = [(c.get("text") or "")[:1500] for c in candidates]
    q_emb = embedder.encode([query], normalize_embeddings=True)
    t_emb = embedder.encode(texts, normalize_embeddings=True)

    sims = (t_emb @ q_emb[0]).tolist()
    for c, s in zip(candidates, sims):
        c["rerank_score"] = float(s)

    return sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)[:top_k]


def _concat_chunks(chunks: List[Dict[str, Any]], max_chars: int) -> str:
    parts: List[str] = []
    for i, c in enumerate(chunks, start=1):
        src = c.get("source_file", "unknown")
        cid = c.get("chunk_id", f"chunk_{i}")
        txt = (c.get("text") or "").strip()
        if not txt:
            continue
        parts.append(f"[CHUNK {i} | {src} | {cid}]\n{txt}\n")
    joined = "\n---\n".join(parts)
    return joined[:max_chars]


def load_bm25_index(index_dir: str):
    """
    Loads chunks.jsonl and builds an in-memory BM25.
    Returns retrieve(query, top_n) -> list of candidates with BM25 score.
    """
    from rank_bm25 import BM25Okapi

    index_dir = Path(index_dir)
    chunks_path = index_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    chunks: List[Dict[str, Any]] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))

    corpus = [(c.get("text") or "").lower().split() for c in chunks]
    bm25 = BM25Okapi(corpus)
    print(f"  Loaded BM25 index: {len(chunks)} chunks ({chunks_path.name})")

    def retrieve(query: str, top_n: int = 20) -> List[Dict[str, Any]]:
        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)
        top_idx = scores.argsort()[-top_n:][::-1]
        out: List[Dict[str, Any]] = []
        for idx in top_idx:
            out.append({
                "chunk_id": chunks[idx].get("chunk_id", f"chunk_{idx}"),
                "text": chunks[idx].get("text", ""),
                "score": float(scores[idx]),
                "source_file": chunks[idx].get("source_file", "unknown"),
            })
        return out

    return retrieve


def _percentiles(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {}
    xs_sorted = sorted(xs)
    n = len(xs_sorted)

    def pct(p: float) -> float:
        if n == 1:
            return float(xs_sorted[0])
        k = (n - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, n - 1)
        if f == c:
            return float(xs_sorted[f])
        d = k - f
        return float(xs_sorted[f] * (1 - d) + xs_sorted[c] * d)

    return {"p10": pct(10), "p25": pct(25), "p50": pct(50), "p75": pct(75), "p90": pct(90)}


def _threshold_for_strategy(strategy: str, thresholds: Dict[str, float]) -> float:
    # Strategy names in your queries: descriptor/causal/context/profile
    if strategy in thresholds and thresholds[strategy] is not None:
        return float(thresholds[strategy])
    return float(thresholds.get("default", 0.0))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG extraction (Run5 style) with top-k, gating, rerank")
    parser.add_argument("--index-dir", required=True, help="BM25 index directory (expects chunks.jsonl)")
    parser.add_argument("--schema", required=True, help="Schema JSON file")
    parser.add_argument("--queries", required=True, help="Queries JSONL file")
    parser.add_argument("--output", required=True, help="Output raw triples JSONL")
    parser.add_argument("--backend", default="hf", choices=["hf", "ollama"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")

    # Retrieval controls
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--bm25-topn", type=int, default=25)
    parser.add_argument("--max-chars", type=int, default=2800)

    # Global threshold (fallback)
    parser.add_argument("--min-bm25", type=float, default=0.0, help="Default gating threshold if per-strategy not set")

    # Per-strategy thresholds (what you asked for)
    parser.add_argument("--min-bm25-desc", type=float, default=None)
    parser.add_argument("--min-bm25-profile", type=float, default=None)
    parser.add_argument("--min-bm25-causal", type=float, default=None)
    parser.add_argument("--min-bm25-context", type=float, default=None)

    # Optional rerank
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-model", default="sentence-transformers/all-MiniLM-L6-v2")

    # Generation
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-queries", type=int, default=None)

    args = parser.parse_args()

    thresholds = {
        "default": float(args.min_bm25),
        "descriptor": args.min_bm25_desc,
        "profile": args.min_bm25_profile,
        "causal": args.min_bm25_causal,
        "context": args.min_bm25_context,
    }

    # Load schema
    schema = json.load(open(args.schema, "r", encoding="utf-8"))
    relations = schema.get("relations", [])
    types = schema.get("types", [])
    relation_names = [r["name"] if isinstance(r, dict) else r for r in relations]
    type_names = [t["name"] if isinstance(t, dict) else t for t in types]

    # Load queries
    queries: List[Dict[str, Any]] = []
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    if args.max_queries is not None:
        queries = queries[:args.max_queries]

    # Backend + retrieval
    generate = load_backend(args.backend, args.model)
    retrieve = load_bm25_index(args.index_dir)

    # Optional reranker
    embedder = _try_load_embedder(args.rerank_model) if args.rerank else None
    if args.rerank and embedder is None:
        print("  [WARN] --rerank enabled but embedder could not be loaded. Continuing without rerank.")

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {
        "attempted_non_empty": 0,          # non-empty queries seen
        "processed_passed_gating": 0,      # queries that passed gating & were sent to LLM
        "total_triples": 0,
        "by_strategy": defaultdict(int),
        "by_salience": defaultdict(int),
        "errors": 0,
        "skipped_no_chunks": 0,
        "skipped_low_bm25": 0,
        "skipped_by_strategy": defaultdict(int),
        "processed_by_strategy": defaultdict(int),
        "used_rerank": bool(args.rerank),
        "rerank_available": bool(embedder is not None) if args.rerank else False,
        "thresholds": {
            "descriptor": thresholds["descriptor"] if thresholds["descriptor"] is not None else thresholds["default"],
            "profile": thresholds["profile"] if thresholds["profile"] is not None else thresholds["default"],
            "causal": thresholds["causal"] if thresholds["causal"] is not None else thresholds["default"],
            "context": thresholds["context"] if thresholds["context"] is not None else thresholds["default"],
            "default": thresholds["default"],
        },
        "top_k": args.top_k,
        "bm25_topn": args.bm25_topn,
        "max_chars": args.max_chars,
        "temperature": args.temperature,
        "model": args.model,
        "backend": args.backend,
        "bm25_best_scores": [],
    }

    print(f"\n  Processing {len(queries)} queries...")
    t0 = time.time()

    with open(outpath, "w", encoding="utf-8") as fout:
        for qi, q in enumerate(queries):
            query_text = (q.get("query", q.get("text", "")) or "").strip()
            if not query_text:
                continue

            stats["attempted_non_empty"] += 1
            strategy = q.get("strategy", "descriptor") or "descriptor"
            focus = q.get("focus", "")

            # Retrieval
            candidates = retrieve(query_text, top_n=args.bm25_topn)
            if not candidates:
                stats["skipped_no_chunks"] += 1
                stats["skipped_by_strategy"][strategy] += 1
                continue

            best_bm25 = float(candidates[0].get("score", 0.0))
            stats["bm25_best_scores"].append(best_bm25)

            # Per-strategy gating
            thr = _threshold_for_strategy(strategy, thresholds)
            if thr > 0.0 and best_bm25 < thr:
                stats["skipped_low_bm25"] += 1
                stats["skipped_by_strategy"][strategy] += 1
                continue

            # Rerank / select top-k
            if args.rerank and embedder is not None:
                selected = _cosine_rerank(query_text, candidates, embedder, top_k=args.top_k)
            else:
                selected = candidates[:args.top_k]

            if not selected:
                stats["skipped_no_chunks"] += 1
                stats["skipped_by_strategy"][strategy] += 1
                continue

            stats["processed_passed_gating"] += 1
            stats["processed_by_strategy"][strategy] += 1

            # Build multi-chunk context
            context_text = _concat_chunks(selected, max_chars=args.max_chars)
            best_chunk = selected[0]

            prompt_kwargs = {
                "chunk_text": context_text,
                "query": query_text,
                "relations": ", ".join(relation_names),
                "types": ", ".join(type_names),
            }

            if strategy == "profile":
                focus_obj = focus.split("→")[0].strip() if "→" in focus else focus.strip()
                prompt_kwargs["focus_object"] = focus_obj if focus_obj else "mass transport deposit"

            try:
                prompt = get_prompt(strategy, **prompt_kwargs)
                response = generate(prompt, temperature=args.temperature, max_tokens=args.max_tokens)
                triples = parse_llm_response(response, strategy=strategy)
            except Exception as e:
                print(f"  ERROR at query {qi+1}/{len(queries)}: {e}")
                stats["errors"] += 1
                continue

            # Write triples
            for t in triples:
                if not isinstance(t, dict):
                    continue

                if t.get("relation") == "hasDescriptor" and "salience" not in t:
                    t["salience"] = "common"

                t["_provenance"] = {
                    "query": query_text,
                    "strategy": strategy,
                    "focus": focus,
                    "source_files": list(dict.fromkeys([c.get("source_file", "unknown") for c in selected])),
                    "chunk_scores": [float(c.get("score", 0.0)) for c in selected],
                    "rerank_scores": [c.get("rerank_score", None) for c in selected],
                    "selected_chunk_ids": [c.get("chunk_id", "") for c in selected],
                    "best_chunk_id": best_chunk.get("chunk_id", ""),
                    "context_preview": context_text[:1500],
                    "best_bm25": best_bm25,
                    "bm25_threshold": thr,
                    "bm25_topn": args.bm25_topn,
                    "top_k": args.top_k,
                    "rerank": bool(args.rerank and embedder is not None),
                    "temperature": args.temperature,
                }

                fout.write(json.dumps(t, ensure_ascii=False) + "\n")
                stats["total_triples"] += 1
                stats["by_strategy"][strategy] += 1
                if "salience" in t:
                    stats["by_salience"][t["salience"]] += 1

            if (qi + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = (qi + 1) / elapsed if elapsed > 0 else 0.0
                print(f"  [{qi+1}/{len(queries)}] triples={stats['total_triples']} ({rate:.2f} queries/s)")

    # Finalize stats
    bm25_scores = [float(x) for x in stats["bm25_best_scores"]]
    stats["bm25_best_percentiles"] = _percentiles(bm25_scores)
    stats["by_strategy"] = dict(stats["by_strategy"])
    stats["by_salience"] = dict(stats["by_salience"])
    stats["skipped_by_strategy"] = dict(stats["skipped_by_strategy"])
    stats["processed_by_strategy"] = dict(stats["processed_by_strategy"])
    stats["wall_clock_seconds"] = time.time() - t0

    stats_path = outpath.parent / "raw_triples_v5_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n  Done: {stats['total_triples']} triples")
    print(f"  Attempted (non-empty): {stats['attempted_non_empty']}")
    print(f"  Processed (passed gating): {stats['processed_passed_gating']}")
    print(f"  By strategy: {stats['by_strategy']}")
    if stats["by_salience"]:
        print(f"  By salience: {stats['by_salience']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Skipped (low BM25): {stats['skipped_low_bm25']} | by strategy: {stats['skipped_by_strategy']}")
    print(f"  Skipped (no chunks): {stats['skipped_no_chunks']}")
    if stats.get("bm25_best_percentiles"):
        p = stats["bm25_best_percentiles"]
        print(
            f"  BM25 best score percentiles: "
            f"p10={p['p10']:.2f} p25={p['p25']:.2f} p50={p['p50']:.2f} p75={p['p75']:.2f} p90={p['p90']:.2f}"
        )
    print(f"  Output: {outpath}")
    print(f"  Stats:  {stats_path}")


if __name__ == "__main__":
    main()