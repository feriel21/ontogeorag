#!/usr/bin/env python3
"""
EXP-B: No-RAG Ablation — Does BM25 retrieval actually help?
=============================================================
Runs triple extraction WITHOUT BM25 context (LLM sees only the query +
few-shot examples). Then runs the same verifier on the output.

Compares to Run 7 (WITH RAG):
  - NOT_SUPPORTED rate WITH RAG    (from step7/verification_stats_v5.json)
  - NOT_SUPPORTED rate WITHOUT RAG (computed here)

This isolates whether the hallucination reduction is genuinely due to
RAG grounding or just threshold-based filtering artefacts.

Runtime: ~6 hours on CPU (200 triples × ~2 queries each, no GPU)
         Submit as SLURM job on convergence partition.

Usage:
  python exp_b_no_rag_ablation.py \
      --queries    output/step7/raw_triples_v7.jsonl \
      --lexicon    configs/lexicon.json \
      --schema     configs/schema_step1.json \
      --out        output/exp_b/ \
      --model      Qwen/Qwen2.5-7B-Instruct \
      --backend    hf \
      --n-triples  100

SLURM submission:
  sbatch exp_b_slurm.sh
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional


# ── Allowed relations (same as main pipeline) ─────────────────────────

ALLOWED_RELATIONS = {
    "hasDescriptor", "occursIn", "formedBy", "partOf",
    "triggers", "causes", "controls", "affects",
    "overlies", "underlies", "indicates", "evidences", "relatedTo",
}

RELATION_MAP = {
    "hasdescriptor": "hasDescriptor", "occursin": "occursIn",
    "formedby": "formedBy", "partof": "partOf",
    "triggeredby": "triggers", "triggered_by": "triggers",
    "overlays": "overlies", "relatedto": "relatedTo",
    "related_to": "relatedTo", "related to": "relatedTo",
}


# ── Few-shot examples (identical to main pipeline) ────────────────────

FEW_SHOT_EXTRACTION = """You are a geoscience knowledge graph expert.
Extract a knowledge graph triple (subject, relation, object) from the query.

Allowed relations: hasDescriptor, occursIn, formedBy, partOf, triggers,
causes, controls, affects, overlies, underlies, indicates, evidences, relatedTo

Return ONLY valid JSON:
{{"subject": "...", "relation": "...", "object": "..."}}

Examples:
Q: What seismic character does a mass transport deposit show?
A: {{"subject": "mass transport deposit", "relation": "hasDescriptor", "object": "chaotic"}}

Q: Where do turbidites typically occur?
A: {{"subject": "turbidite", "relation": "occursIn", "object": "basin floor"}}

Q: What causes slope failure?
A: {{"subject": "earthquake", "relation": "triggers", "object": "slope failure"}}

Now extract from this query (NO additional context — use only your training knowledge):
Q: {query}
A:"""


FEW_SHOT_VERIFICATION = """You are a strict geological fact-checker.
Assess whether the triple is supported by geological knowledge.

Triple: {subject} --{relation}--> {object}

Respond with ONLY one of:
  STRONG_SUPPORT  — well-established geological fact
  WEAK_SUPPORT    — plausible but less certain
  NOT_SUPPORTED   — not supported or contradicts geology

Verdict:"""


# ── LLM backends ──────────────────────────────────────────────────────

def load_hf_model(model_name: str):
    """Load model via HuggingFace transformers pipeline."""
    try:
        from transformers import pipeline as hf_pipeline
        print(f"[EXP-B] Loading HF model: {model_name}")
        pipe = hf_pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False,
        )
        print("[EXP-B] Model loaded.")
        return pipe
    except Exception as e:
        print(f"[EXP-B] HF load failed: {e}", file=sys.stderr)
        sys.exit(1)


def call_hf(pipe, prompt: str, max_tokens: int = 128) -> str:
    result = pipe(prompt, max_new_tokens=max_tokens, return_full_text=False)
    return result[0]["generated_text"].strip()


def call_ollama(url: str, model: str, prompt: str) -> str:
    import urllib.request
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        f"{url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data.get("response", "").strip()


# ── Extraction (NO RAG — no chunk context) ────────────────────────────

def extract_no_rag(llm_fn, query: str) -> Optional[Dict]:
    """Call LLM with query only, no retrieved chunk."""
    prompt = FEW_SHOT_EXTRACTION.format(query=query)
    try:
        response = llm_fn(prompt)
        # Parse JSON from response
        match = re.search(r'\{[^}]+\}', response)
        if match:
            triple = json.loads(match.group())
            # Normalize relation
            rel = triple.get("relation", "")
            rel_norm = RELATION_MAP.get(rel.lower(), rel)
            if rel_norm not in ALLOWED_RELATIONS:
                return None
            triple["relation"] = rel_norm
            return triple
    except Exception as e:
        pass
    return None


# ── Verification (same as main pipeline) ──────────────────────────────

def verify_triple(llm_fn, subject: str, relation: str, object_: str) -> str:
    """Verify a triple WITHOUT providing any chunk context (pure parametric)."""
    prompt = FEW_SHOT_VERIFICATION.format(
        subject=subject, relation=relation, object=object_
    )
    try:
        response = llm_fn(prompt).upper()
        for verdict in ["STRONG_SUPPORT", "WEAK_SUPPORT", "NOT_SUPPORTED"]:
            if verdict in response:
                return verdict
        return "UNPARSEABLE"
    except Exception:
        return "BACKEND_ERROR"


# ── Main ──────────────────────────────────────────────────────────────

def run_exp_b(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the Run-7 raw triples (these contain the original queries)
    print(f"[EXP-B] Loading triples from {args.queries}")
    triples = []
    with open(args.queries) as f:
        for line in f:
            line = line.strip()
            if line:
                triples.append(json.loads(line))

    # Use subset if requested
    if args.n_triples and args.n_triples < len(triples):
        # Stratified sample: take proportional from each relation type
        from collections import defaultdict
        by_relation = defaultdict(list)
        for t in triples:
            by_relation[t.get("relation", "unknown")].append(t)
        selected = []
        per_rel = max(1, args.n_triples // len(by_relation))
        for rel, rel_triples in by_relation.items():
            selected.extend(rel_triples[:per_rel])
        triples = selected[:args.n_triples]
        print(f"[EXP-B] Using stratified sample of {len(triples)} triples")
    else:
        print(f"[EXP-B] Processing all {len(triples)} triples")

    # Load LLM
    if args.backend == "hf":
        pipe = load_hf_model(args.model)
        llm_fn = lambda p: call_hf(pipe, p)
    else:
        llm_fn = lambda p: call_ollama(args.ollama_url, args.model, p)
        print(f"[EXP-B] Using Ollama at {args.ollama_url} model={args.model}")

    # --- Phase 1: Extract triples WITHOUT RAG ---
    print("\n[EXP-B] Phase 1: Extraction WITHOUT RAG context...")
    no_rag_triples = []
    failed_extraction = 0
    t0 = time.time()

    for i, orig in enumerate(triples):
        query = orig.get("query", f"Describe {orig.get('subject','')} {orig.get('relation','')} {orig.get('object','')}")
        extracted = extract_no_rag(llm_fn, query)
        if extracted:
            extracted["_original_query"] = query
            extracted["_orig_subject"] = orig.get("subject", "")
            extracted["_orig_relation"] = orig.get("relation", "")
            extracted["_orig_object"] = orig.get("object", "")
            no_rag_triples.append(extracted)
        else:
            failed_extraction += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            eta = (len(triples) - i - 1) / rate
            print(f"  [{i+1}/{len(triples)}] extracted={len(no_rag_triples)} "
                  f"failed={failed_extraction} rate={rate:.2f}/s ETA={eta/60:.1f}min")

    print(f"[EXP-B] Phase 1 done: {len(no_rag_triples)} triples extracted, "
          f"{failed_extraction} failed")

    # Save raw extraction
    raw_out = out_dir / "exp_b_no_rag_raw.jsonl"
    with open(raw_out, "w") as f:
        for t in no_rag_triples:
            f.write(json.dumps(t) + "\n")

    # --- Phase 2: Verify WITHOUT chunk context ---
    print("\n[EXP-B] Phase 2: Verification WITHOUT chunk context...")
    verdicts = {"STRONG_SUPPORT": 0, "WEAK_SUPPORT": 0, "NOT_SUPPORTED": 0,
                "UNPARSEABLE": 0, "BACKEND_ERROR": 0}
    verified = []
    t0 = time.time()

    for i, t in enumerate(no_rag_triples):
        verdict = verify_triple(llm_fn, t["subject"], t["relation"], t["object"])
        t["_verdict_no_rag"] = verdict
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        verified.append(t)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i+1)/elapsed
            decided = verdicts["STRONG_SUPPORT"] + verdicts["WEAK_SUPPORT"] + verdicts["NOT_SUPPORTED"]
            ns_rate = verdicts["NOT_SUPPORTED"] / max(decided, 1)
            print(f"  [{i+1}/{len(no_rag_triples)}] NOT_SUPPORTED={ns_rate:.1%} rate={rate:.2f}/s")

    # --- Compute stats ---
    decided = verdicts["STRONG_SUPPORT"] + verdicts["WEAK_SUPPORT"] + verdicts["NOT_SUPPORTED"]
    ns_rate_no_rag = verdicts["NOT_SUPPORTED"] / max(decided, 1)

    # Load Run7 WITH-RAG stats for comparison
    run7_stats_path = os.path.expanduser("output/step7/verification_stats_v5.json")
    run7_ns_rate = None
    if os.path.exists(run7_stats_path):
        with open(run7_stats_path) as f:
            run7 = json.load(f)
        run7_ns_rate = run7.get("hallucination_rate_decided", None)

    print("\n" + "="*60)
    print("EXP-B RESULTS — No-RAG Ablation")
    print("="*60)
    print(f"  Triples evaluated         : {len(no_rag_triples)}")
    print(f"  Decided (S+W+NS)          : {decided}")
    print()
    print(f"  WITHOUT RAG — NOT_SUPPORTED rate : {ns_rate_no_rag:.1%}  "
          f"({verdicts['NOT_SUPPORTED']}/{decided})")
    if run7_ns_rate is not None:
        print(f"  WITH    RAG (Run 7)       : {run7_ns_rate:.1%}")
        delta = ns_rate_no_rag - run7_ns_rate
        print(f"  Delta (RAG improvement)   : {delta:+.1%}")
        if delta > 0.05:
            print("  → RAG genuinely reduces hallucinations (not just filtering)")
        elif delta < -0.05:
            print("  → WARNING: RAG may be introducing noise")
        else:
            print("  → Marginal RAG effect — improvement mainly from thresholding")
    print()
    print(f"  Verdict breakdown (no-RAG):")
    for k, v in verdicts.items():
        print(f"    {k:20s}: {v}")
    print("="*60)

    # --- Save outputs ---
    verified_out = out_dir / "exp_b_no_rag_verified.jsonl"
    with open(verified_out, "w") as f:
        for t in verified:
            f.write(json.dumps(t) + "\n")

    stats = {
        "description": "EXP-B No-RAG Ablation",
        "model": args.model,
        "n_triples_input": len(triples),
        "n_extracted_no_rag": len(no_rag_triples),
        "extraction_failure_rate": failed_extraction / max(len(triples), 1),
        "verdicts": verdicts,
        "decided_total": decided,
        "not_supported_rate_no_rag": round(ns_rate_no_rag, 4),
        "not_supported_rate_with_rag_run7": round(run7_ns_rate, 4) if run7_ns_rate else None,
        "rag_improvement_delta": round(ns_rate_no_rag - run7_ns_rate, 4) if run7_ns_rate else None,
    }
    stats_out = out_dir / "exp_b_stats.json"
    with open(stats_out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[EXP-B] Stats saved → {stats_out}")
    print(f"[EXP-B] Details  → {verified_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries",  default=os.path.expanduser(
        "output/step7/raw_triples_v7.jsonl"),
        help="Run-7 raw triples JSONL (contains original queries)")
    parser.add_argument("--lexicon",  default=os.path.expanduser(
        "configs/lexicon.json"))
    parser.add_argument("--schema",   default=os.path.expanduser(
        "configs/schema_step1.json"))
    parser.add_argument("--out",      default=os.path.expanduser(
        "output/exp_b/"))
    parser.add_argument("--model",    default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--backend",  default="hf", choices=["hf", "ollama"])
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--n-triples", type=int, default=100,
        help="Number of triples to process (default 100 for speed; set 0 for all)")
    args = parser.parse_args()

    run_exp_b(args)