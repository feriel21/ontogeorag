#!/usr/bin/env python3
"""
EXP-D: Cross-Model Verification — Does Qwen self-verify its own triples?
=========================================================================
Takes the Run-7 verified triples (verified by Qwen2.5-7B) and re-verifies
the same triples using a DIFFERENT model (Llama-3.1-8B or Qwen2.5-1.5B).

Computes:
  1. Agreement rate (%) between Qwen-7B and the cross-verifier
  2. NOT_SUPPORTED rate difference (self-bias indicator)
  3. Cohen's kappa (inter-rater reliability)
  4. Per-verdict confusion matrix

If NOT_SUPPORTED rate differs by >10pp → evidence of self-verification bias.
This directly addresses reviewer concern W4.

Runtime: ~3–4 hours on CPU for 100 triples.

Usage:
  python exp_d_cross_model_verification.py \
      --verified  ~/kg_test/output/step7/verified_triples_v7.jsonl \
      --bm25      ~/kg_test/output/step2/bm25_index.json \
      --out       ~/kg_test/output/exp_d/ \
      --model     Qwen/Qwen2.5-1.5B-Instruct \
      --backend   hf \
      --n-triples 100

SLURM submission:
  sbatch exp_d_slurm.sh
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


# ── Allowed relations ─────────────────────────────────────────────────

ALLOWED_RELATIONS = {
    "hasDescriptor", "occursIn", "formedBy", "partOf",
    "triggers", "causes", "controls", "affects",
    "overlies", "underlies", "indicates", "evidences", "relatedTo",
}

VERDICT_MAP = {"STRONG_SUPPORT": 2, "WEAK_SUPPORT": 1, "NOT_SUPPORTED": 0}
VERDICT_ORDER = ["STRONG_SUPPORT", "WEAK_SUPPORT", "NOT_SUPPORTED"]


# ── Verification prompt (identical to 02b_verify_triples_v5.py) ───────

VERIFY_PROMPT_WITH_CHUNK = """You are a strict geological fact-checker.
Your task: decide if the triple (subject, relation, object) is supported by
the provided context passage.

Triple: {subject} --{relation}--> {object}

Context:
\"\"\"
{chunk}
\"\"\"

Respond with ONLY one of:
  STRONG_SUPPORT  — the context clearly supports the triple
  WEAK_SUPPORT    — the context partially or implicitly supports it
  NOT_SUPPORTED   — the context does not support or contradicts the triple

Verdict:"""

VERIFY_PROMPT_NO_CHUNK = """You are a strict geological fact-checker.
Assess whether this triple reflects established geological knowledge.

Triple: {subject} --{relation}--> {object}

Respond with ONLY one of:
  STRONG_SUPPORT  — well-established geological fact
  WEAK_SUPPORT    — plausible but less certain
  NOT_SUPPORTED   — not supported or contradicts geology

Verdict:"""


# ── LLM loader ────────────────────────────────────────────────────────

def load_hf_model(model_name: str):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        print(f"[EXP-D] Loading cross-verifier: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", dtype=torch.float16
        )
        model.eval()
        print("[EXP-D] Cross-verifier loaded.")
        return (model, tokenizer)
    except Exception as e:
        print(f"[EXP-D] HF load failed: {e}", file=sys.stderr)
        sys.exit(1)


def call_hf(pipe, prompt: str) -> str:
    import torch
    model, tokenizer = pipe
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=32, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.strip()


def call_ollama(url: str, model: str, prompt: str) -> str:
    import urllib.request
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        f"{url}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read()).get("response", "").strip()


# ── BM25 chunk loader ─────────────────────────────────────────────────

def load_chunks_by_id(bm25_path: str) -> dict:
    """Return dict of chunk_id → text for quick lookup."""
    with open(bm25_path) as f:
        raw = json.load(f)

    chunks_list = raw if isinstance(raw, list) else raw.get(
        "chunks", raw.get("documents", raw.get("index", [])))

    result = {}
    for c in chunks_list:
        if isinstance(c, dict):
            cid = str(c.get("id", c.get("chunk_id", "")))
            text = c.get("text", c.get("content", ""))
            result[cid] = text
    return result


# ── Verdict parsing ───────────────────────────────────────────────────

def parse_verdict(response: str) -> str:
    r = response.upper()
    # Check in order of specificity
    if "NOT_SUPPORTED" in r or "NOT SUPPORTED" in r:
        return "NOT_SUPPORTED"
    if "STRONG_SUPPORT" in r or "STRONG SUPPORT" in r:
        return "STRONG_SUPPORT"
    if "WEAK_SUPPORT" in r or "WEAK SUPPORT" in r:
        return "WEAK_SUPPORT"
    return "UNPARSEABLE"


# ── Cohen's Kappa ─────────────────────────────────────────────────────

def cohen_kappa(verdicts_a: list, verdicts_b: list, categories: list) -> float:
    """Compute Cohen's kappa for two lists of categorical verdicts."""
    n = len(verdicts_a)
    if n == 0:
        return 0.0

    # Observed agreement
    po = sum(1 for a, b in zip(verdicts_a, verdicts_b) if a == b) / n

    # Expected agreement
    pe = 0.0
    for cat in categories:
        pa = verdicts_a.count(cat) / n
        pb = verdicts_b.count(cat) / n
        pe += pa * pb

    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


# ── Confusion matrix ──────────────────────────────────────────────────

def print_confusion_matrix(verdicts_qwen: list, verdicts_cross: list):
    cats = VERDICT_ORDER
    print(f"\n  Confusion matrix (rows=Qwen-7B, cols={args_global.model.split('/')[-1]}):")
    header = f"{'':20s}" + "".join(f"{c:16s}" for c in cats)
    print(f"  {header}")
    for c1 in cats:
        row = f"  {c1:20s}"
        for c2 in cats:
            count = sum(1 for a, b in zip(verdicts_qwen, verdicts_cross) if a == c1 and b == c2)
            row += f"{count:16d}"
        print(row)


# ── Main ──────────────────────────────────────────────────────────────

args_global = None  # needed for confusion matrix helper

def run_exp_d(args):
    global args_global
    args_global = args

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Run-7 verified triples (Qwen-7B verdicts)
    print(f"[EXP-D] Loading verified triples from {args.verified}")
    all_triples = []
    with open(args.verified) as f:
        for line in f:
            line = line.strip()
            if line:
                all_triples.append(json.loads(line))
    print(f"[EXP-D] Loaded {len(all_triples)} verified triples")

    # Extract Qwen-7B verdict — handle multiple field name conventions
    def get_qwen_verdict(t):
        # Handle _verification.verdict (Run 7 format)
        ver = t.get("_verification", {})
        if isinstance(ver, dict):
            v = ver.get("verdict", "")
            if v in VERDICT_MAP:
                return v
        # Handle flat _verdict or verdict fields
        for key in ["_verdict", "verdict"]:
            val = t.get(key)
            if isinstance(val, str) and val in VERDICT_MAP:
                return val
        return None

    # Filter to triples that have a valid Qwen-7B verdict
    triples_with_verdict = [t for t in all_triples if get_qwen_verdict(t) in VERDICT_MAP]
    print(f"[EXP-D] Triples with valid Qwen-7B verdict: {len(triples_with_verdict)}")

    # Stratified sample
    if args.n_triples and args.n_triples < len(triples_with_verdict):
        by_verdict = defaultdict(list)
        for t in triples_with_verdict:
            by_verdict[get_qwen_verdict(t)].append(t)
        selected = []
        per_verdict = max(1, args.n_triples // len(by_verdict))
        for v, vtriples in by_verdict.items():
            selected.extend(vtriples[:per_verdict])
        triples_with_verdict = selected[:args.n_triples]
        print(f"[EXP-D] Stratified sample: {len(triples_with_verdict)} triples "
              f"({per_verdict} per verdict category)")

    # Load chunk lookup (for providing evidence context to cross-verifier)
    chunks_by_id = {}
    if os.path.exists(args.bm25):
        print(f"[EXP-D] Loading chunk index...")
        chunks_by_id = load_chunks_by_id(args.bm25)
        print(f"[EXP-D] {len(chunks_by_id)} chunks loaded")

    # Load cross-verifier model
    if args.backend == "hf":
        pipe = load_hf_model(args.model)
        llm_fn = lambda p: call_hf(pipe, p)
    else:
        llm_fn = lambda p: call_ollama(args.ollama_url, args.model, p)

    # --- Run cross-verification ---
    print(f"\n[EXP-D] Cross-verifying {len(triples_with_verdict)} triples "
          f"with {args.model}...")

    results = []
    verdicts_qwen = []
    verdicts_cross = []
    cross_verdict_counts = defaultdict(int)
    t0 = time.time()

    for i, t in enumerate(triples_with_verdict):
        subject = t.get("subject", t.get("source", ""))
        relation = t.get("relation", "")
        object_ = t.get("object", t.get("target", ""))
        qwen_verdict = get_qwen_verdict(t)

        # Get evidence chunk if available
        chunk_id = t.get("best_chunk_id", t.get("_provenance", {}).get("best_chunk_id", t.get("_provenance", {}).get("chunk_id", "")))
        chunk_text = chunks_by_id.get(str(chunk_id), "")

        if chunk_text:
            prompt = VERIFY_PROMPT_WITH_CHUNK.format(
                subject=subject, relation=relation,
                object=object_, chunk=chunk_text[:1500]
            )
        else:
            prompt = VERIFY_PROMPT_NO_CHUNK.format(
                subject=subject, relation=relation, object=object_
            )

        try:
            response = llm_fn(prompt)
            cross_verdict = parse_verdict(response)
        except Exception as e:
            cross_verdict = "BACKEND_ERROR"

        cross_verdict_counts[cross_verdict] += 1

        # Record only DECIDED verdicts for agreement analysis
        if qwen_verdict in VERDICT_MAP and cross_verdict in VERDICT_MAP:
            verdicts_qwen.append(qwen_verdict)
            verdicts_cross.append(cross_verdict)

        results.append({
            "subject": subject,
            "relation": relation,
            "object": object_,
            "qwen7b_verdict": qwen_verdict,
            "cross_verdict": cross_verdict,
            "agree": qwen_verdict == cross_verdict,
            "chunk_used": bool(chunk_text),
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i+1)/elapsed
            decided_so_far = len(verdicts_qwen)
            agree_so_far = sum(1 for r in results if r.get("agree") and r["qwen7b_verdict"] in VERDICT_MAP) / max(decided_so_far, 1)
            print(f"  [{i+1}/{len(triples_with_verdict)}] "
                  f"agreement={agree_so_far:.1%} rate={rate:.2f}/s")

    # --- Compute statistics ---
    n_decided = len(verdicts_qwen)
    if n_decided == 0:
        print("[EXP-D] ERROR: No decided verdicts to compare.", file=sys.stderr)
        sys.exit(1)

    agreement_rate = sum(1 for a, b in zip(verdicts_qwen, verdicts_cross) if a == b) / n_decided
    kappa = cohen_kappa(verdicts_qwen, verdicts_cross, VERDICT_ORDER)

    qwen_ns_rate = verdicts_qwen.count("NOT_SUPPORTED") / n_decided
    cross_ns_rate = verdicts_cross.count("NOT_SUPPORTED") / n_decided
    ns_delta = abs(qwen_ns_rate - cross_ns_rate)

    print("\n" + "="*60)
    print("EXP-D RESULTS — Cross-Model Verification")
    print("="*60)
    print(f"  Triples evaluated      : {n_decided}")
    print(f"  Qwen-7B model          : Qwen/Qwen2.5-7B-Instruct")
    print(f"  Cross-verifier         : {args.model}")
    print()
    print(f"  Agreement rate         : {agreement_rate:.1%}")
    print(f"  Cohen's kappa          : {kappa:.3f}")
    print()
    print(f"  NOT_SUPPORTED (Qwen-7B): {qwen_ns_rate:.1%}  ({verdicts_qwen.count('NOT_SUPPORTED')}/{n_decided})")
    print(f"  NOT_SUPPORTED (cross)  : {cross_ns_rate:.1%}  ({verdicts_cross.count('NOT_SUPPORTED')}/{n_decided})")
    print(f"  NS rate delta          : {ns_delta:.1%}")
    print()

    # Self-bias interpretation
    if ns_delta > 0.10:
        if qwen_ns_rate < cross_ns_rate:
            print("  → SELF-BIAS DETECTED: Qwen-7B is more lenient verifying its own triples")
            print("    (cross-verifier finds more NOT_SUPPORTED)")
        else:
            print("  → REVERSE BIAS: Qwen-7B is stricter than cross-verifier")
        print("    → Recommend: use cross-verifier rates in paper, or average both")
    elif ns_delta > 0.05:
        print("  → MODERATE BIAS: Some self-verification bias present (5–10pp delta)")
        print("    → Report both rates with confidence intervals in paper")
    else:
        print("  → MINIMAL BIAS: Self-verification bias is small (<5pp delta)")
        print("    → Qwen-7B self-verification is reliable")

    kappa_interp = ("poor" if kappa < 0.2 else "fair" if kappa < 0.4 else
                    "moderate" if kappa < 0.6 else "substantial" if kappa < 0.8 else "almost perfect")
    print(f"\n  Kappa={kappa:.3f} → {kappa_interp} inter-rater agreement")
    print("="*60)

    print_confusion_matrix(verdicts_qwen, verdicts_cross)

    # --- Save outputs ---
    results_out = out_dir / "exp_d_cross_verification_details.jsonl"
    with open(results_out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    stats = {
        "description": "EXP-D Cross-Model Verification",
        "qwen7b_model": "Qwen/Qwen2.5-7B-Instruct",
        "cross_verifier_model": args.model,
        "n_triples_evaluated": n_decided,
        "agreement_rate": round(agreement_rate, 4),
        "cohen_kappa": round(kappa, 4),
        "kappa_interpretation": kappa_interp,
        "qwen7b_not_supported_rate": round(qwen_ns_rate, 4),
        "cross_verifier_not_supported_rate": round(cross_ns_rate, 4),
        "ns_rate_delta": round(ns_delta, 4),
        "self_bias_detected": ns_delta > 0.10,
        "qwen7b_verdict_distribution": {
            v: verdicts_qwen.count(v) for v in VERDICT_ORDER
        },
        "cross_verifier_verdict_distribution": {
            v: verdicts_cross.count(v) for v in VERDICT_ORDER
        },
        "cross_verdict_all_counts": dict(cross_verdict_counts),
    }
    stats_out = out_dir / "exp_d_stats.json"
    with open(stats_out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[EXP-D] Stats saved   → {stats_out}")
    print(f"[EXP-D] Details saved → {results_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified", default=os.path.expanduser(
        "~/kg_test/output/step7/verified_triples_v7.jsonl"),
        help="Run-7 verified triples JSONL (Qwen-7B verdicts)")
    parser.add_argument("--bm25",    default=os.path.expanduser(
        "~/kg_test/output/step2/bm25_index.json"))
    parser.add_argument("--out",     default=os.path.expanduser(
        "~/kg_test/output/exp_d/"))
    parser.add_argument("--model",   default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Cross-verifier model. Options: Qwen/Qwen2.5-1.5B-Instruct, "
             "meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--backend", default="hf", choices=["hf", "ollama"])
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--n-triples", type=int, default=100,
        help="Number of triples (stratified sample). Set 0 for all.")
    args = parser.parse_args()

    run_exp_d(args)