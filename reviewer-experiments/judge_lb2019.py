#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
judge_lb2019.py (ADDITIF) — juge la recuperabilite des 174 edges LB2019.
Reutilise parse_cot du pipeline. Prompt SANS type de relation impose (zero ALLOWED_RELATIONS).
Backend hf, meme modele que run11 (Qwen2.5-7B). Filtre passages dechet en amont.
Sort: judge_lb2019.csv + judge_summary.json
"""
import argparse, json, csv, sys, re
from pathlib import Path
from collections import defaultdict

REPO = "/home/talbi/ontogeorag"
sys.path.insert(0, REPO)
sys.path.insert(0, str(Path(__file__).parent))
from pipeline.rag.constants import ALLOWED_RELATIONS  # importe juste pour info (non utilise)
import importlib.util
# parse_cot vit dans 03_verify_triples.py (nom avec chiffre -> import par chemin)
spec = importlib.util.spec_from_file_location("vt", f"{REPO}/pipeline/03_verify_triples.py")
vt = importlib.util.module_from_spec(spec); spec.loader.exec_module(vt)
parse_cot = vt.parse_cot

from passage_filter import is_junk, both_present

# --- prompt "relation libre" : meme grammaire de sortie que COT_PROMPT, SANS {relation}/{gloss}
SYSTEM = ("You are a precise geological fact-checker. "
          "Use ONLY the provided source text. Follow formatting exactly.")
PROMPT = """\
You are a geological fact-checker. Decide whether the source text expresses ANY
relationship between two geological concepts (Subject and Object).
=== SOURCE TEXT ===
{chunk_text}
=== END SOURCE TEXT ===
=== CONCEPTS ===
  Subject: {source}
  Object:  {target}
=== END CONCEPTS ===
Follow steps:
STEP 1 — EVIDENCE: Copy the most relevant sentence(s) linking Subject and Object.
If none, write "NO EVIDENCE FOUND".
STEP 2 — REASONING: In 1-2 sentences, explain whether the text states or implies a
relationship between them. Use ONLY the source text.
STEP 3 — VERDICT: Choose exactly one:
  STRONG_SUPPORT   — text explicitly states a relationship between Subject and Object
  WEAK_SUPPORT     — text implies / supports a direct inference of a relationship
  NOT_SUPPORTED    — no relationship between them is stated or implied
Format EXACTLY:
EVIDENCE: <...>
REASONING: <...>
VERDICT: <STRONG_SUPPORT or WEAK_SUPPORT or NOT_SUPPORTED>
"""

def build_generate_fn(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[hf] chargement {model_name} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    def gen(system, user):
        msgs = [{"role":"system","content":system},{"role":"user","content":user}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=450, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return gen

def evidence_passages(row):
    try: ev = json.loads(row["evidence"])
    except Exception: ev = []
    return [e.get("snip","") for e in ev if e.get("snip")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="recoverability_audit_v2/edge_recoverability_v2.csv")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--outdir", default="./judge_audit")
    ap.add_argument("--chunk-chars", type=int, default=1500)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    gen = build_generate_fn(args.model)

    results, counts = [], defaultdict(int)
    for i, r in enumerate(rows, 1):
        src, tgt, cat = r["source"], r["target"], r["category"]
        if cat == "C_absent":
            verdict, why, used = "NO_CHUNK", "absent du corpus", ""
        else:
            # passages: garde non-dechet ET contenant les 2 entites
            cands = [p for p in evidence_passages(r)
                     if not is_junk(p)]
            if not cands:
                verdict, why, used = "NO_USABLE_PASSAGE", "passages dechet/non co-presents", ""
            else:
                best, rank = None, -1
                order = {"STRONG_SUPPORT":3,"WEAK_SUPPORT":2,"NOT_SUPPORTED":1,"UNPARSEABLE":0}
                for p in cands[:1]:
                    resp = gen(SYSTEM, PROMPT.format(chunk_text=p[:args.chunk_chars], source=src, target=tgt))
                    parsed = parse_cot(resp)
                    s = order.get(parsed["verdict"], 0)
                    if s > rank: rank, best, used = s, parsed, p
                verdict = best["verdict"]; why = best.get("reasoning","")
        counts[verdict] += 1
        results.append({"source":src,"target":tgt,"relation":r.get("relation"),
                        "token_category":cat,"verdict":verdict,
                        "reasoning":why[:300],"passage_used":used[:300]})
        if i % 10 == 0: print(f"  ...{i}/{len(rows)}", flush=True)

    with (out/"judge_lb2019.csv").open("w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    tot = len(results)
    supported = counts["STRONG_SUPPORT"] + counts["WEAK_SUPPORT"]
    summ = {"model":args.model,"total":tot,"counts":dict(counts),
            "supported_pct": round(100*supported/tot,1),
            "strong_pct": round(100*counts['STRONG_SUPPORT']/tot,1)}
    (out/"judge_summary.json").write_text(json.dumps(summ,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\n=== JUGEMENT LB2019 (Qwen2.5-7B, meme juge que run11) ===")
    for k in sorted(counts): print(f"  {k:20s}: {counts[k]:3d}")
    print(f"  -------------------------------------------")
    print(f"  SUPPORTED (STRONG+WEAK) : {supported}/{tot} = {summ['supported_pct']}%")
    print(f"  dont STRONG            : {summ['strong_pct']}%")
    print(f"  -> {out}/judge_lb2019.csv ; judge_summary.json")

if __name__ == "__main__":
    main()
