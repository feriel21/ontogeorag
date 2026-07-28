#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
judge_retrieval.py (ADDITIF) — recuperabilite LB2019 via LE retrieval du pipeline.
Pour chaque edge: requete "{source} {target}" -> hybrid_retriever (BM25+dense+CrossEncoder)
-> top passages filtres (is_junk) -> juge Qwen-7B (prompt relation libre, parse_cot).
Mesure: le pipeline retrouve-t-il un passage qui ENONCE la relation ?
Sort: judge_retrieval.csv + judge_retrieval_summary.json
"""
import argparse, json, csv, sys, importlib.util
from pathlib import Path
from collections import defaultdict

REPO = "/home/talbi/ontogeorag"
sys.path.insert(0, REPO); sys.path.insert(0, str(Path(__file__).parent))
from pipeline.rag.hybrid_retriever import load_hybrid_retriever
spec = importlib.util.spec_from_file_location("vt", f"{REPO}/pipeline/03_verify_triples.py")
vt = importlib.util.module_from_spec(spec); spec.loader.exec_module(vt)
parse_cot = vt.parse_cot
from passage_filter import is_junk

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
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto").eval()
    def gen(system, user):
        msgs=[{"role":"system","content":system},{"role":"user","content":user}]
        ids=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out=model.generate(ids, max_new_tokens=450, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return gen

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--kg", default="/home/talbi/ontogeorag/reference/reference_kg.json")
    ap.add_argument("--index-dir", default="/home/talbi/ontogeorag/output/step1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--outdir", default="./judge_retrieval_audit")
    a=ap.parse_args()

    kg=json.loads(Path(a.kg).read_text(encoding="utf-8")); edges=kg["edges"]
    out=Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    print("[retriever] chargement hybrid (BM25+dense+CrossEncoder)...", flush=True)
    retrieve = load_hybrid_retriever(a.index_dir, reranker_model=a.reranker)
    gen = build_generate_fn(a.model)

    order={"STRONG_SUPPORT":3,"WEAK_SUPPORT":2,"NOT_SUPPORTED":1,"UNPARSEABLE":0}
    results, counts = [], defaultdict(int)
    for i,e in enumerate(edges,1):
        src,tgt = e["source"], e["target"]
        query = f"{src} {tgt}"
        try:
            hits = retrieve(query, top_n=a.top_n)
        except Exception as ex:
            hits=[]; print("retrieve err:", ex)
        passages=[h.get("text","") for h in hits if h.get("text") and not is_junk(h.get("text",""))]
        if not passages:
            verdict,why,used="NO_PASSAGE_RETRIEVED","",""
        else:
            best,rank=None,-1
            for p in passages[:3]:
                parsed=parse_cot(gen(SYSTEM, PROMPT.format(chunk_text=p[:1500], source=src, target=tgt)))
                s=order.get(parsed["verdict"],0)
                if s>rank: rank,best,used=s,parsed,p
            verdict=best["verdict"]; why=best.get("reasoning","")
        counts[verdict]+=1
        results.append({"source":src,"target":tgt,"relation":e.get("relation"),
            "verdict":verdict,"query":query,"reasoning":why[:300],"passage_used":used[:400]})
        if i%10==0: print(f"  ...{i}/{len(edges)}", flush=True)

    with (out/"judge_retrieval.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    tot=len(results); sup=counts["STRONG_SUPPORT"]+counts["WEAK_SUPPORT"]
    summ={"model":a.model,"top_n":a.top_n,"total":tot,"counts":dict(counts),
          "supported_pct":round(100*sup/tot,1),"strong_pct":round(100*counts['STRONG_SUPPORT']/tot,1)}
    (out/"judge_retrieval_summary.json").write_text(json.dumps(summ,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\n=== RECUPERABILITE via RETRIEVAL du pipeline (Qwen-7B juge) ===")
    for k in sorted(counts): print(f"  {k:22s}: {counts[k]:3d}")
    print(f"  SUPPORTED (STRONG+WEAK): {sup}/{tot} = {summ['supported_pct']}%")
    print(f"  -> {out}/")

if __name__=="__main__":
    main()
