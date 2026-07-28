#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_v2_tokens.py (ADDITIF) — recuperabilite LB2019 par chevauchement de tokens.
Source = reference_kg.json (90 noeuds propres, 174 edges). Pas le xlsx brut.
Matching = tokens porteurs du label (sans stopwords/qualificatifs), seuil de couverture.
Sort: edge_recoverability_v2.csv + summary_v2.json + node_presence_v2.csv
"""
import argparse, json, re, csv
from pathlib import Path
from collections import defaultdict

TEXT_FIELDS  = ["text","content","chunk","passage","body"]
DOCID_FIELDS = ["doc_id","source","document","file","path","doc","filename"]
EXCLUDE_DOC_SUBSTR = ["1-s2.0-S0025322701001141"]
DROP_CHECKPOINT = True
N_EVIDENCE = 3
MIN_TOKEN_COVER = 0.6   # fraction des tokens porteurs du label requis dans un chunk

STOP = set("""of the a an in on to and or for with by at from into is are be as
than over under between within without not no if attached its their our it
this that these those high low more less most very near other another each
all any some general typical resp etc via per due e.g i.e""".split())
# qualificatifs LB2019 a ignorer (mots non-discriminants)
QUALIF = set("""distribution indicator downslope downwards zone proportion sub-horizontal
connection presence variation evolution morphology ratio aspect angle increase
elements laterally structural sup inf""".split())
PREFIX = {"bs":"basal surface","us":"upper surface","hs":"headscarp","mtd":"mass transport deposit"}

def tokenize(label):
    s = re.sub(r"[’']","'",label.lower())
    s = s.split(":")[0]                       # 'flow behavior: ...' -> 'flow behavior'
    s = re.sub(r"[^a-z0-9\- ]"," ",s)
    raw = [t for t in s.split() if t]
    toks=[]
    for t in raw:
        toks += PREFIX.get(t,t).split()       # expand bs/us/hs/mtd
    toks=[t for t in toks if t not in STOP and t not in QUALIF and len(t)>=3]
    # singularise grossierement
    toks=[re.sub(r"s$","",t) if len(t)>4 and t.endswith("s") else t for t in toks]
    return toks

def make_matcher(label):
    toks=set(tokenize(label))
    if not toks: toks={re.sub(r"[^a-z]","",label.lower())[:6] or "zzz"}
    pats=[re.compile(r"\b"+re.escape(t)+r"s?\b",re.I) for t in toks]
    need=max(1,round(MIN_TOKEN_COVER*len(toks)))
    return pats,need
def hit(matcher,text):
    pats,need=matcher
    return sum(1 for p in pats if p.search(text))>=need

def load_chunks(path):
    rows=[]
    for line in path.open(encoding="utf-8"):
        line=line.strip()
        if line:
            try: rows.append(json.loads(line))
            except: pass
    keys=rows[0].keys()
    tf=next((k for k in TEXT_FIELDS if k in keys),None)
    df=next((k for k in DOCID_FIELDS if k in keys),None)
    print(f"[chunks] texte='{tf}' doc='{df}'")
    out=[]
    for i,r in enumerate(rows):
        doc=str(r.get(df,f"doc_{i}")) if df else f"doc_{i}"
        if DROP_CHECKPOINT and "-checkpoint" in doc: continue
        if any(s in doc for s in EXCLUDE_DOC_SUBSTR): continue
        t=r.get(tf) or ""
        if isinstance(t,str) and t.strip(): out.append({"doc":doc,"idx":i,"text":t})
    print(f"[chunks] retenus: {len(out)}")
    return out

SENT=re.compile(r"(?<=[.!?])\s+")
def classify(ms,mt,chunks):
    sd,td=set(),set(); ss=sc=False; ev=[]
    for c in chunks:
        sh=hit(ms,c["text"]); th=hit(mt,c["text"])
        if sh: sd.add(c["doc"])
        if th: td.add(c["doc"])
        if sh and th:
            sc=True
            for s in SENT.split(c["text"]):
                if hit(ms,s) and hit(mt,s):
                    ss=True
                    if len(ev)<N_EVIDENCE: ev.append({"doc":c["doc"],"tier":"same_sentence","snip":s.strip()[:300]})
                    break
            if not ss and len(ev)<N_EVIDENCE: ev.append({"doc":c["doc"],"tier":"same_chunk","snip":c["text"].strip()[:300]})
    sdoc=bool(sd and td and (sd&td))
    cat=("C_absent" if not(sd and td) else "A_direct" if ss else
         "A_or_B_same_chunk" if sc else "B_distributed" if sdoc else "B_weak_cross_doc")
    return cat,len(sd),len(td),ev

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--kg",default="/home/talbi/ontogeorag/reference/reference_kg.json")
    ap.add_argument("--chunks",required=True)
    ap.add_argument("--outdir",default="./recoverability_audit_v2")
    a=ap.parse_args()
    kg=json.loads(Path(a.kg).read_text(encoding="utf-8"))
    nodes=kg["nodes"]; edges=kg["edges"]
    chunks=load_chunks(Path(a.chunks))
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    M={lbl:make_matcher(v.get("label",lbl)) for lbl,v in nodes.items()}

    # presence par noeud (diagnostic du matcher)
    npres=[]
    for lbl,m in M.items():
        nd=len({c["doc"] for c in chunks if hit(m,c["text"])})
        npres.append((nd,lbl,nodes[lbl].get("main_category","")))
    zero=[x for x in npres if x[0]==0]
    with (out/"node_presence_v2.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["n_docs","label","main_category"])
        for nd,lbl,mc in sorted(npres): w.writerow([nd,lbl,mc])
    print(f"\n[noeuds] {len(M)} | 0 occurrence: {len(zero)}")
    for _,lbl,_ in zero[:25]: print("   ZERO:",lbl)

    rows=[]; counts=defaultdict(int)
    for k,e in enumerate(edges,1):
        s,t=e["source"],e["target"]
        if s not in M or t not in M:   # noeud absent du dict (rare)
            M.setdefault(s,make_matcher(s)); M.setdefault(t,make_matcher(t))
        cat,ns,nt,ev=classify(M[s],M[t],chunks); counts[cat]+=1
        rows.append({"source":s,"target":t,"relation":e.get("relation"),
            "directed":e.get("directed"),"n_cit":len(e.get("citations",[])),
            "category":cat,"n_docs_src":ns,"n_docs_tgt":nt,
            "evidence":json.dumps(ev,ensure_ascii=False)})
        if k%25==0: print(f"  ...{k}/{len(edges)}")
    with (out/"edge_recoverability_v2.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    tot=sum(counts.values())
    A=counts["A_direct"]; present=tot-counts["C_absent"]
    summ={"total":tot,"counts":dict(counts),
        "A_direct_pct":round(100*A/tot,1),
        "present_pct":round(100*present/tot,1),
        "C_absent_pct":round(100*counts['C_absent']/tot,1),
        "MIN_TOKEN_COVER":MIN_TOKEN_COVER}
    (out/"summary_v2.json").write_text(json.dumps(summ,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\n=== RECUPERABILITE v2 (token-overlap) ===")
    for c in ["A_direct","A_or_B_same_chunk","B_distributed","B_weak_cross_doc","C_absent"]:
        print(f"  {c:20s}: {counts[c]:3d} ({100*counts[c]/tot:.0f}%)")
    print(f"  present dans la litterature : {summ['present_pct']}%")
    print(f"  -> {out}/")

if __name__=="__main__":
    main()
