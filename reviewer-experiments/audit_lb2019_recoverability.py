#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_lb2019_recoverability.py  (ADDITIF — ne touche pas run11)
Classe chaque edge LB2019: A_direct / A_or_B_same_chunk / B_distributed /
B_weak_cross_doc / C_absent, avec passages-preuve. Mesure le vrai taux de
recuperabilite sur le corpus.
A CONFIRMER (bloc CONFIG): chemin chunks.jsonl ; champs TEXT/DOCID ; SYNONYMS.
Usage: python audit_lb2019_recoverability.py --schema ./schema_generated/lb2019_schema_generated.json --chunks /chemin/chunks.jsonl --outdir ./recoverability_audit
"""
import argparse, json, re, sys, csv
from pathlib import Path
from collections import defaultdict

# ----------------------------- CONFIG -------------------------------------- #
TEXT_FIELDS   = ["text", "content", "chunk", "passage", "body"]
DOCID_FIELDS  = ["doc_id", "source", "document", "file", "path", "doc", "filename"]
EXCLUDE_DOC_SUBSTR = ["1-s2.0-S0025322701001141"]   # PDF encodage casse
DROP_CHECKPOINT    = True
N_EVIDENCE         = 3
ALLOW_PLURAL       = True
# >>> MEILLEUR : remplace patterns_for() par TON normaliseur/synonymes du pipeline.
SYNONYMS = {
    "mass transport deposit": ["mtd", "mtc", "mass-transport deposit", "mass transport complex",
                               "mass-transport complex", "mtds", "mass transport deposits"],
    "slope failure": ["slope failure", "slope instability", "submarine landslide", "slump"],
    "chaotic": ["chaotic", "chaotic facies", "chaotic reflection", "chaotic seismic facies"],
    "transparent": ["transparent", "transparent facies", "low-amplitude", "acoustically transparent"],
    "deformed": ["deformed", "deformed facies", "deformation"],
    "ridged": ["ridged", "ridges", "pressure ridge", "compressional ridge"],
    "strong amplitude": ["strong amplitude", "high amplitude", "high-amplitude", "strong reflector"],
    "runout distance": ["runout distance", "run-out distance", "runout length"],
    "fluid overpressure": ["fluid overpressure", "overpressure", "excess pore pressure"],
    "pore pressure": ["pore pressure", "pore-pressure", "fluid pressure"],
    "headscarp": ["headscarp", "head scarp", "headwall", "scarp"],
    "turbidity current": ["turbidity current", "turbidite"],
}
TRAILING_QUALIFIERS = ["distribution", "indicator", "downwards", "if attached",
    "in source zone", "in sedimentary pile", "to toe", "or ridges", "at toe"]
PREFIX_EXPAND = {"BS ": "basal surface ", "US ": "upper surface ",
                 "HS ": "headscarp ", "MTD ": "mass transport deposit "}
# --------------------------------------------------------------------------- #

def norm(s: str) -> str:
    s = s.lower().strip(); s = re.sub(r"[’']", "'", s); return re.sub(r"\s+", " ", s)

def keyword_of(label: str) -> str:
    s = norm(label)
    for p, full in PREFIX_EXPAND.items():
        if s.startswith(p.lower()): s = full + s[len(p):]
    s = s.split(":")[0]; s = re.sub(r",.*$", "", s)
    for q in sorted(TRAILING_QUALIFIERS, key=len, reverse=True):
        s = re.sub(r"\b" + re.escape(q) + r"\b\s*$", "", s).strip()
    return s.strip()

def patterns_for(label: str):
    kw = keyword_of(label); variants = {kw, norm(label)}
    for key, syns in SYNONYMS.items():
        if key in kw or kw in key: variants.update(norm(x) for x in syns)
    variants = {v for v in variants if len(v) >= 3}
    pats = []
    for v in variants:
        tail = r"s?\b" if (ALLOW_PLURAL and v[-1].isalpha() and v[-1] != "s") else r"\b"
        pats.append(re.compile(r"\b" + re.escape(v) + tail, re.I))
    return pats

def load_chunks(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: continue
    if not rows: sys.exit("aucune ligne JSON lue — verifie CHUNKS_JSONL")
    keys = rows[0].keys()
    tf = next((k for k in TEXT_FIELDS if k in keys), None)
    df = next((k for k in DOCID_FIELDS if k in keys), None)
    if tf is None: sys.exit(f"champ texte introuvable. cles: {list(keys)} ; edite TEXT_FIELDS")
    print(f"[chunks] champ texte = '{tf}' ; champ doc = '{df or '(aucun)'}'")
    chunks = []
    for i, r in enumerate(rows):
        doc = str(r.get(df, f"doc_{i}")) if df else f"doc_{i}"
        if DROP_CHECKPOINT and "-checkpoint" in doc: continue
        if any(sub in doc for sub in EXCLUDE_DOC_SUBSTR): continue
        txt = r.get(tf) or ""
        if not isinstance(txt, str) or not txt.strip(): continue
        chunks.append({"doc": doc, "idx": i, "text": txt})
    print(f"[chunks] retenus : {len(chunks)}")
    return chunks

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def matches_in_chunk(pats, text):
    return any(p.search(text) for p in pats)

def classify(src_label, tgt_label, chunks):
    ps, pt = patterns_for(src_label), patterns_for(tgt_label)
    src_docs, tgt_docs = set(), set()
    same_sentence = same_chunk = False
    evidence = []
    for c in chunks:
        s_hit = matches_in_chunk(ps, c["text"]); t_hit = matches_in_chunk(pt, c["text"])
        if s_hit: src_docs.add(c["doc"])
        if t_hit: tgt_docs.add(c["doc"])
        if s_hit and t_hit:
            same_chunk = True
            for sent in SENT_SPLIT.split(c["text"]):
                if matches_in_chunk(ps, sent) and matches_in_chunk(pt, sent):
                    same_sentence = True
                    if len(evidence) < N_EVIDENCE:
                        evidence.append({"doc": c["doc"], "idx": c["idx"],
                                         "tier": "same_sentence", "snippet": sent.strip()[:300]})
                    break
            if not same_sentence and len(evidence) < N_EVIDENCE:
                evidence.append({"doc": c["doc"], "idx": c["idx"],
                                 "tier": "same_chunk", "snippet": c["text"].strip()[:300]})
    same_doc = bool(src_docs and tgt_docs and (src_docs & tgt_docs))
    if not (src_docs and tgt_docs): cat = "C_absent"
    elif same_sentence: cat = "A_direct"
    elif same_chunk: cat = "A_or_B_same_chunk"
    elif same_doc: cat = "B_distributed"
    else: cat = "B_weak_cross_doc"
    return cat, len(src_docs), len(tgt_docs), evidence

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="./schema_generated/lb2019_schema_generated.json")
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--outdir", default="./recoverability_audit")
    args = ap.parse_args()
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    edges = schema["edges"]; nodes = schema["nodes"]
    chunks = load_chunks(Path(args.chunks))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rows, counts = [], defaultdict(int)
    by_role = defaultdict(lambda: defaultdict(int))
    for k, e in enumerate(edges, 1):
        if e.get("raw_label", "").lower().startswith("undetermined"): continue
        cat, ns, nt, ev = classify(e["source"], e["target"], chunks)
        counts[cat] += 1
        tgt_role = nodes.get(e["target"], {}).get("role", "?")
        tgt_prop = nodes.get(e["target"], {}).get("property")
        by_role[tgt_role][cat] += 1
        rows.append({"source": e["source"], "target": e["target"], "type": e["type"],
            "refs": ";".join(map(str, e["refs"])), "xlsx_row": e["xlsx_row"],
            "target_role": tgt_role, "target_property": tgt_prop, "category": cat,
            "n_docs_src": ns, "n_docs_tgt": nt,
            "evidence": json.dumps(ev, ensure_ascii=False)})
        if k % 25 == 0: print(f"  ...{k}/{len(edges)}")
    with (outdir / "edge_recoverability.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    total = sum(counts.values())
    recoverable = counts["A_direct"]
    plausible = counts["A_direct"] + counts["A_or_B_same_chunk"] + counts["B_distributed"]
    summary = {"total_edges_tested": total, "counts": dict(counts),
        "A_direct_pct": round(100*recoverable/total, 1) if total else 0,
        "present_in_literature_pct": round(100*plausible/total, 1) if total else 0,
        "C_absent_pct": round(100*counts["C_absent"]/total, 1) if total else 0,
        "by_target_role": {r: dict(c) for r, c in by_role.items()}}
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== RECUPERABILITE LB2019 sur le corpus ===")
    for cat in ["A_direct", "A_or_B_same_chunk", "B_distributed", "B_weak_cross_doc", "C_absent"]:
        print(f"  {cat:20s}: {counts[cat]:3d}  ({100*counts[cat]/total:.0f}%)")
    print(f"  enonce-direct (A)           : {summary['A_direct_pct']}%")
    print(f"  present dans la litterature : {summary['present_in_literature_pct']}%")
    print(f"  vraiment absent (C)         : {summary['C_absent_pct']}%")
    print(f"  -> {outdir}/edge_recoverability.csv  ;  {outdir}/summary.json")

if __name__ == "__main__":
    main()
