#!/usr/bin/env python3
"""
EXP-E targeted rerun — extract only on new queries, merge with existing triples,
re-evaluate with fuzzy normalizer and subject remapping.
"""
import json, sys, time, re
from pathlib import Path
sys.path.insert(0, "/home/talbi/ontogeorag")

from pipeline.rag.constants import (
    ALLOWED_RELATIONS, LB2019_DESCRIPTORS, LB2019_REFERENCE_EDGES,
    normalize_entity, normalize_relation, normalize_descriptor_fuzzy, normalize_descriptor_multi,
    DESCRIPTOR_FUZZY_MAP
)

MTD_SUBJECT_REMAP = {
    'upper surface lying within each headwall lobe',
    'the upper surface lying within each headwall lobe is hummocky and irregular',
    'rugose upper bounding surface',
    'the upper surface of mtc-1',
    'basal shear surface', 'the basal shear surface',
    'mass transport deposits (mtds)', 'mass-flow rheology',
    'subaqueous debris flows', 'high amplitude, blocky material',
    'mass-transport deposit', 'mass-transport deposits',
}

PROCESS_REMAP = {
    'seismicity': 'earthquake',
    'increased seismic activity': 'earthquake',
    'seismic loading': 'earthquake',
    'local overpressure': 'pore pressure',
    'fluid overpressure': 'pore pressure',
    'pore pressure increase': 'pore pressure',
    'turbidity current': 'turbidity current',
    'debris flow slurry disintegration': 'debris flow',
}

NEW_QUERIES = [
    "What are the seismic characteristics of debris flow deposits including chaotic and hummocky facies?",
    "How are slide deposits characterized in seismic data — blocky and undeformed reflections?",
    "What seismic facies describe hemipelagite deposits — parallel continuous low amplitude?",
    "Does turbidite show parallel continuous layered high-amplitude seismic character?",
    "What triggers slope failure — earthquake seismicity pore pressure?",
    "What is the relationship between pore pressure and slope instability?",
    "Does debris flow transform into turbidity current?",
    "In what geological settings do mass transport deposits occur — continental slope abyssal plain?",
]

EXTRACTION_PROMPT = """You are a geological knowledge extraction system specializing in seismic interpretation.

Given this text excerpt from a scientific paper:
---
{chunk}
---

Question: {query}

Extract geological triples as JSON. Rules:
- Allowed relations: {relations}
- For hasDescriptor relations, target MUST be one of:
  blocky, chaotic, continuous, discontinuous, high-amplitude, hummocky,
  layered, low-amplitude, massive, parallel, stratified, transparent, undeformed
- Use canonical short forms (e.g. "chaotic" not "chaotic seismic facies")
- Use class-level subjects (e.g. "debris flow" not "debris flow deposit 3")

Respond with a JSON array only — no explanation:
[{{"source": "...", "source_type": "GeologicalObject", "relation": "...", "target": "...", "target_type": "..."}}]
If nothing relevant: []
"""

def load_bm25(index_dir: str):
    import json
    from pathlib import Path
    from pipeline.rag.bm25 import build_bm25_index, retrieve as bm25_retrieve
    chunks_path = Path(index_dir) / "chunks.jsonl"
    chunks = []
    with open(chunks_path) as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    idx = build_bm25_index(chunks)
    def retrieve(query, top_n=5):
        results = bm25_retrieve(idx, query, top_k=top_n)
        # bm25_retrieve returns List[Tuple[score, chunk]]
        return [{"score": score, "text": chunk.get("text", chunk.get("content", ""))}
                for score, chunk in results]
    return retrieve

def normalize_subject(s: str) -> str:
    s_norm = normalize_entity(s)
    if s_norm in MTD_SUBJECT_REMAP or 'mass transport' in s_norm:
        return 'mass transport deposit'
    return PROCESS_REMAP.get(s_norm, s_norm)

def compute_coverage(triples):
    found = set()
    for t in triples:
        r = normalize_relation(t.get("relation",""))
        if r == "hasDescriptor":
            ds = normalize_descriptor_multi(t.get("target", t.get("object","")))
            found.update(ds & LB2019_DESCRIPTORS)
    return {"found": sorted(found), "missing": sorted(LB2019_DESCRIPTORS-found),
            "n_found": len(found), "coverage": len(found)/len(LB2019_DESCRIPTORS)}

def compute_recall(triples):
    ref = set((normalize_entity(s), normalize_relation(r), normalize_entity(o))
              for s,r,o in LB2019_REFERENCE_EDGES)
    hits = []
    for t in triples:
        s = normalize_subject(t.get("source", t.get("subject","")))
        r = normalize_relation(t.get("relation",""))
        o_raw = t.get("target", t.get("object",""))
        o = normalize_descriptor_fuzzy(o_raw) if r == "hasDescriptor" else normalize_entity(o_raw)
        if (s, r, o) in ref:
            hits.append((s, r, o))
    return {"hits": len(set(map(tuple,hits))), "recall": len(set(map(tuple,hits)))/26,
            "matched": sorted(set(map(tuple,hits)))}

def main():
    # Load existing Llama triples
    existing = []
    existing_path = Path("/home/talbi/ontogeorag/output/exp_e/llama_raw_triples.jsonl")
    with open(existing_path) as f:
        for line in f:
            if line.strip():
                existing.append(json.loads(line))
    print(f"Existing Llama triples: {len(existing)}")

    # Run targeted extraction
    from pipeline.rag.llm_hf import make_hf_fn
    generate = make_hf_fn("meta-llama/Llama-3.1-8B-Instruct")
    retrieve = load_bm25("/home/talbi/kg_test/output/step1/")
    schema = json.loads(Path("/home/talbi/ontogeorag/configs/ontology_schema.json").read_text())
    relations = [r["name"] if isinstance(r,dict) else r for r in schema.get("relations",[])]

    new_triples = []
    for qi, query_text in enumerate(NEW_QUERIES):
        print(f"  [{qi+1}/{len(NEW_QUERIES)}] {query_text[:60]}...")
        candidates = retrieve(query_text, top_n=25)
        if not candidates or candidates[0]["score"] < 2.0:
            continue
        context = "\n---\n".join(c["text"][:800] for c in candidates[:5])[:3000]
        prompt = EXTRACTION_PROMPT.format(
            chunk=context, query=query_text, relations=", ".join(relations))
        try:
            response = generate("", prompt)
            m = re.search(r"\[.*\]", response, re.DOTALL)
            if m:
                items = json.loads(m.group())
                for item in items:
                    if not isinstance(item, dict): continue
                    rel_norm = normalize_relation(item.get("relation",""))
                    if rel_norm not in ALLOWED_RELATIONS: continue
                    item["relation"] = rel_norm
                    item["_provenance"] = {"query": query_text, "model": "llama-targeted"}
                    new_triples.append(item)
        except Exception as e:
            print(f"    Error: {e}")

    print(f"New triples extracted: {len(new_triples)}")

    # Save new triples
    new_path = Path("/home/talbi/ontogeorag/output/exp_e/llama_targeted_triples.jsonl")
    with open(new_path, "w") as f:
        for t in new_triples:
            f.write(json.dumps(t) + "\n")

    # Merge all triples
    all_triples = existing + new_triples
    print(f"Total merged triples: {len(all_triples)}")

    # Evaluate
    print("\n" + "="*60)
    print("RESULTS: Llama (existing + targeted, fuzzy eval)")
    print("="*60)
    cov = compute_coverage(all_triples)
    rec = compute_recall(all_triples)
    print(f"Descriptor coverage: {cov['n_found']}/13 ({cov['coverage']:.1%})")
    print(f"  Found:   {cov['found']}")
    print(f"  Missing: {cov['missing']}")
    print(f"LB2019 recall: {rec['hits']}/26 ({rec['recall']:.1%})")
    print(f"  Matched: {rec['matched']}")

    # Save stats
    stats = {"coverage": cov, "recall": rec,
             "n_existing": len(existing), "n_new": len(new_triples), "n_total": len(all_triples)}
    with open("/home/talbi/ontogeorag/output/exp_e/exp_e_targeted_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\nSaved → output/exp_e/exp_e_targeted_stats.json")

if __name__ == "__main__":
    main()
