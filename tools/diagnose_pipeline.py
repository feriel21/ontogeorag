#!/usr/bin/env python3
"""
OntoGeoRAG Pipeline Diagnostic v2
===================================
Fixed: correct filenames (raw_triples.jsonl, verified_triples.jsonl)
Added: deep analysis of WHY the LLM misses the 13 QUERY_MISS edges

Usage:
    python diagnose_pipeline_v2.py 2>&1 | tee output/diagnostic_report_v2.txt
"""

import json, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/home/talbi/ontogeorag")

REPO    = Path("/home/talbi/ontogeorag")
KG_TEST = Path("/home/talbi/kg_test")

# ── CORRECTED file paths ───────────────────────────────────────────────
RUN9_A_RAW      = REPO / "output/run9_a/raw_triples.jsonl"
RUN9_A_VERIFIED = REPO / "output/run9_a/verified_triples.jsonl"
RUN9_A_CANON    = REPO / "output/run9_a/canonical_triples_v5.jsonl"
RUN9_A_REJECTED = REPO / "output/run9_a/rejected_triples_v5.jsonl"
RUN9_A_VSTATS   = REPO / "output/run9_a/verification_stats.json"

RUN9_B_RAW      = REPO / "output/run9_b/raw_triples.jsonl"
RUN9_B_VERIFIED = REPO / "output/run9_b/verified_triples.jsonl"
RUN9_B_CANON    = REPO / "output/run9_b/canonical_triples_v5.jsonl"
RUN9_B_REJECTED = REPO / "output/run9_b/rejected_triples_v5.jsonl"

FINAL_KG  = REPO / "output/run9_kg/tmp_ab.json"
BENCHMARK = REPO / "configs/lb_reference_edges.json"
QUERIES   = REPO / "configs/descriptor_queries.jsonl"
CHUNKS    = KG_TEST / "output/step1/chunks.jsonl"

# ── Helpers ────────────────────────────────────────────────────────────
def load_jsonl(path):
    if not Path(path).exists():
        print(f"  [MISSING] {path}", file=sys.stderr)
        return []
    triples, errors = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                triples.append(json.loads(line))
            except Exception:
                errors += 1
    if errors:
        print(f"  [WARN] {path}: {errors} JSON parse errors", file=sys.stderr)
    return triples

def load_json_kg(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("triples", data) if isinstance(data, dict) else data

def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower().strip()).rstrip(".,;:")

def get_field(t, *fields):
    for f in fields:
        v = t.get(f)
        if v:
            return v
    return ""

def subj(t): return norm(get_field(t, "subject", "source"))
def obj(t):  return norm(get_field(t, "object",  "target"))
def rel(t):  return norm(get_field(t, "relation"))
def verdict(t):
    v = (t.get("verdict") or t.get("_verdict") or
         (t.get("_verification") or {}).get("verdict") or
         t.get("verif_verdict") or "UNKNOWN")
    return str(v).upper()

def matches_ref(t, ref):
    ks, ko, kr = subj(t), obj(t), rel(t)
    rs = norm(ref.get("subject", ""))
    ro = norm(ref.get("object",  ""))
    rr = norm(ref.get("relation", ""))
    return ((ks == rs or ks in rs or rs in ks) and
            (ko == ro or ko in ro or ro in ko) and kr == rr)

lines = []
def hdr(t):
    lines.append(f"\n{'='*70}\n  {t}\n{'='*70}")
def sub(t): lines.append(f"\n--- {t} ---")
def p(t=""): lines.append(str(t))

# ══════════════════════════════════════════════════════════════════════
# LOAD ALL DATA
# ══════════════════════════════════════════════════════════════════════
p("Loading data...")
raw_a    = load_jsonl(RUN9_A_RAW)
ver_a    = load_jsonl(RUN9_A_VERIFIED)
can_a    = load_jsonl(RUN9_A_CANON)
rej_a    = load_jsonl(RUN9_A_REJECTED)

raw_b    = load_jsonl(RUN9_B_RAW)
ver_b    = load_jsonl(RUN9_B_VERIFIED)
can_b    = load_jsonl(RUN9_B_CANON)
rej_b    = load_jsonl(RUN9_B_REJECTED)

final_kg = load_json_kg(FINAL_KG)

ref_data  = json.load(open(BENCHMARK))
ref_edges = ref_data.get("edges", ref_data)

chunks = []
if CHUNKS.exists():
    chunks = [json.loads(l) for l in open(CHUNKS) if l.strip()]

def chunk_text(c): return (c.get("text") or c.get("content") or "").lower()

# Load queries (handle malformed JSONL)
queries = []
if QUERIES.exists():
    with open(QUERIES) as f:
        raw_content = f.read().strip()
    # Try line-by-line first
    for line in raw_content.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            queries.append(json.loads(line))
        except Exception:
            pass
    if not queries:
        # Try as JSON array
        try:
            queries = json.loads(raw_content)
        except Exception:
            pass

p(f"Loaded: raw_a={len(raw_a)}, ver_a={len(ver_a)}, can_a={len(can_a)}, rej_a={len(rej_a)}")
p(f"Loaded: raw_b={len(raw_b)}, ver_b={len(ver_b)}, can_b={len(can_b)}, rej_b={len(rej_b)}")
p(f"Loaded: final_kg={len(final_kg)}, chunks={len(chunks)}, queries={len(queries)}")

# ══════════════════════════════════════════════════════════════════════
# 1. FUNNEL
# ══════════════════════════════════════════════════════════════════════
hdr("1. EXTRACTION FUNNEL")

p(f"{'Stage':<35} {'Pass A':>8} {'Pass B':>8} {'Total':>8}")
p("-" * 62)
p(f"{'Raw extracted':<35} {len(raw_a):>8} {len(raw_b):>8} {len(raw_a)+len(raw_b):>8}")
p(f"{'Verified (all verdicts)':<35} {len(ver_a):>8} {len(ver_b):>8} {len(ver_a)+len(ver_b):>8}")
p(f"{'Rejected at cleaning':<35} {len(rej_a):>8} {len(rej_b):>8} {len(rej_a)+len(rej_b):>8}")
p(f"{'Canonical (kept)':<35} {len(can_a):>8} {len(can_b):>8} {len(can_a)+len(can_b):>8}")
p(f"{'Final KG (fused)':<35} {'':>8} {'':>8} {len(final_kg):>8}")

for label, raw, ver, can in [("A", raw_a, ver_a, can_a), ("B", raw_b, ver_b, can_b)]:
    if raw and ver:
        v_loss = (len(raw) - len(ver)) / len(raw) * 100
        c_loss = (len(ver) - len(can)) / len(ver) * 100 if ver else 0
        p(f"Pass {label}: raw→verified loss={v_loss:.1f}%  verified→canonical loss={c_loss:.1f}%")

# ══════════════════════════════════════════════════════════════════════
# 2. VERDICT BREAKDOWN
# ══════════════════════════════════════════════════════════════════════
hdr("2. VERDICT BREAKDOWN")

for label, vt in [("Pass A verified", ver_a), ("Pass B verified", ver_b),
                  ("Pass A rejected", rej_a), ("Pass B rejected", rej_b)]:
    if not vt:
        continue
    sub(label)
    vc = Counter(verdict(t) for t in vt)
    total = len(vt)
    for v, cnt in sorted(vc.items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / total * 40)
        p(f"  {v:<22} {cnt:>5} ({cnt/total*100:5.1f}%)  {bar}")

# ══════════════════════════════════════════════════════════════════════
# 3. VERIFICATION STATS
# ══════════════════════════════════════════════════════════════════════
hdr("3. VERIFICATION STATS FROM JSON FILES")
for path in [RUN9_A_VSTATS]:
    if path.exists():
        stats = json.load(open(path))
        p(json.dumps(stats, indent=2))

# ══════════════════════════════════════════════════════════════════════
# 4. BENCHMARK GAP — root cause per missing edge
# ══════════════════════════════════════════════════════════════════════
hdr("4. BENCHMARK GAP — Root cause per missing edge")

all_raw = raw_a + raw_b
all_ver = ver_a + ver_b
all_can = can_a + can_b
all_rej = rej_a + rej_b

matched = [r for r in ref_edges if any(matches_ref(t, r) for t in final_kg)]
missing = [r for r in ref_edges if not any(matches_ref(t, r) for t in final_kg)]

p(f"Matched: {len(matched)}/26   Missing: {len(missing)}/26")

root_cause_counts = Counter()

for ref in missing:
    rs = norm(ref.get("subject", ""))
    ro = norm(ref.get("object",  ""))
    rr = norm(ref.get("relation", ""))

    in_raw   = any(matches_ref(t, ref) for t in all_raw)
    in_ver   = any(matches_ref(t, ref) for t in all_ver)
    in_can   = any(matches_ref(t, ref) for t in all_can)
    in_rej   = any(matches_ref(t, ref) for t in all_rej)
    in_kg    = any(matches_ref(t, ref) for t in final_kg)

    subj_in_raw = any(rs in subj(t) or subj(t) in rs for t in all_raw if subj(t))
    obj_in_raw  = any(ro in obj(t)  or obj(t)  in ro for t in all_raw if obj(t))

    corpus_hits = sum(1 for c in chunks if rs in chunk_text(c) and ro in chunk_text(c))

    # Root cause determination
    if corpus_hits == 0:
        cause = "CORPUS_ABSENT"
    elif not in_raw and not subj_in_raw:
        cause = "SUBJECT_NEVER_EXTRACTED"
    elif not in_raw and subj_in_raw:
        cause = "RELATION_MISSED — subject extracted but not with this object/relation"
    elif in_raw and not in_ver:
        cause = "VERIFIER_KILL"
    elif in_ver and not in_can and in_rej:
        cause = "CANON_KILL"
    elif in_can and not in_kg:
        cause = "FUSION_LOSS"
    else:
        cause = "QUERY_MISS_UNKNOWN"

    root_cause_counts[cause] += 1

    p(f"\n  MISSING: {ref['subject']} --[{ref['relation']}]--> {ref['object']}")
    p(f"    corpus co-occur: {corpus_hits} chunks")
    p(f"    in raw:          {'YES' if in_raw else 'NO'}  |  "
      f"subj seen: {'YES' if subj_in_raw else 'NO'}  |  "
      f"obj seen:  {'YES' if obj_in_raw else 'NO'}")
    p(f"    in verified:     {'YES' if in_ver else 'NO'}  |  "
      f"in rejected:  {'YES' if in_rej else 'NO'}")
    p(f"    in canonical:    {'YES' if in_can else 'NO'}  |  "
      f"in final KG:  {'YES' if in_kg else 'NO'}")
    p(f"    ROOT CAUSE: {cause}")

    # If subject extracted but not this relation — show what WAS extracted
    if subj_in_raw and not in_raw:
        extracted_for_subj = [(rel(t), obj(t)) for t in all_raw
                              if rs in subj(t) or subj(t) in rs][:5]
        p(f"    What WAS extracted for '{ref['subject']}':")
        for r_, o_ in extracted_for_subj:
            p(f"      --[{r_}]--> {o_}")

# ══════════════════════════════════════════════════════════════════════
# 5. QUERY STRUCTURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════
hdr("5. QUERY STRUCTURE — What do the 249 queries look like?")

if queries:
    p(f"Total queries loaded: {len(queries)}")
    p(f"Keys in first query: {list(queries[0].keys())}")
    p(f"\nSample of 5 queries:")
    for q in queries[:5]:
        p(f"  {json.dumps(q)}")

    # Find the subject field
    for field in ["subject", "entity", "geo_object", "object",
                  "geological_object", "query_subject"]:
        vals = Counter(q.get(field, "") for q in queries if q.get(field))
        if vals and len(vals) > 1:
            p(f"\nField [{field}]: {len(vals)} unique values")
            for v, c in vals.most_common(15):
                p(f"  {v}: {c}")

    # Show what strategies are covered
    for field in ["strategy", "type", "query_type", "extraction_strategy"]:
        vals = Counter(q.get(field, "") for q in queries if q.get(field))
        if vals:
            p(f"\nField [{field}]: {dict(vals)}")
else:
    p("No queries loaded — check file format")
    if QUERIES.exists():
        p("First 300 chars of file:")
        p(open(QUERIES).read()[:300])

# ══════════════════════════════════════════════════════════════════════
# 6. DEEP DIVE — Why is each non-corpus-absent edge never extracted?
# ══════════════════════════════════════════════════════════════════════
hdr("6. DEEP DIVE — QUERY_MISS analysis for key edges")

QUERY_MISS_EDGES = [
    ("turbidite",         "hasDescriptor", "high-amplitude",      31),
    ("debris flow",       "hasDescriptor", "chaotic",             14),
    ("slide",             "hasDescriptor", "undeformed",          54),
    ("slope failure",     "causes",        "mass transport deposit", 44),
    ("turbidity current", "formedBy",      "debris flow",         69),
    ("turbidite",         "occursIn",      "basin floor",         46),
    ("debris flow",       "occursIn",      "continental slope",   32),
    ("slide",             "overlies",      "hemipelagite",        18),
    ("slide",             "hasDescriptor", "blocky",              11),
    ("debris flow",       "hasDescriptor", "hummocky",             4),
]

for rs, rr, ro, n_chunks in QUERY_MISS_EDGES:
    sub(f"{rs} --[{rr}]--> {ro}  ({n_chunks} chunks)")

    # 1. Does a query exist for this subject+relation?
    has_query = False
    matching_queries = []
    for q in queries:
        q_text = json.dumps(q).lower()
        if rs in q_text and (rr in q_text or ro in q_text):
            has_query = True
            matching_queries.append(q)
    p(f"  Query exists for this edge? {'YES — ' + str(len(matching_queries)) + ' queries' if has_query else 'NO QUERY FOUND'}")
    if matching_queries:
        p(f"  Sample query: {json.dumps(matching_queries[0])[:200]}")

    # 2. What does the raw extraction produce for this subject?
    raw_for_subj = [(rel(t), obj(t), verdict(t))
                    for t in all_raw
                    if rs in subj(t) or subj(t) in rs]
    p(f"  Raw triples with subject '{rs}': {len(raw_for_subj)}")
    for r_, o_, v_ in raw_for_subj[:5]:
        p(f"    [{v_}] --[{r_}]--> {o_}")
    if not raw_for_subj:
        p(f"    !! Subject '{rs}' NEVER appears as subject in raw extraction")

    # 3. Show a corpus chunk that contains both terms
    good_chunks = [c for c in chunks
                   if rs in chunk_text(c) and ro in chunk_text(c)]
    if good_chunks:
        best_text = (good_chunks[0].get("text") or good_chunks[0].get("content", ""))
        # Find the sentence(s) with both terms
        sentences = re.split(r'[.!?]\s+', best_text)
        best_sent = [s for s in sentences
                     if rs in s.lower() and ro in s.lower()]
        if best_sent:
            p(f"  Best corpus sentence: \"{best_sent[0][:300]}\"")
        else:
            p(f"  Best corpus chunk (first 300): {best_text[:300]}")
    
    # 4. Why might BM25 miss this chunk?
    # Check if query terms would match the chunk vocabulary
    if matching_queries:
        query_text = json.dumps(matching_queries[0]).lower()
        overlap_terms = [w for w in query_text.split()
                        if len(w) > 4 and
                        any(w in chunk_text(c) for c in good_chunks[:3])]
        p(f"  BM25 overlap terms (query∩chunk): {overlap_terms[:10]}")

# ══════════════════════════════════════════════════════════════════════
# 7. WHAT DOES THE LLM ACTUALLY EXTRACT FOR MISSING SUBJECTS?
# ══════════════════════════════════════════════════════════════════════
hdr("7. LLM OUTPUT — What does the model extract for missing subjects?")

MISSING_SUBJECTS = ["turbidite", "debris flow", "slide",
                    "hemipelagite", "slope failure", "turbidity current"]

for ms in MISSING_SUBJECTS:
    sub(f"Subject: '{ms}'")
    raw_t  = [(rel(t), obj(t), verdict(t)) for t in all_raw
              if ms in subj(t)]
    ver_t  = [(rel(t), obj(t), verdict(t)) for t in all_ver
              if ms in subj(t)]
    can_t  = [(rel(t), obj(t)) for t in all_can if ms in subj(t)]
    rej_t  = [(rel(t), obj(t), verdict(t)) for t in all_rej
              if ms in subj(t)]
    kg_t   = [(rel(t), obj(t), t.get("tier")) for t in final_kg
              if ms in subj(t)]

    p(f"  raw={len(raw_t)}  verified={len(ver_t)}  "
      f"canonical={len(can_t)}  rejected={len(rej_t)}  kg={len(kg_t)}")
    if raw_t:
        p(f"  Raw extractions:")
        for r_, o_, v_ in raw_t[:8]:
            p(f"    [{v_:<15}] --[{r_}]--> {o_}")
    if rej_t:
        p(f"  Rejected triples:")
        for r_, o_, v_ in rej_t[:5]:
            p(f"    [{v_:<15}] --[{r_}]--> {o_}")
    if kg_t:
        p(f"  In final KG (T{kg_t[0][2] if kg_t else '?'}):")
        for r_, o_, tier_ in kg_t[:5]:
            p(f"    [T{tier_}] --[{r_}]--> {o_}")

# ══════════════════════════════════════════════════════════════════════
# 8. CANONICALIZATION — What gets rejected and why?
# ══════════════════════════════════════════════════════════════════════
hdr("8. CANONICALIZATION — Rejection reasons")

all_rej_combined = rej_a + rej_b
if all_rej_combined:
    # Verdict distribution in rejected
    rej_verdict = Counter(verdict(t) for t in all_rej_combined)
    p(f"Rejected triples total: {len(all_rej_combined)}")
    for v, cnt in sorted(rej_verdict.items(), key=lambda x: -x[1]):
        p(f"  {v:<25} {cnt:>5} ({cnt/len(all_rej_combined)*100:.1f}%)")

    # Relation distribution in rejected
    sub("Relations in rejected triples")
    rej_rel = Counter(rel(t) for t in all_rej_combined)
    for r_, cnt in sorted(rej_rel.items(), key=lambda x: -x[1])[:15]:
        p(f"  {r_:<25} {cnt}")

    # Subject distribution in rejected
    sub("Top 15 subjects in rejected triples")
    rej_subj = Counter(subj(t) for t in all_rej_combined)
    for s_, cnt in rej_subj.most_common(15):
        p(f"  {s_:<45} {cnt}")

    # Sample of rejected triples for key subjects
    sub("Rejected triples for missing-subject entities")
    for ms in ["turbidite", "debris flow", "slide", "hemipelagite"]:
        rej_for_ms = [(rel(t), obj(t), verdict(t)) for t in all_rej_combined
                      if ms in subj(t)]
        if rej_for_ms:
            p(f"\n  '{ms}' rejected ({len(rej_for_ms)} total):")
            for r_, o_, v_ in rej_for_ms[:5]:
                p(f"    [{v_}] --[{r_}]--> {o_}")

# ══════════════════════════════════════════════════════════════════════
# 9. SYNTHESIS
# ══════════════════════════════════════════════════════════════════════
hdr("9. SYNTHESIS — Root causes and recommended actions")

p("ROOT CAUSE COUNTS:")
for cause, cnt in root_cause_counts.most_common():
    p(f"  {cause:<45} {cnt}")

p()
fixable = sum(v for k, v in root_cause_counts.items()
              if k != "CORPUS_ABSENT")
hard_ceiling = root_cause_counts.get("CORPUS_ABSENT", 0)
max_recall = (len(matched) + fixable) / 26 * 100

p(f"Matched:          {len(matched)}/26 = {len(matched)/26*100:.1f}%")
p(f"Fixable:          {fixable}/16 missing edges")
p(f"Hard ceiling:     {hard_ceiling}/16 (corpus absent)")
p(f"Max possible recall if all fixed: ({len(matched)}+{fixable})/26 = {max_recall:.1f}%")

p()
p("=" * 60)
p("WHAT TO DO NEXT — ordered by expected recall gain")
p("=" * 60)

never_subj = root_cause_counts.get("SUBJECT_NEVER_EXTRACTED", 0)
rel_missed  = root_cause_counts.get("RELATION_MISSED — subject extracted but not with this object/relation", 0)
corpus_abs  = root_cause_counts.get("CORPUS_ABSENT", 0)

if never_subj > 0:
    p(f"""
[FIX 1] SUBJECT NEVER EXTRACTED — {never_subj} edges
  Problem: The LLM never outputs 'turbidite', 'debris flow', 'slide',
           'hemipelagite', 'slope failure', 'turbidity current' as subject.
  
  Diagnosis path: Check if queries send these subjects to the LLM.
  In your extraction prompt, is the subject injected in the query?
  
  Fix options:
    A) Add explicit subject injection: "For the geological object 
       '{subject}', extract all hasDescriptor triples..."
    B) Check if rescue queries (P4) actually generate queries with 
       these subjects as the focal object
    C) Add few-shot examples where these specific subjects are 
       the triple subject, not just mentioned in context
""")

p(f"""
[FIX 2] PROMPTS — Add BAD EXAMPLES for the specific missed patterns
  The model extracts 'mass transport deposit' well but ignores 
  co-occurring objects in the same chunk. 
  
  Action: For each QUERY_MISS edge, manually inspect the best chunk
  (printed in Section 6), then add a few-shot example using that 
  exact chunk showing how to extract the missed triple.
  
  Priority edges (most corpus evidence):
    turbidity current --[formedBy]--> debris flow  (69 chunks)
    slide             --[hasDescriptor]--> undeformed (54 chunks)
    turbidite         --[occursIn]--> basin floor  (46 chunks)
    slope failure     --[causes]--> mass transport deposit (44 chunks)
""")

p(f"""
[FIX 3] BM25 QUERY TERMS — Check if rescue queries retrieve the right chunks
  The rescue queries may exist but BM25 may not retrieve the chunks 
  that contain both subject+relation+object together.
  
  Action: For each QUERY_MISS edge, run BM25 manually with the query
  and check if any of the 'good chunks' (from Section 6) are in top-3.
  If not: rephrase query terms to match corpus vocabulary.
  
  Test command:
    python3 -c "
    import sys; sys.path.insert(0,'.')
    from pipeline.rag.bm25_index import load_index
    idx = load_index('/home/talbi/kg_test/output/step1/')
    results = idx.search('turbidite high-amplitude seismic facies', k=5)
    for r in results: print(r['score'], r['text'][:200])
    "
""")

p(f"""
[FIX 4] CORPUS ABSENT — {corpus_abs} edges (hard limit)
  massive, turbidite→layered, hemipelagite→low-amplitude
  have zero corpus co-occurrence. Only corpus expansion fixes this.
  Note for paper: frame as corpus limitation, not pipeline failure.
""")

# ── Save ───────────────────────────────────────────────────────────────
out_txt  = REPO / "output/diagnostic_report_v2.txt"
out_json = REPO / "output/diagnostic_data_v2.json"

out_txt.write_text("\n".join(lines), encoding="utf-8")
out_json.write_text(json.dumps({
    "funnel": {
        "raw_a": len(raw_a), "raw_b": len(raw_b),
        "ver_a": len(ver_a), "ver_b": len(ver_b),
        "can_a": len(can_a), "can_b": len(can_b),
        "rej_a": len(rej_a), "rej_b": len(rej_b),
        "final_kg": len(final_kg)
    },
    "matched": len(matched),
    "missing": len(missing),
    "root_causes": dict(root_cause_counts),
    "fixable": fixable,
    "hard_ceiling": hard_ceiling,
    "max_possible_recall_pct": round(max_recall, 1)
}, indent=2), encoding="utf-8")

print("\n".join(lines))
print(f"\n✓ Report: {out_txt}")
print(f"✓ Data:   {out_json}")