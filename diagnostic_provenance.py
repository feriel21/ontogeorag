#!/usr/bin/env python3
"""
diagnostic_provenance.py
========================
Diagnose why 29/100 Tier-1 triples have no retrievable evidence in Exp D.
Also:
  - Computes RAG net contribution (C10 edges not in Exp B)
  - Verifies T1 regression claim (C9→C10)
  - Computes inter-model kappa (Qwen vs Llama on 71 triples)

Run: python3 diagnostic_provenance.py
"""

import json
import re
import math
from pathlib import Path
from collections import Counter

# ── paths ─────────────────────────────────────────────────────────────
KG_C10   = Path('output/run11_kg/tiered_kg_run11.json')
KG_C9    = Path('output/run10_kg/tiered_kg_run10_final.json')
EXPD     = Path('output/expD/results_expD.jsonl')
EXPB     = Path('output/expB/canonical_expB.jsonl')
REF      = Path('configs/lb_reference_edges.json')
CHUNKS   = Path('/home/talbi/kg_test/output/step1/chunks.jsonl')

def norm(s):
    return re.sub(r'\s+', ' ', (s or '').lower().strip()).rstrip('.,;:')

def load_kg(path):
    kg = json.load(open(path))
    return kg.get('triples', kg) if isinstance(kg, dict) else kg

def load_ref():
    ref = json.load(open(REF))
    return ref.get('edges', ref) if isinstance(ref, dict) else ref

# ══════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 1 — Why 29 triples have no evidence
# ══════════════════════════════════════════════════════════════════════
def diag_provenance():
    print('\n' + '='*65)
    print('DIAGNOSTIC 1 — PROVENANCE OF 29 NO-EVIDENCE TRIPLES')
    print('='*65)

    triples = load_kg(KG_C10)
    tier1   = [t for t in triples if t.get('tier') == 1]

    # Check what provenance fields exist
    has_evidence_quote = sum(1 for t in tier1 if t.get('evidence_quote','').strip())
    has_chunk_id       = sum(1 for t in tier1 if t.get('_chunk_id') or t.get('source_chunk'))
    has_supporting     = sum(1 for t in tier1 if t.get('supporting_sentence','').strip())
    has_papers         = sum(1 for t in tier1 if t.get('supporting_papers'))
    has_any            = sum(1 for t in tier1 if
                             t.get('evidence_quote','').strip() or
                             t.get('_chunk_id') or
                             t.get('source_chunk') or
                             t.get('supporting_sentence','').strip())

    print(f'\nTier-1 total triples : {len(tier1)}')
    print(f'Has evidence_quote   : {has_evidence_quote} ({has_evidence_quote/len(tier1)*100:.0f}%)')
    print(f'Has chunk_id         : {has_chunk_id} ({has_chunk_id/len(tier1)*100:.0f}%)')
    print(f'Has supporting_sent  : {has_supporting} ({has_supporting/len(tier1)*100:.0f}%)')
    print(f'Has supporting_papers: {has_papers} ({has_papers/len(tier1)*100:.0f}%)')
    print(f'Has ANY evidence     : {has_any} ({has_any/len(tier1)*100:.0f}%)')
    print(f'Missing ALL evidence : {len(tier1)-has_any} ({(len(tier1)-has_any)/len(tier1)*100:.0f}%)')

    # Sample triples with no evidence to understand why
    no_ev = [t for t in tier1 if not (
        t.get('evidence_quote','').strip() or
        t.get('_chunk_id') or
        t.get('source_chunk') or
        t.get('supporting_sentence','').strip()
    )]

    print(f'\nSample triples with NO evidence fields ({min(5,len(no_ev))} shown):')
    for t in no_ev[:5]:
        subj = t.get('subject') or t.get('source','')
        rel  = t.get('relation','')
        obj  = t.get('object')  or t.get('target','')
        keys = list(t.keys())
        print(f'  ({subj}, {rel}, {obj})')
        print(f'  Keys: {keys}')

    # Check if chunks file exists and has text-search matches
    if CHUNKS.exists():
        chunks = [json.loads(l) for l in open(CHUNKS) if l.strip()]
        print(f'\nChunks available: {len(chunks)}')
        print(f'Text-search fallback possible: YES')

        # For each no-evidence triple, check if text search finds it
        found_via_search = 0
        for t in no_ev:
            subj = norm(t.get('subject') or t.get('source',''))
            obj  = norm(t.get('object')  or t.get('target',''))
            hits = [c for c in chunks
                    if subj in c.get('text','').lower()
                    and obj  in c.get('text','').lower()]
            if hits:
                found_via_search += 1
        print(f'Found via text search  : {found_via_search}/{len(no_ev)} '
              f'({found_via_search/max(len(no_ev),1)*100:.0f}%)')
    else:
        print(f'\nChunks file not found at {CHUNKS}')
        print('Text-search fallback: NOT available')

    return no_ev


# ══════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 2 — RAG net contribution (C10 minus Exp B)
# ══════════════════════════════════════════════════════════════════════
def diag_rag_contribution():
    print('\n' + '='*65)
    print('DIAGNOSTIC 2 — RAG NET CONTRIBUTION')
    print('='*65)

    ref     = load_ref()
    c10     = load_kg(KG_C10)
    expb    = [json.loads(l) for l in open(EXPB) if l.strip()]

    def matched_edges(triples, ref_edges):
        matched = []
        for r in ref_edges:
            rs = norm(r.get('subject',''))
            ro = norm(r.get('object',''))
            rr = r.get('relation','').strip().lower()
            for t in triples:
                ts = norm(t.get('subject','') or t.get('source',''))
                to = norm(t.get('object','')  or t.get('target',''))
                tr = t.get('relation','').strip().lower()
                if rs in ts and ro in to and rr == tr:
                    matched.append(r)
                    break
        return matched

    c10_matched  = matched_edges(c10, ref)
    expb_matched = matched_edges(expb, ref)

    c10_set  = {(norm(e['subject']), e['relation'], norm(e['object']))
                for e in c10_matched}
    expb_set = {(norm(e['subject']), e['relation'], norm(e['object']))
                for e in expb_matched}

    rag_only   = c10_set - expb_set   # edges that RAG adds over parametric memory
    param_only = expb_set - c10_set   # edges in Exp B but not in C10 (regressions)
    both       = c10_set & expb_set   # edges in both

    print(f'\nC10 matched edges      : {len(c10_set)}/26 = {len(c10_set)/26*100:.1f}%')
    print(f'Exp B matched edges    : {len(expb_set)}/26 = {len(expb_set)/26*100:.1f}%')
    print(f'Both (parametric+RAG)  : {len(both)}')
    print(f'RAG-only (net new)     : {len(rag_only)}  ← pure RAG contribution')
    print(f'Exp B only (regression): {len(param_only)}  ← lost in C10 vs Exp B')

    print(f'\nRAG-only edges (C10 adds over parametric memory):')
    for s, r, o in sorted(rag_only):
        print(f'  ✓  {s:35s} --[{r}]--> {o}')

    print(f'\nShared edges (Exp B + C10 both find):')
    for s, r, o in sorted(both):
        print(f'  =  {s:35s} --[{r}]--> {o}')

    if param_only:
        print(f'\nExp B finds but C10 misses (regressions):')
        for s, r, o in sorted(param_only):
            print(f'  ✗  {s:35s} --[{r}]--> {o}')

    print(f'\nSUMMARY FOR PAPER:')
    print(f'  Total C10 recall     : {len(c10_set)}/26 = {len(c10_set)/26*100:.1f}%')
    print(f'  Parametric baseline  : {len(expb_set)}/26 = {len(expb_set)/26*100:.1f}%')
    print(f'  Net RAG contribution : {len(rag_only)}/26 = {len(rag_only)/26*100:.1f}%')
    print(f'  Shared (both)        : {len(both)}/26 = {len(both)/26*100:.1f}%')

    return rag_only, both, param_only


# ══════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 3 — Inter-model kappa (Qwen vs Llama)
# ══════════════════════════════════════════════════════════════════════
def diag_kappa():
    print('\n' + '='*65)
    print('DIAGNOSTIC 3 — INTER-MODEL KAPPA (Qwen vs Llama)')
    print('='*65)

    results = [json.loads(l) for l in open(EXPD) if l.strip()]
    # Only use triples where Llama had evidence
    evaluated = [r for r in results if r.get('verdict_llama') not in
                 ('NO_EVIDENCE', 'UNRESOLVED', None)]

    print(f'\nTriples with Llama verdict: {len(evaluated)}/100')

    # Binarize: SUPPORTED = STRONG + WEAK, REJECTED = NOT_SUPPORTED
    def binarize(v):
        if v in ('STRONG_SUPPORT', 'WEAK_SUPPORT'):
            return 'SUPPORTED'
        return 'NOT_SUPPORTED'

    qwen_labels  = [binarize(r['verdict_qwen'])  for r in evaluated]
    llama_labels = [binarize(r['verdict_llama']) for r in evaluated]

    n = len(evaluated)
    agree = sum(q == l for q, l in zip(qwen_labels, llama_labels))
    p_o = agree / n  # observed agreement

    # Expected agreement for Cohen's kappa
    q_sup  = qwen_labels.count('SUPPORTED')  / n
    q_ns   = qwen_labels.count('NOT_SUPPORTED') / n
    l_sup  = llama_labels.count('SUPPORTED') / n
    l_ns   = llama_labels.count('NOT_SUPPORTED') / n
    p_e = q_sup * l_sup + q_ns * l_ns

    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0

    # Confusion matrix
    tp = sum(q == 'SUPPORTED'     and l == 'SUPPORTED'     for q, l in zip(qwen_labels, llama_labels))
    tn = sum(q == 'NOT_SUPPORTED' and l == 'NOT_SUPPORTED' for q, l in zip(qwen_labels, llama_labels))
    fp = sum(q == 'SUPPORTED'     and l == 'NOT_SUPPORTED' for q, l in zip(qwen_labels, llama_labels))
    fn = sum(q == 'NOT_SUPPORTED' and l == 'SUPPORTED'     for q, l in zip(qwen_labels, llama_labels))

    print(f'\nConfusion matrix (rows=Qwen, cols=Llama):')
    print(f'            Llama:SUP  Llama:NS')
    print(f'  Qwen:SUP    {tp:4d}      {fp:4d}')
    print(f'  Qwen:NS     {fn:4d}      {tn:4d}')
    print(f'\nObserved agreement p_o : {p_o:.3f} ({agree}/{n})')
    print(f'Expected agreement p_e : {p_e:.3f}')
    print(f'Cohen\'s kappa          : {kappa:.3f}')

    if kappa < 0.2:
        interp = 'Slight (very poor)'
    elif kappa < 0.4:
        interp = 'Fair'
    elif kappa < 0.6:
        interp = 'Moderate'
    elif kappa < 0.8:
        interp = 'Substantial'
    else:
        interp = 'Almost perfect'
    print(f'Interpretation         : {interp}')

    print(f'\nFOR PAPER:')
    print(f'  Qwen-Llama inter-model κ = {kappa:.2f} ({interp.lower()})')
    print(f'  Qwen H_T1 = 0%,  Llama H_T1 = {fp}/{n} = {fp/n*100:.1f}%')
    print(f'  The gap ({fp/n*100:.1f}pp) reflects partial self-verification bias')

    return kappa, fp/n*100


# ══════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 4 — T1 regression: verify "redistribution" claim
# ══════════════════════════════════════════════════════════════════════
def diag_t1_regression():
    print('\n' + '='*65)
    print('DIAGNOSTIC 4 — T1 REGRESSION C9→C10')
    print('='*65)

    ref = load_ref()

    def matched_t1(kg_path, ref_edges):
        triples = load_kg(kg_path)
        tier1   = [t for t in triples if t.get('tier') == 1]
        matched = []
        for r in ref_edges:
            rs = norm(r.get('subject',''))
            ro = norm(r.get('object',''))
            rr = r.get('relation','').strip().lower()
            for t in tier1:
                ts = norm(t.get('subject','') or t.get('source',''))
                to = norm(t.get('object','')  or t.get('target',''))
                tr = t.get('relation','').strip().lower()
                if rs in ts and ro in to and rr == tr:
                    matched.append((rs, rr, ro))
                    break
        return set(matched)

    def matched_t12(kg_path, ref_edges):
        triples = load_kg(kg_path)
        matched = []
        for r in ref_edges:
            rs = norm(r.get('subject',''))
            ro = norm(r.get('object',''))
            rr = r.get('relation','').strip().lower()
            for t in triples:
                ts = norm(t.get('subject','') or t.get('source',''))
                to = norm(t.get('object','')  or t.get('target',''))
                tr = t.get('relation','').strip().lower()
                if rs in ts and ro in to and rr == tr:
                    matched.append((rs, rr, ro))
                    break
        return set(matched)

    c9_t1   = matched_t1(KG_C9,  ref)
    c10_t1  = matched_t1(KG_C10, ref)
    c9_t12  = matched_t12(KG_C9,  ref)
    c10_t12 = matched_t12(KG_C10, ref)

    lost_t1      = c9_t1  - c10_t1   # in C9 T1 but not in C10 T1
    gained_t1    = c10_t1 - c9_t1    # in C10 T1 but not in C9 T1
    redistributed = lost_t1 & c10_t12 # lost from T1 but still in T1+2 of C10

    print(f'\nC9  Tier-1 matched : {len(c9_t1)}/26')
    print(f'C10 Tier-1 matched : {len(c10_t1)}/26')
    print(f'Lost from T1 (C9→C10) : {len(lost_t1)}')
    print(f'  Of which still in C10 T1+2: {len(redistributed)} ← confirmed redistribution')
    print(f'  Of which truly lost       : {len(lost_t1) - len(redistributed)} ← actual regression')
    print(f'Gained in T1 (C9→C10) : {len(gained_t1)}')

    if lost_t1:
        print(f'\nEdges lost from T1:')
        for s, r, o in sorted(lost_t1):
            status = 'still in T1+2' if (s,r,o) in redistributed else 'TRULY LOST'
            print(f'  ({s}, {r}, {o}) → {status}')

    print(f'\nCONCLUSION:')
    if len(redistributed) == len(lost_t1):
        print(f'  ✅ T1 regression is PURE redistribution — no edges truly lost')
        print(f'     Claim "redistribution T1→T2" is CONFIRMED')
    else:
        truly_lost = len(lost_t1) - len(redistributed)
        print(f'  ⚠️  {truly_lost} edges truly lost from KG (not in C10 T1+2)')
        print(f'     Revise claim — not pure redistribution')


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('OntoGeoRAG — Full Provenance & Contribution Diagnostic')
    print('='*65)

    no_ev = diag_provenance()
    rag_only, both, param_only = diag_rag_contribution()
    kappa, ns_gap = diag_kappa()
    diag_t1_regression()

    print('\n' + '='*65)
    print('SUMMARY FOR PAPER REVISION')
    print('='*65)
    print(f'  Point 1 (provenance) : see Diagnostic 1 above')
    print(f'  Point 2 (RAG net)    : {len(rag_only)} new edges from RAG')
    print(f'  Point 3 (kappa)      : κ = {kappa:.2f} Qwen-Llama')
    print(f'  Point 4 (T1 regress) : see Diagnostic 4 above')