#!/usr/bin/env python3
"""
Experiment B — No-RAG Baseline
================================
Same LLM (Qwen2.5-7B-Instruct), same queries, same prompts.
NO BM25, NO chunks, NO retrieval context.
LLM answers from parametric memory only.

This establishes the baseline recall to show what RAG adds.

Usage:
    python pipeline/expB_no_rag.py \
        --queries configs/descriptor_queries.jsonl \
        --ref     configs/lb_reference_edges.json \
        --output  output/expB/ \
        --model   Qwen/Qwen2.5-7B-Instruct \
        --device  cuda

Output:
    output/expB/raw_triples_expB.jsonl
    output/expB/canonical_expB.jsonl
    output/expB/metrics_expB.json
    output/expB/report_expB.txt
"""

import argparse
import json
import os
import re
import time
import logging
from collections import defaultdict, Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ── constants (same as pipeline) ──────────────────────────────────────
ALLOWED_RELATIONS = {
    'hasDescriptor', 'occursIn', 'causes', 'triggers',
    'controls', 'formedBy', 'overlies', 'associatedWith', 'affects'
}

DESCRIPTOR_SYSTEM = """You are a geological knowledge extractor specializing in seismic facies and mass transport deposits.

Extract triples of the form (subject, relation, object) from YOUR GEOLOGICAL KNOWLEDGE.
You are NOT given any text passage. Use only well-established geological facts.

Allowed relations: hasDescriptor, occursIn, causes, triggers, controls, formedBy, overlies, associatedWith, affects

Rules:
- Only output triples you are confident are geologically established facts
- Do NOT invent relations
- Output valid JSON list only

Output format: JSON list of objects with keys: subject, relation, object, confidence (high/medium/low)
If you cannot extract any confident triple, output: []"""

DESCRIPTOR_USER = """Geological object: {subject}
Query: {query}

Extract geological triples involving "{subject}" from your knowledge.
Focus on: seismic descriptors, depositional settings, causal relationships.

Output JSON only:"""


def load_model(model_name, device):
    log.info(f'Loading model: {model_name}')
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    mdl.eval()
    log.info('Model loaded.')
    return tok, mdl


def generate(tok, mdl, system, user, max_new_tokens=512, temperature=0.0):
    messages = [
        {'role': 'system',  'content': system},
        {'role': 'user',    'content': user},
    ]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors='pt').to(mdl.device)

    with torch.no_grad():
        if temperature == 0.0:
            out = mdl.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tok.eos_token_id
            )
        else:
            out = mdl.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tok.eos_token_id
            )

    new_tokens = out[0][inputs['input_ids'].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def parse_json(text):
    """Extract JSON list from LLM output."""
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try extracting JSON block
    m = re.search(r'\[.*?\]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return []


def normalize_relation(rel):
    if not rel:
        return ''
    rel = rel.strip()
    # Map common variants
    mapping = {
        'has_descriptor':   'hasDescriptor',
        'has descriptor':   'hasDescriptor',
        'occurs_in':        'occursIn',
        'occurs in':        'occursIn',
        'formed_by':        'formedBy',
        'formed by':        'formedBy',
        'associated_with':  'associatedWith',
        'associated with':  'associatedWith',
    }
    return mapping.get(rel.lower(), rel)


def normalize_text(s):
    return re.sub(r'\s+', ' ', (s or '').lower().strip()).rstrip('.,;:')


def run_expB(queries, tok, mdl, output_dir, temperature=0.0):
    """Run no-RAG extraction for all queries."""
    raw_path = output_dir / 'raw_triples_expB.jsonl'
    all_triples = []
    silent = 0

    log.info(f'Running Exp B on {len(queries)} queries (temp={temperature})')

    with open(raw_path, 'w') as fout:
        for i, q in enumerate(queries):
            subject  = q.get('subject', q.get('object', ''))
            query    = q.get('query', '')
            strategy = q.get('strategy', 'descriptor')

            user_prompt = DESCRIPTOR_USER.format(
                subject=subject, query=query)

            t0  = time.time()
            raw = generate(tok, mdl, DESCRIPTOR_SYSTEM, user_prompt,
                           temperature=temperature)
            elapsed = time.time() - t0

            triples = parse_json(raw)

            # Filter to allowed relations
            valid = []
            for t in triples:
                rel = normalize_relation(t.get('relation', ''))
                if rel in ALLOWED_RELATIONS:
                    valid.append({
                        'subject':  normalize_text(t.get('subject', '')),
                        'relation': rel,
                        'object':   normalize_text(t.get('object', '')),
                        'confidence': t.get('confidence', 'medium'),
                        '_query':   query,
                        '_subject': subject,
                        '_strategy': strategy,
                        '_source':  'expB_no_rag',
                        '_elapsed': round(elapsed, 2),
                    })

            if not valid:
                silent += 1

            all_triples.extend(valid)

            for t in valid:
                fout.write(json.dumps(t) + '\n')

            if (i+1) % 25 == 0:
                log.info(f'  [{i+1}/{len(queries)}]  '
                         f'triples so far: {len(all_triples)}  '
                         f'silent: {silent}')

    log.info(f'Exp B done. Total raw: {len(all_triples)}, '
             f'silent: {silent}/{len(queries)} '
             f'({silent/len(queries)*100:.1f}%)')
    return all_triples, silent


def canonicalize(triples):
    """Simple deduplication — no SciBERT (keep it fast)."""
    seen = {}
    for t in triples:
        key = (t['subject'], t['relation'], t['object'])
        if key not in seen:
            seen[key] = t
        else:
            # prefer high confidence
            if t.get('confidence') == 'high':
                seen[key] = t
    return list(seen.values())


def evaluate(canonical, ref_edges):
    """Compute recall against LB2019 benchmark."""
    def norm(s): return re.sub(r'\s+', ' ',
                                (s or '').lower().strip()).rstrip('.,;:')

    matched, unmatched = [], []
    for r in ref_edges:
        rs = norm(r.get('subject', ''))
        ro = norm(r.get('object', ''))
        rr = normalize_relation(r.get('relation', ''))
        found = False
        for t in canonical:
            ts = norm(t.get('subject', ''))
            to = norm(t.get('object', ''))
            tr = normalize_relation(t.get('relation', ''))
            if rs in ts and ro in to and rr == tr:
                found = True
                break
        if found:
            matched.append(r)
        else:
            unmatched.append(r)

    return matched, unmatched


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queries', default='configs/descriptor_queries.jsonl')
    p.add_argument('--ref',     default='configs/lb_reference_edges.json')
    p.add_argument('--output',  default='output/expB')
    p.add_argument('--model',   default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--device',  default='cuda')
    p.add_argument('--temperature', type=float, default=0.0)
    args = p.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load queries
    queries = [json.loads(l) for l in open(args.queries) if l.strip()]
    log.info(f'Loaded {len(queries)} queries')

    # Load reference
    ref = json.load(open(args.ref))
    ref = ref.get('edges', ref) if isinstance(ref, dict) else ref
    log.info(f'Loaded {len(ref)} reference edges')

    # Load model
    tok, mdl = load_model(args.model, args.device)

    # Run Exp B
    t_start = time.time()
    raw_triples, n_silent = run_expB(
        queries, tok, mdl, output_dir, args.temperature)
    wall = time.time() - t_start

    # Canonicalize
    canonical = canonicalize(raw_triples)
    log.info(f'Canonical triples: {len(canonical)}')

    # Save canonical
    canon_path = output_dir / 'canonical_expB.jsonl'
    with open(canon_path, 'w') as f:
        for t in canonical:
            f.write(json.dumps(t) + '\n')

    # Evaluate
    matched, unmatched = evaluate(canonical, ref)
    recall = len(matched) / len(ref) * 100

    # Relation distribution
    rel_dist = Counter(t['relation'] for t in canonical)

    # Metrics
    metrics = {
        'experiment':     'B_no_rag',
        'n_queries':      len(queries),
        'n_raw':          len(raw_triples),
        'n_canonical':    len(canonical),
        'n_silent':       n_silent,
        'silent_pct':     round(n_silent / len(queries) * 100, 1),
        'n_matched':      len(matched),
        'n_ref':          len(ref),
        'recall_pct':     round(recall, 1),
        'relation_dist':  dict(rel_dist),
        'wall_seconds':   round(wall, 1),
        'temperature':    args.temperature,
        'model':          args.model,
    }

    with open(output_dir / 'metrics_expB.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Text report
    report_lines = [
        '=' * 60,
        'EXPERIMENT B — NO-RAG BASELINE',
        '=' * 60,
        f'Model:        {args.model}',
        f'Temperature:  {args.temperature}',
        f'Wall time:    {wall/60:.1f} min',
        '',
        'EXTRACTION',
        f'  Queries:    {len(queries)}',
        f'  Raw triples:{len(raw_triples)}',
        f'  Canonical:  {len(canonical)}',
        f'  Silent:     {n_silent}/{len(queries)} ({n_silent/len(queries)*100:.1f}%)',
        '',
        'EVALUATION vs LB2019 (26 edges)',
        f'  Matched:    {len(matched)}/26 = {recall:.1f}%',
        '',
        'MATCHED EDGES:',
    ]
    for e in matched:
        report_lines.append(
            f'  ✓  {e.get("subject"):30s} '
            f'--[{normalize_relation(e.get("relation"))}]-->  '
            f'{e.get("object")}')
    report_lines.append('\nUNMATCHED EDGES:')
    for e in unmatched:
        report_lines.append(
            f'  ✗  {e.get("subject"):30s} '
            f'--[{normalize_relation(e.get("relation"))}]-->  '
            f'{e.get("object")}')
    report_lines += [
        '',
        'RELATION DISTRIBUTION:',
    ]
    for rel, cnt in rel_dist.most_common():
        report_lines.append(f'  {rel:30s} {cnt}')
    report_lines += [
        '',
        'COMPARISON (for paper):',
        f'  Exp B  (no-RAG)          : {recall:.1f}%',
        f'  C9     (BM25 only)       : 50.0%',
        f'  C10    (BM25+reranker)   : 69.2%',
        f'  Delta RAG vs no-RAG      : +{50.0 - recall:.1f}pp (C9)',
        f'  Delta reranker vs no-RAG : +{69.2 - recall:.1f}pp (C10)',
        '=' * 60,
    ]

    report = '\n'.join(report_lines)
    with open(output_dir / 'report_expB.txt', 'w') as f:
        f.write(report)

    print(report)
    log.info(f'Results saved to {output_dir}')


if __name__ == '__main__':
    main()