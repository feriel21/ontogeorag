#!/usr/bin/env python3
"""
Experiment D — Cross-Model Verifier
=====================================
Verifies 100 Tier-1 triples from C10 using Llama-3.1-8B-Instruct
as an INDEPENDENT verifier (different from Qwen-7B extractor).

Goal: assess whether 0% NOT_SUPPORTED in Tier-1 is robust to
verifier model choice, or reflects self-verification bias.

Usage:
    python pipeline/expD_cross_model.py \
        --kg    output/run11_kg/tiered_kg_run11.json \
        --index output/step1/ \
        --output output/expD/ \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --n-triples 100 \
        --seed 42
"""

import argparse
import json
import os
import re
import random
import time
import logging
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ── Verification prompt (same structure as Qwen verifier) ─────────────
VERIFY_SYSTEM = """You are a scientific fact-checker for geological knowledge graphs.

Your task: determine whether a geological triple is supported by the provided evidence text.

You MUST assign one of these exact verdicts:
  STRONG_SUPPORT  — the text explicitly and directly states this relationship
  WEAK_SUPPORT    — the text implies or is consistent with this relationship
  NOT_SUPPORTED   — the text does NOT state, imply, or support this relationship

Do NOT use any other verdict. Do NOT use geological knowledge outside the provided text."""

VERIFY_USER = """Evidence text:
{evidence}

Triple to verify: ({subject}, {relation}, {object})

Step 1: Quote the most relevant sentence from the evidence.
Step 2: Reason: does this sentence support the triple?
Step 3: Output your verdict as one of: STRONG_SUPPORT / WEAK_SUPPORT / NOT_SUPPORTED

Format your response as:
Quote: <quote>
Reasoning: <reasoning>
Verdict: <STRONG_SUPPORT|WEAK_SUPPORT|NOT_SUPPORTED>"""


def load_model(model_name):
    log.info(f'Loading model: {model_name}')
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    mdl.eval()
    log.info('Model loaded.')
    return tok, mdl


def generate(tok, mdl, system, user, max_new_tokens=512):
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user',   'content': user},
    ]
    # Use chat template if available, else manual format
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"

    inputs = tok(text, return_tensors='pt').to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id
        )
    new_tokens = out[0][inputs['input_ids'].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def parse_verdict(text):
    """Extract verdict from model output."""
    text_upper = text.upper()
    # Search in order of strictness
    if 'NOT_SUPPORTED' in text_upper or 'NOT SUPPORTED' in text_upper:
        return 'NOT_SUPPORTED'
    if 'STRONG_SUPPORT' in text_upper or 'STRONG SUPPORT' in text_upper:
        return 'STRONG_SUPPORT'
    if 'WEAK_SUPPORT' in text_upper or 'WEAK SUPPORT' in text_upper:
        return 'WEAK_SUPPORT'
    # Fallback: look for verdict line
    for line in text.split('\n'):
        if line.strip().startswith('Verdict:'):
            val = line.split(':', 1)[1].strip().upper()
            if 'NOT' in val:
                return 'NOT_SUPPORTED'
            if 'STRONG' in val:
                return 'STRONG_SUPPORT'
            if 'WEAK' in val:
                return 'WEAK_SUPPORT'
    return 'UNRESOLVED'


def load_chunks(index_dir):
    """Load BM25 chunks for evidence retrieval."""
    chunks_path = Path(index_dir) / 'chunks.jsonl'
    if not chunks_path.exists():
        # Try alternate path used in the project
        chunks_path = Path('/home/talbi/kg_test/output/step1/chunks.jsonl')
    if not chunks_path.exists():
        log.warning(f'Chunks file not found at {chunks_path}')
        return []
    chunks = [json.loads(l) for l in open(chunks_path) if l.strip()]
    log.info(f'Loaded {len(chunks)} chunks from {chunks_path}')
    return chunks


def find_evidence(triple, chunks, max_chunks=3):
    """
    Find the best evidence chunks for a triple.
    Uses stored source_chunk if available, otherwise text search.
    """
    # 1. Use stored evidence from the triple itself
    # Field is named 'evidence' in the C10 KG (confirmed from tiered_kg_run11.json)
    stored = (triple.get('evidence') or
              triple.get('evidence_quote') or
              triple.get('supporting_sentence', ''))
    # Strip surrounding quotes if present
    if stored:
        stored = stored.strip().strip('"')
    if stored and len(stored) > 20:
        return stored[:1500]

    # 2. Use stored source chunk ID
    chunk_id = triple.get('_chunk_id') or triple.get('source_chunk')
    if chunk_id and chunks:
        for c in chunks:
            if c.get('chunk_id') == chunk_id or c.get('id') == chunk_id:
                return c.get('text', '')[:1500]

    # 3. Text search fallback
    if chunks:
        subj = (triple.get('subject') or triple.get('source', '')).lower()
        obj  = (triple.get('object')  or triple.get('target', '')).lower()
        hits = [c for c in chunks
                if subj in c.get('text', '').lower()
                and obj  in c.get('text', '').lower()]
        if hits:
            return hits[0].get('text', '')[:1500]

    # 4. Last resort: use stored provenance note
    note = triple.get('provenance_note', '')
    if note:
        return note[:1500]

    return ''


def sample_tier1_triples(kg_path, n, seed):
    """Sample n Tier-1 triples stratified by relation type."""
    kg = json.load(open(kg_path))
    triples = kg.get('triples', kg) if isinstance(kg, dict) else kg
    tier1 = [t for t in triples if t.get('tier') == 1]
    log.info(f'Total Tier-1 triples: {len(tier1)}')

    # Stratify by relation
    by_rel = {}
    for t in tier1:
        rel = t.get('relation', 'unknown')
        by_rel.setdefault(rel, []).append(t)

    log.info('Tier-1 relation distribution:')
    for rel, lst in sorted(by_rel.items(), key=lambda x: -len(x[1])):
        log.info(f'  {rel:30s} {len(lst)}')

    # Stratified sample proportional to relation frequency
    random.seed(seed)
    sampled = []
    total = len(tier1)
    for rel, lst in by_rel.items():
        k = max(1, round(n * len(lst) / total))
        sampled.extend(random.sample(lst, min(k, len(lst))))

    # Trim or top-up to exactly n
    random.shuffle(sampled)
    if len(sampled) > n:
        sampled = sampled[:n]
    elif len(sampled) < n:
        remaining = [t for t in tier1 if t not in sampled]
        sampled += random.sample(remaining, min(n - len(sampled), len(remaining)))

    log.info(f'Sampled {len(sampled)} Tier-1 triples for verification')
    return sampled


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kg',        default='output/run11_kg/tiered_kg_run11.json')
    p.add_argument('--index',     default='output/step1/')
    p.add_argument('--output',    default='output/expD')
    p.add_argument('--model',     default='meta-llama/Llama-3.1-8B-Instruct')
    p.add_argument('--n-triples', type=int, default=100)
    p.add_argument('--seed',      type=int, default=42)
    args = p.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    sampled = sample_tier1_triples(args.kg, args.n_triples, args.seed)
    chunks  = load_chunks(args.index)

    # Load Llama model
    tok, mdl = load_model(args.model)

    t_start = time.time()
    results = []
    verdict_counts = Counter()

    log.info(f'Verifying {len(sampled)} triples with {args.model}')

    for i, triple in enumerate(sampled):
        subj = triple.get('subject') or triple.get('source', '')
        rel  = triple.get('relation', '')
        obj  = triple.get('object')  or triple.get('target', '')

        evidence = find_evidence(triple, chunks)

        if not evidence:
            verdict = 'NO_EVIDENCE'
            raw_output = ''
            log.warning(f'  [{i+1}] No evidence found for ({subj}, {rel}, {obj})')
        else:
            user_prompt = VERIFY_USER.format(
                evidence=evidence[:1500],
                subject=subj, relation=rel, object=obj
            )
            t0  = time.time()
            raw = generate(tok, mdl, VERIFY_SYSTEM, user_prompt)
            elapsed = time.time() - t0
            verdict = parse_verdict(raw)
            raw_output = raw

        verdict_counts[verdict] += 1

        result = {
            'index':      i,
            'subject':    subj,
            'relation':   rel,
            'object':     obj,
            'tier':       triple.get('tier', 1),
            'verdict_llama': verdict,
            'verdict_qwen':  triple.get('verdict', 'STRONG_SUPPORT'),
            'raw_llama':  raw_output[:500],
            'evidence_used': bool(evidence),
        }
        results.append(result)

        # Agreement check
        q_verdict = triple.get('verdict', 'STRONG_SUPPORT')
        agree = (
            (verdict in ('STRONG_SUPPORT', 'WEAK_SUPPORT') and
             q_verdict in ('STRONG_SUPPORT', 'WEAK_SUPPORT'))
            or verdict == q_verdict
        )

        if (i + 1) % 10 == 0:
            ns_rate = verdict_counts.get('NOT_SUPPORTED', 0) / (i+1) * 100
            log.info(f'  [{i+1}/{len(sampled)}]  '
                     f'NS_so_far={ns_rate:.1f}%  '
                     f'dist={dict(verdict_counts)}')

    wall = time.time() - t_start

    # ── Metrics ────────────────────────────────────────────────────────
    n_total   = len(results)
    n_ns      = verdict_counts.get('NOT_SUPPORTED', 0)
    n_strong  = verdict_counts.get('STRONG_SUPPORT', 0)
    n_weak    = verdict_counts.get('WEAK_SUPPORT', 0)
    n_unres   = verdict_counts.get('UNRESOLVED', 0)
    n_noev    = verdict_counts.get('NO_EVIDENCE', 0)
    ns_rate   = n_ns / n_total * 100

    # Agreement between Llama and Qwen (both supported vs NS)
    agree_count = sum(
        1 for r in results
        if (r['verdict_llama'] in ('STRONG_SUPPORT', 'WEAK_SUPPORT') and
            r['verdict_qwen']  in ('STRONG_SUPPORT', 'WEAK_SUPPORT'))
        or r['verdict_llama'] == r['verdict_qwen']
    )
    agreement_rate = agree_count / n_total * 100

    # NS by relation type
    ns_by_rel = Counter()
    total_by_rel = Counter()
    for r in results:
        total_by_rel[r['relation']] += 1
        if r['verdict_llama'] == 'NOT_SUPPORTED':
            ns_by_rel[r['relation']] += 1

    metrics = {
        'experiment':        'D_cross_model_verifier',
        'model_extractor':   'Qwen/Qwen2.5-7B-Instruct',
        'model_verifier':    args.model,
        'n_triples':         n_total,
        'n_strong':          n_strong,
        'n_weak':            n_weak,
        'n_not_supported':   n_ns,
        'n_unresolved':      n_unres,
        'n_no_evidence':     n_noev,
        'ns_rate_pct':       round(ns_rate, 1),
        'agreement_rate_pct': round(agreement_rate, 1),
        'wall_seconds':      round(wall, 1),
        'verdict_dist':      dict(verdict_counts),
        'ns_by_relation':    dict(ns_by_rel),
    }

    with open(output_dir / 'metrics_expD.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / 'results_expD.jsonl', 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    # ── Report ─────────────────────────────────────────────────────────
    report_lines = [
        '=' * 65,
        'EXPERIMENT D — CROSS-MODEL VERIFIER',
        '=' * 65,
        f'Extractor model : Qwen/Qwen2.5-7B-Instruct',
        f'Verifier model  : {args.model}',
        f'Triples sampled : {n_total} Tier-1 triples',
        f'Wall time       : {wall/60:.1f} min',
        '',
        'VERDICT DISTRIBUTION (Llama verifier):',
        f'  STRONG_SUPPORT  : {n_strong} ({n_strong/n_total*100:.1f}%)',
        f'  WEAK_SUPPORT    : {n_weak}   ({n_weak/n_total*100:.1f}%)',
        f'  NOT_SUPPORTED   : {n_ns}     ({n_ns/n_total*100:.1f}%)',
        f'  UNRESOLVED      : {n_unres}  ({n_unres/n_total*100:.1f}%)',
        f'  NO_EVIDENCE     : {n_noev}   ({n_noev/n_total*100:.1f}%)',
        '',
        'KEY METRICS:',
        f'  H_T1 (Llama)   : {ns_rate:.1f}%',
        f'  H_T1 (Qwen)    : 0.0%  (original pipeline)',
        f'  Agreement rate : {agreement_rate:.1f}%',
        '',
    ]

    if ns_by_rel:
        report_lines.append('NOT_SUPPORTED by relation:')
        for rel, cnt in sorted(ns_by_rel.items(), key=lambda x: -x[1]):
            pct = cnt / total_by_rel[rel] * 100
            report_lines.append(
                f'  {rel:30s} {cnt}/{total_by_rel[rel]} ({pct:.0f}%)')
        report_lines.append('')

    report_lines += [
        'INTERPRETATION:',
        f'  Self-verification bias: ',
    ]
    if ns_rate < 5.0:
        report_lines.append(
            f'  Llama H_T1={ns_rate:.1f}% is consistent with Qwen H_T1=0%.')
        report_lines.append(
            f'  The 0% NOT_SUPPORTED result is ROBUST to verifier model choice.')
        report_lines.append(
            f'  Self-verification bias does NOT explain the Tier-1 reliability claim.')
    elif ns_rate < 15.0:
        report_lines.append(
            f'  Llama H_T1={ns_rate:.1f}% — modest disagreement with Qwen 0%.')
        report_lines.append(
            f'  Partial self-verification effect possible. Discuss in limitations.')
    else:
        report_lines.append(
            f'  Llama H_T1={ns_rate:.1f}% — substantial disagreement with Qwen 0%.')
        report_lines.append(
            f'  Self-verification bias likely contributes to the 0% result.')
        report_lines.append(
            f'  Revise Tier-1 reliability claim accordingly.')

    report_lines += [
        '',
        'PAPER LANGUAGE (fill in):',
        f'  "An independent cross-model verification using {args.model.split("/")[-1]}',
        f'   on the same 100 Tier-1 triples yields H_T1={ns_rate:.1f}%',
        f'   with {agreement_rate:.1f}% agreement with the Qwen-7B verifier,',
        f'   confirming that the Tier-1 reliability result is [robust to / partially',
        f'   affected by] verifier model choice."',
        '=' * 65,
    ]

    report = '\n'.join(report_lines)
    with open(output_dir / 'report_expD.txt', 'w') as f:
        f.write(report)

    print(report)
    log.info(f'Results saved to {output_dir}')


if __name__ == '__main__':
    main()