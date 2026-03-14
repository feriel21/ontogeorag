#!/bin/bash
#SBATCH --job-name=run9_ontogeorag

#SBATCH --partition=convergence

#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/talbi/ontogeorag/logs/run9_%j.out
#SBATCH --error=/home/talbi/ontogeorag/logs/run9_%j.err

REPO=/home/talbi/ontogeorag
VENV=/home/talbi/kg_test/venv
INDEX=/home/talbi/kg_test/output/step1
SCHEMA=$REPO/configs/ontology_schema.json
QUERIES=$REPO/configs/descriptor_queries.jsonl
MODEL=Qwen/Qwen2.5-7B-Instruct
A=$REPO/output/run9_a
B=$REPO/output/run9_b
KG=$REPO/output/run9_kg
V4=/home/talbi/kg_test/output/step4/canonical_triples_v4.jsonl
V7=/home/talbi/kg_test/output/step7/canonical_triples_v7.jsonl

mkdir -p $A $B $KG $REPO/logs
cd $REPO
source $VENV/bin/activate
export PYTHONPATH=$REPO:$PYTHONPATH

echo "============================================"
echo " OntoGeoRAG Run 9 (GPU) — $(date)"
echo " Node: $(hostname)"
echo " GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo " P1: extended RELATION_MAP"
echo " P2: few-shot gold examples"
echo " P3: evidence aggregation"
echo " P4: +15 queries (turbidite/slide/debris flow/hemipelagite)"
echo " Fusion: run9_a + run9_b + v4 + v7"
echo "============================================"

# ── PASS A (temp=0.0) ─────────────────────────────
echo ""
echo "[A1/3] Extract (pass A, temp=0.0)..."
python -u pipeline/02_extract_triples.py \
    --index-dir   $INDEX \
    --schema      $SCHEMA \
    --queries     $QUERIES \
    --output      $A/raw_triples.jsonl \
    --model       $MODEL \
    --backend     hf \
    --top-k       5 \
    --bm25-topn   10 \
    --min-bm25    2.0 \
    --temperature 0.0
[ $? -ne 0 ] && echo "FAILED at A1" && exit 1
echo "  -> $(wc -l < $A/raw_triples.jsonl) raw triples"

echo "[A2/3] Verify (pass A)..."
python -u pipeline/03_verify_triples.py \
    --input   $A/raw_triples.jsonl \
    --output  $A/verified_triples.jsonl \
    --model   $MODEL \
    --backend hf
[ $? -ne 0 ] && echo "FAILED at A2" && exit 1
echo "  -> $(wc -l < $A/verified_triples.jsonl) verified triples"

echo "[A3/3] Clean + canonicalize (pass A)..."
python -u pipeline/04_clean_validate.py \
    --input  $A/verified_triples.jsonl \
    --outdir $A
[ $? -ne 0 ] && echo "FAILED at A3" && exit 1
echo "  -> $(wc -l < $A/canonical_triples_v5.jsonl) canonical triples"

# ── PASS B (temp=0.3) ─────────────────────────────
echo ""
echo "[B1/3] Extract (pass B, temp=0.3)..."
python -u pipeline/02_extract_triples.py \
    --index-dir   $INDEX \
    --schema      $SCHEMA \
    --queries     $QUERIES \
    --output      $B/raw_triples.jsonl \
    --model       $MODEL \
    --backend     hf \
    --top-k       5 \
    --bm25-topn   10 \
    --min-bm25    2.0 \
    --temperature 0.3
[ $? -ne 0 ] && echo "FAILED at B1" && exit 1
echo "  -> $(wc -l < $B/raw_triples.jsonl) raw triples"

echo "[B2/3] Verify (pass B)..."
python -u pipeline/03_verify_triples.py \
    --input   $B/raw_triples.jsonl \
    --output  $B/verified_triples.jsonl \
    --model   $MODEL \
    --backend hf
[ $? -ne 0 ] && echo "FAILED at B2" && exit 1
echo "  -> $(wc -l < $B/verified_triples.jsonl) verified triples"

echo "[B3/3] Clean + canonicalize (pass B)..."
python -u pipeline/04_clean_validate.py \
    --input  $B/verified_triples.jsonl \
    --outdir $B
[ $? -ne 0 ] && echo "FAILED at B3" && exit 1
echo "  -> $(wc -l < $B/canonical_triples_v5.jsonl) canonical triples"

# ── FUSION EN CASCADE ─────────────────────────────
# Étape 1 : A + B → tmp
echo ""
echo "[5a] Fusion A + B -> tmp..."
python -u pipeline/06_tiered_fusion.py \
    --iter-a $A/canonical_triples_v5.jsonl \
    --iter-b $B/canonical_triples_v5.jsonl \
    --output $KG/tmp_ab.json
[ $? -ne 0 ] && echo "FAILED at fusion A+B" && exit 1

# Étape 2 : tmp + v7 → tmp2
echo "[5b] Fusion (A+B) + v7 -> tmp2..."
python -u pipeline/06_tiered_fusion.py \
    --iter-a $KG/tmp_ab.json \
    --iter-b $V7 \
    --output $KG/tmp_ab_v7.json
[ $? -ne 0 ] && echo "FAILED at fusion +v7" && exit 1

# Étape 3 : tmp2 + v4 → final
echo "[5c] Fusion (A+B+v7) + v4 -> final..."
python -u pipeline/06_tiered_fusion.py \
    --iter-a $KG/tmp_ab_v7.json \
    --iter-b $V4 \
    --output $KG/tiered_kg_run9.json
[ $? -ne 0 ] && echo "FAILED at fusion +v4" && exit 1

echo "  -> Fusion complete"

# ── METRICS ───────────────────────────────────────
echo ""
echo "[6] Final metrics..."
python -u pipeline/07_final_metrics.py \
    --kg     $KG/tiered_kg_run9.json \
    --output $KG/metrics_run9.json
[ $? -ne 0 ] && echo "FAILED at metrics" && exit 1

# ── SUMMARY ───────────────────────────────────────
echo ""
echo "============================================"
echo " Run 9 COMPLETE: $(date)"
echo "============================================"
python3 -c "
import json, re, sys
sys.path.insert(0, '$REPO')
from pipeline.rag.constants import normalize_relation, LB2019_DESCRIPTORS
kg = json.load(open('$KG/tiered_kg_run9.json'))
triples = kg.get('triples', kg) if isinstance(kg, dict) else kg
tier1 = [t for t in triples if t.get('tier')==1]
tier2 = [t for t in triples if t.get('tier')==2]
def norm(s): return re.sub(r'\s+', ' ', (s or '').lower().strip()).rstrip('.,;:')
def matches(a, b):
    ks,ko,kr = norm(a.get('subject','')), norm(a.get('object','')), normalize_relation(a.get('relation',''))
    rs,ro,rr = norm(b.get('subject','')), norm(b.get('object','')), normalize_relation(b.get('relation',''))
    return (ks==rs or ks in rs or rs in ks) and (ko==ro or ko in ro or ro in ko) and kr==rr
ref = json.load(open('$REPO/configs/lb_reference_edges.json'))
ref = ref.get('edges', ref) if isinstance(ref, dict) else ref
t1_hit  = sum(1 for r in ref if any(matches(t,r) for t in tier1))
t12_hit = sum(1 for r in ref if any(matches(t,r) for t in triples))
covered = set()
for t in triples:
    if 'descriptor' in t.get('relation','').lower():
        o = norm(t.get('object',''))
        for d in LB2019_DESCRIPTORS:
            if d in o: covered.add(d)
print(f'  Total triples      : {len(triples)} (T1={len(tier1)}, T2={len(tier2)})')
print(f'  Recall T1          : {t1_hit}/26 = {t1_hit/26*100:.1f}%')
print(f'  Recall T1+2        : {t12_hit}/26 = {t12_hit/26*100:.1f}%')
print(f'  Descriptor coverage: {len(covered)}/13')
print(f'  Missing            : {[d for d in LB2019_DESCRIPTORS if d not in covered]}')
"
