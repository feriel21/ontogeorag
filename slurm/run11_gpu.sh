#!/bin/bash
#SBATCH --job-name=run11_ontogeorag
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/talbi/ontogeorag/logs/run11_%j.out
#SBATCH --error=/home/talbi/ontogeorag/logs/run11_%j.err

REPO=/home/talbi/ontogeorag
VENV=/home/talbi/kg_test/venv
INDEX=$REPO/output/step1
SCHEMA=$REPO/configs/ontology_schema.json
QUERIES=$REPO/configs/descriptor_queries.jsonl
MODEL=Qwen/Qwen2.5-7B-Instruct
A=$REPO/output/run11_a
B=$REPO/output/run11_b
KG=$REPO/output/run11_kg

mkdir -p $A $B $KG $REPO/logs
cd $REPO
source $VENV/bin/activate
export PYTHONPATH=$REPO:$PYTHONPATH

echo "============================================"
echo " OntoGeoRAG Run 11 (GPU) — $(date)"
echo " Node: $(hostname)"
echo " GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo " P10: BM25 top-20 + CrossEncoder reranker -> top-5"
echo "============================================"

echo "[A1/3] Extract (pass A, temp=0.0)..."
python -u pipeline/02_extract_triples.py \
    --index-dir $INDEX --schema $SCHEMA --queries $QUERIES \
    --output $A/raw_triples.jsonl --model $MODEL --backend hf \
    --top-k 5 --bm25-topn 20 --min-bm25 2.0 --temperature 0.0 \
    --reranker cross-encoder/ms-marco-MiniLM-L-6-v2
[ $? -ne 0 ] && echo "FAILED A1" && exit 1

echo "[A2/3] Verify (pass A)..."
python -u pipeline/03_verify_triples.py \
    --input $A/raw_triples.jsonl --output $A/verified_triples.jsonl \
    --model $MODEL --backend hf
[ $? -ne 0 ] && echo "FAILED A2" && exit 1

echo "[A3/3] Clean + canonicalize (pass A)..."
python -u pipeline/04_clean_validate.py \
    --input $A/verified_triples.jsonl --outdir $A
[ $? -ne 0 ] && echo "FAILED A3" && exit 1

echo "[B1/3] Extract (pass B, temp=0.3)..."
python -u pipeline/02_extract_triples.py \
    --index-dir $INDEX --schema $SCHEMA --queries $QUERIES \
    --output $B/raw_triples.jsonl --model $MODEL --backend hf \
    --top-k 5 --bm25-topn 20 --min-bm25 2.0 --temperature 0.3 \
    --reranker cross-encoder/ms-marco-MiniLM-L-6-v2
[ $? -ne 0 ] && echo "FAILED B1" && exit 1

echo "[B2/3] Verify (pass B)..."
python -u pipeline/03_verify_triples.py \
    --input $B/raw_triples.jsonl --output $B/verified_triples.jsonl \
    --model $MODEL --backend hf
[ $? -ne 0 ] && echo "FAILED B2" && exit 1

echo "[B3/3] Clean + canonicalize (pass B)..."
python -u pipeline/04_clean_validate.py \
    --input $B/verified_triples.jsonl --outdir $B
[ $? -ne 0 ] && echo "FAILED B3" && exit 1

echo "[5] Fusion..."
PYTHONPATH=$REPO python -u pipeline/06_tiered_fusion.py \
    --iter-a $A/canonical_triples_v5.jsonl \
    --iter-b $B/canonical_triples_v5.jsonl \
    --output $KG/tiered_kg_run11.json
[ $? -ne 0 ] && echo "FAILED fusion" && exit 1

echo "[6] Metrics..."
python -u pipeline/07_final_metrics.py \
    --kg $KG/tiered_kg_run11.json --output $KG/metrics_run11.json

echo "============================================"
echo " Run 11 COMPLETE: $(date)"
echo "============================================"
python3 -c "
import json, re, sys
sys.path.insert(0, '$REPO')
from pipeline.rag.constants import normalize_relation, LB2019_DESCRIPTORS
kg = json.load(open('$KG/tiered_kg_run11.json'))
triples = kg.get('triples', kg) if isinstance(kg, dict) else kg
tier1 = [t for t in triples if t.get('tier')==1]
tier2 = [t for t in triples if t.get('tier')==2]
ref = json.load(open('$REPO/configs/lb_reference_edges.json'))
ref = ref.get('edges', ref) if isinstance(ref, dict) else ref
t1  = sum(1 for r in ref if any(
    r.get('subject','').lower() in (t.get('subject','') or '').lower() and
    r.get('object','').lower() in (t.get('object','') or '').lower() and
    normalize_relation(r.get('relation',''))==normalize_relation(t.get('relation',''))
    for t in tier1))
t12 = sum(1 for r in ref if any(
    r.get('subject','').lower() in (t.get('subject','') or '').lower() and
    r.get('object','').lower() in (t.get('object','') or '').lower() and
    normalize_relation(r.get('relation',''))==normalize_relation(t.get('relation',''))
    for t in triples))
print(f'  Total: {len(triples)} (T1={len(tier1)}, T2={len(tier2)})')
print(f'  Recall T1:   {t1}/26 = {t1/26*100:.1f}%')
print(f'  Recall T1+2: {t12}/26 = {t12/26*100:.1f}%')
"
