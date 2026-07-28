#!/bin/bash
#SBATCH --job-name=run12b
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/talbi/ontogeorag/logs/run12b_%j.out
#SBATCH --error=/home/talbi/ontogeorag/logs/run12b_%j.err

REPO=/home/talbi/ontogeorag
VENV=/home/talbi/kg_test/venv
INDEX=$REPO/output/step1
SCHEMA=$REPO/configs/ontology_schema.json
QUERIES=$REPO/configs/descriptor_queries.jsonl
MODEL=Qwen/Qwen2.5-7B-Instruct
B=$REPO/output/run12_b
KG=$REPO/output/run12_kg

mkdir -p $B $KG $REPO/logs
cd $REPO
source $VENV/bin/activate
export PYTHONPATH=$REPO:$PYTHONPATH

echo "Run 12 — Pass B only (Pass A already done)"
echo "Node: $(hostname) | $(date)"

echo "[B1/3] Extract pass B (temp=0.3) — HYBRID..."
python -u pipeline/02_extract_triples.py \
    --index-dir $INDEX --schema $SCHEMA --queries $QUERIES \
    --output $B/raw_triples.jsonl \
    --model $MODEL --backend hf \
    --top-k 5 --bm25-topn 20 --min-bm25 0.0 \
    --temperature 0.3 \
    --hybrid \
    --hybrid-model BAAI/bge-small-en-v1.5 \
    --fusion-alpha 0.5 \
    --reranker cross-encoder/ms-marco-MiniLM-L-6-v2
[ $? -ne 0 ] && echo "FAILED B1" && exit 1

echo "[B2/3] Verify pass B..."
python -u pipeline/03_verify_triples.py \
    --input  $B/raw_triples.jsonl \
    --output $B/verified_triples.jsonl \
    --model  $MODEL --backend hf
[ $? -ne 0 ] && echo "FAILED B2" && exit 1

echo "[B3/3] Clean pass B..."
python -u pipeline/04_clean_validate.py \
    --input  $B/verified_triples.jsonl \
    --outdir $B
[ $? -ne 0 ] && echo "FAILED B3" && exit 1

echo "[4] Fusion A+B..."
python -u pipeline/06_tiered_fusion.py \
    --iter-a $REPO/output/run12_a/canonical_triples_v5.jsonl \
    --iter-b $B/canonical_triples_v5.jsonl \
    --output $KG/tiered_kg_run12.json
[ $? -ne 0 ] && echo "FAILED fusion" && exit 1

echo "[5] Metrics..."
python -u pipeline/07_final_metrics.py \
    --kg     $KG/tiered_kg_run12.json \
    --output $KG/metrics_run12.json

echo "============================================"
echo "Run 12 COMPLETE: $(date)"
echo "============================================"

python3 -c "
import json
from pathlib import Path

def load_kg(p):
    kg = json.load(open(p))
    return kg.get('triples', kg) if isinstance(kg, dict) else kg

r11 = load_kg('$REPO/output/run11_kg/tiered_kg_run11.json')
r12 = load_kg('$KG/tiered_kg_run12.json')

t1_11 = [t for t in r11 if t.get('tier')==1]
t1_12 = [t for t in r12 if t.get('tier')==1]

print()
print('='*50)
print('  RUN 11 vs RUN 12 COMPARISON')
print('='*50)
print(f\"  {'Metric':<30} {'Run11':>8} {'Run12':>8} {'Delta':>8}\")
print(f\"  {'-'*54}\")
print(f\"  {'Total triples':<30} {len(r11):>8} {len(r12):>8} {len(r12)-len(r11):>+8}\")
print(f\"  {'Tier-1 triples':<30} {len(t1_11):>8} {len(t1_12):>8} {len(t1_12)-len(t1_11):>+8}\")
print('='*50)
"
