#!/bin/bash
#SBATCH --job-name=ontogeorag_gen
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/talbi/ontogeorag/logs/generalization_%j.out
#SBATCH --error=/home/talbi/ontogeorag/logs/generalization_%j.err

REPO=/home/talbi/ontogeorag
VENV=/home/talbi/kg_test/venv
SCHEMA=$REPO/configs/ontology_schema.json
QUERIES=$REPO/configs/descriptor_queries.jsonl
MODEL=Qwen/Qwen2.5-7B-Instruct
NEW_INDEX=$REPO/output/generalization/step1
A=$REPO/output/generalization/run_a
B=$REPO/output/generalization/run_b
KG=$REPO/output/generalization/kg

mkdir -p $A $B $KG $REPO/logs
cd $REPO
source $VENV/bin/activate
export PYTHONPATH=$REPO:$PYTHONPATH

echo "============================================"
echo " OntoGeoRAG Generalization — $(date)"
echo " Corpus: 11 MTD papers (post-2019, unseen)"
echo " Index:  661 chunks"
echo " Node:   $(hostname)"
echo " GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "============================================"

echo "[A1/3] Extract pass A (temp=0.0)..."
python -u pipeline/02_extract_triples.py \
    --index-dir $NEW_INDEX \
    --schema    $SCHEMA \
    --queries   $QUERIES \
    --output    $A/raw_triples.jsonl \
    --model     $MODEL --backend hf \
    --top-k 5 --bm25-topn 20 --min-bm25 2.0 \
    --temperature 0.0 \
    --reranker cross-encoder/ms-marco-MiniLM-L-6-v2
[ $? -ne 0 ] && echo "FAILED A1" && exit 1

echo "[A2/3] Verify pass A..."
python -u pipeline/03_verify_triples.py \
    --input  $A/raw_triples.jsonl \
    --output $A/verified_triples.jsonl \
    --model  $MODEL --backend hf
[ $? -ne 0 ] && echo "FAILED A2" && exit 1

echo "[A3/3] Clean pass A..."
python -u pipeline/04_clean_validate.py \
    --input  $A/verified_triples.jsonl \
    --outdir $A
[ $? -ne 0 ] && echo "FAILED A3" && exit 1

echo "[B1/3] Extract pass B (temp=0.3)..."
python -u pipeline/02_extract_triples.py \
    --index-dir $NEW_INDEX \
    --schema    $SCHEMA \
    --queries   $QUERIES \
    --output    $B/raw_triples.jsonl \
    --model     $MODEL --backend hf \
    --top-k 5 --bm25-topn 20 --min-bm25 2.0 \
    --temperature 0.3 \
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

echo "[5] Fusion..."
python -u pipeline/06_tiered_fusion.py \
    --iter-a $A/canonical_triples_v5.jsonl \
    --iter-b $B/canonical_triples_v5.jsonl \
    --output $KG/tiered_kg_generalization.json
[ $? -ne 0 ] && echo "FAILED fusion" && exit 1

echo "[6] Metrics..."
python -u pipeline/07_final_metrics.py \
    --kg     $KG/tiered_kg_generalization.json \
    --output $KG/metrics_generalization.json

echo "============================================"
echo " Generalization COMPLETE: $(date)"
echo "============================================"

python3 -c "
import json, sys
sys.path.insert(0, '$REPO')
kg = json.load(open('$KG/tiered_kg_generalization.json'))
triples = kg.get('triples', kg) if isinstance(kg, dict) else kg
tier1  = [t for t in triples if t.get('tier')==1]
tier12 = [t for t in triples if t.get('tier') in [1,2]]
print('  Tier-1 triples:   ' + str(len(tier1)))
print('  Tier-1+2 triples: ' + str(len(tier12)))
from collections import Counter
rels = Counter(t['relation'] for t in tier12)
print('  Relations:')
for r, n in rels.most_common():
    print('    ' + r + ': ' + str(n))
print()
print('  Sample Tier-1 triples:')
for t in tier1[:10]:
    print('    (' + t['subject'] + ', ' +
          t['relation'] + ', ' + t['object'] + ')')
"
