#!/bin/bash
#SBATCH --job-name=iter1_ontogeorag
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_7g.80gb:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/talbi/ontogeorag/logs/iter1_%j.out
#SBATCH --error=/home/talbi/ontogeorag/logs/iter1_%j.err

REPO=/home/talbi/ontogeorag
VENV=/home/talbi/kg_test/venv
INDEX=$REPO/output/step1
SCHEMA=$REPO/configs/ontology_schema.json
QUERIES=$REPO/output/iter1_queries.jsonl
MODEL=Qwen/Qwen2.5-7B-Instruct
IT=$REPO/output/iter1
KG0=$REPO/output/run11_kg/tiered_kg_run11.json
KG=$REPO/output/iter1_kg

mkdir -p $IT $KG $REPO/logs
cd $REPO
source $VENV/bin/activate
export PYTHONPATH=$REPO:$PYTHONPATH

echo "=== ITERATION 1 $(date) on $(hostname) ==="
echo "Queries: KG-augmented ($(wc -l < $QUERIES) queries) | Model: $MODEL | Verifier: text-only"

echo "[1] Extract (augmented queries, temp=0.0)..."
python -u pipeline/02_extract_triples.py \
    --index-dir $INDEX --schema $SCHEMA --queries $QUERIES \
    --output $IT/raw_triples.jsonl --model $MODEL --backend hf \
    --top-k 5 --bm25-topn 20 --min-bm25 2.0 --temperature 0.0 \
    --reranker cross-encoder/ms-marco-MiniLM-L-6-v2
[ $? -ne 0 ] && echo "FAILED extract" && exit 1

echo "[2] Verify (text-only, same verifier as run11)..."
python -u pipeline/03_verify_triples.py \
    --input $IT/raw_triples.jsonl --output $IT/verified_triples.jsonl \
    --model $MODEL --backend hf
[ $? -ne 0 ] && echo "FAILED verify" && exit 1

echo "[3] Clean + canonicalize..."
python -u pipeline/04_clean_validate.py \
    --input $IT/verified_triples.jsonl --outdir $IT
[ $? -ne 0 ] && echo "FAILED clean" && exit 1

echo "[4] Fuse KG_0 + iteration triples -> KG_1..."
PYTHONPATH=$REPO python -u pipeline/06_tiered_fusion.py \
    --iter-a $KG0 \
    --iter-b $IT/canonical_triples_v5.jsonl \
    --output $KG/tiered_kg_iter1.json
[ $? -ne 0 ] && echo "FAILED fusion" && exit 1

echo "=== ITERATION 1 COMPLETE $(date) ==="
