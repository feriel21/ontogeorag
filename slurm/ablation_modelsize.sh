#!/bin/bash
#SBATCH --job-name=ablation_modelsize
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/talbi/ontogeorag/logs/ablation_modelsize_%j.out

REPO=/home/talbi/ontogeorag
VENV=/home/talbi/kg_test/venv
INDEX=$REPO/output/step1
SCHEMA=$REPO/configs/ontology_schema.json
QUERIES=$REPO/configs/ablation_50q.jsonl
MODEL_7B=Qwen/Qwen2.5-7B-Instruct

cd $REPO
source $VENV/bin/activate
export PYTHONPATH=$REPO:$PYTHONPATH

mkdir -p output/ablation_modelsize

# Condition 1: No retrieval (memory only)
echo "=== CONDITION 1: Memory only (no retrieval) ==="
python -u pipeline/expB_no_rag.py \
    --schema  $SCHEMA \
    --queries $QUERIES \
    --output  output/ablation_modelsize/memory_only.jsonl \
    --model   $MODEL_7B --backend hf

# Condition 2: BM25 only
echo "=== CONDITION 2: BM25 only ==="
python -u pipeline/02_extract_triples.py \
    --index-dir $INDEX --schema $SCHEMA --queries $QUERIES \
    --output output/ablation_modelsize/bm25_only.jsonl \
    --model $MODEL_7B --backend hf \
    --top-k 5 --bm25-topn 5 --min-bm25 2.0 --temperature 0.0

# Condition 3: BM25 + reranker (full C10)
echo "=== CONDITION 3: BM25 + reranker ==="
python -u pipeline/02_extract_triples.py \
    --index-dir $INDEX --schema $SCHEMA --queries $QUERIES \
    --output output/ablation_modelsize/reranked.jsonl \
    --model $MODEL_7B --backend hf \
    --top-k 5 --bm25-topn 20 --min-bm25 2.0 --temperature 0.0 \
    --reranker cross-encoder/ms-marco-MiniLM-L-6-v2
