#!/bin/bash
#SBATCH --job-name=exp70b_ontogeorag
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_7g.80gb:2
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8

#SBATCH --output=/home/talbi/ontogeorag/logs/exp70b_%j.out
#SBATCH --error=/home/talbi/ontogeorag/logs/exp70b_%j.err

REPO=/home/talbi/ontogeorag
VENV=/home/talbi/kg_test/venv
INDEX=$REPO/output/step1
SCHEMA=$REPO/configs/ontology_schema.json
QUERIES=$REPO/configs/descriptor_queries.jsonl
MODEL_EXTRACT=meta-llama/Llama-3.1-70B-Instruct
MODEL_VERIFY=Qwen/Qwen2.5-7B-Instruct
A=$REPO/output/exp70b_a
B=$REPO/output/exp70b_b
KG=$REPO/output/exp70b_kg

mkdir -p $A $B $KG $REPO/logs
cd $REPO
source $VENV/bin/activate
export PYTHONPATH=$REPO:$PYTHONPATH

echo "============================================"
echo " OntoGeoRAG EXP-70B (GPU) — $(date)"
echo " Node: $(hostname)"
echo " GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
echo " Extract model: $MODEL_EXTRACT"
echo " Verify  model: $MODEL_VERIFY (kept = Qwen-7B to isolate extraction effect)"
echo " Retrieval/queries/hyperparams: IDENTICAL to run11"
echo "============================================"

echo "[A1/3] Extract (pass A, temp=0.0) with 70B..."
python -u pipeline/02_extract_triples.py \
    --index-dir $INDEX --schema $SCHEMA --queries $QUERIES \
    --output $A/raw_triples.jsonl --model $MODEL_EXTRACT --backend hf \
    --top-k 5 --bm25-topn 20 --min-bm25 2.0 --temperature 0.0 \
    --reranker cross-encoder/ms-marco-MiniLM-L-6-v2
[ $? -ne 0 ] && echo "FAILED A1" && exit 1

echo "[A2/3] Verify (pass A) with Qwen-7B..."
python -u pipeline/03_verify_triples.py \
    --input $A/raw_triples.jsonl --output $A/verified_triples.jsonl \
    --model $MODEL_VERIFY --backend hf
[ $? -ne 0 ] && echo "FAILED A2" && exit 1

echo "[A3/3] Clean + canonicalize (pass A)..."
python -u pipeline/04_clean_validate.py \
    --input $A/verified_triples.jsonl --outdir $A
[ $? -ne 0 ] && echo "FAILED A3" && exit 1

echo "[B1/3] Extract (pass B, temp=0.3) with 70B..."
python -u pipeline/02_extract_triples.py \
    --index-dir $INDEX --schema $SCHEMA --queries $QUERIES \
    --output $B/raw_triples.jsonl --model $MODEL_EXTRACT --backend hf \
    --top-k 5 --bm25-topn 20 --min-bm25 2.0 --temperature 0.3 \
    --reranker cross-encoder/ms-marco-MiniLM-L-6-v2
[ $? -ne 0 ] && echo "FAILED B1" && exit 1

echo "[B2/3] Verify (pass B) with Qwen-7B..."
python -u pipeline/03_verify_triples.py \
    --input $B/raw_triples.jsonl --output $B/verified_triples.jsonl \
    --model $MODEL_VERIFY --backend hf
[ $? -ne 0 ] && echo "FAILED B2" && exit 1

echo "[B3/3] Clean + canonicalize (pass B)..."
python -u pipeline/04_clean_validate.py \
    --input $B/verified_triples.jsonl --outdir $B
[ $? -ne 0 ] && echo "FAILED B3" && exit 1

echo "[5] Fusion..."
PYTHONPATH=$REPO python -u pipeline/06_tiered_fusion.py \
    --iter-a $A/canonical_triples_v5.jsonl \
    --iter-b $B/canonical_triples_v5.jsonl \
    --output $KG/tiered_kg_exp70b.json
[ $? -ne 0 ] && echo "FAILED fusion" && exit 1

echo "[6] Metrics..."
python -u pipeline/07_final_metrics.py \
    --kg $KG/tiered_kg_exp70b.json --output $KG/metrics_exp70b.json

echo "============================================"
echo " EXP-70B COMPLETE: $(date)"
echo "============================================"
