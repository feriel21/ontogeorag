#!/bin/bash
#SBATCH --job-name=ontogeorag
#SBATCH --partition=convergence
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/talbi/ontogeorag/logs/pipeline_%j.out
#SBATCH --error=/home/talbi/ontogeorag/logs/pipeline_%j.err

REPO=/home/talbi/ontogeorag
VENV=/home/talbi/kg_test/venv
OUT=$REPO/output

cd $REPO
source $VENV/bin/activate
export PYTHONPATH=$REPO:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=""

mkdir -p $REPO/logs

echo "[1/6] Extract triples..."
python -u pipeline/02_extract_triples.py \
    --index-dir $REPO/output/step1/ \
    --schema    $REPO/configs/ontology_schema.json \
    --queries   $REPO/configs/descriptor_queries.jsonl \
    --out       $OUT/step_extract/ \
    --model     Qwen/Qwen2.5-7B-Instruct
[ $? -ne 0 ] && echo "FAILED at step 1" && exit 1

echo "[2/6] Verify triples..."
python -u pipeline/03_verify_triples.py \
    --triples   $OUT/step_extract/raw_triples.jsonl \
    --index-dir $REPO/output/step1/ \
    --schema    $REPO/configs/ontology_schema.json \
    --out       $OUT/step_verify/ \
    --model     Qwen/Qwen2.5-7B-Instruct
[ $? -ne 0 ] && echo "FAILED at step 2" && exit 1

echo "[3/6] Clean & validate..."
python -u pipeline/04_clean_validate.py \
    --triples   $OUT/step_verify/verified_triples.jsonl \
    --schema    $REPO/configs/ontology_schema.json \
    --out       $OUT/step_clean/
[ $? -ne 0 ] && echo "FAILED at step 3" && exit 1

echo "[4/6] Canonicalize..."
python -u pipeline/05_canonicalize.py \
    --triples   $OUT/step_clean/clean_triples.jsonl \
    --out       $OUT/step_canon/
[ $? -ne 0 ] && echo "FAILED at step 4" && exit 1

echo "[5/6] Tiered fusion..."
python -u pipeline/06_tiered_fusion.py \
    --triples   $OUT/step_canon/canonical_triples.jsonl \
    --reference $REPO/configs/lb_reference_edges.json \
    --out       $OUT/improved_kg/
[ $? -ne 0 ] && echo "FAILED at step 5" && exit 1

echo "[6/6] Final metrics..."
python -u pipeline/07_final_metrics.py \
    --kg        $OUT/improved_kg/tiered_kg_final.json \
    --reference $REPO/configs/lb_reference_edges.json \
    --out       $OUT/metrics/
[ $? -ne 0 ] && echo "FAILED at step 6" && exit 1

echo "Pipeline complete: $(date)"
