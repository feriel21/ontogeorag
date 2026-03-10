#!/bin/bash
#SBATCH --job-name=exp_e_llama
#SBATCH --partition=convergence
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/talbi/ontogeorag/logs/exp_e_%j.out
#SBATCH --error=/home/talbi/ontogeorag/logs/exp_e_%j.err

echo "======================================"
echo "EXP-E: Llama-3.1-8B Extraction"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "======================================"

# Absolute paths — SLURM does not inherit your shell's cd or PATH
REPO=/home/talbi/ontogeorag
VENV=/home/talbi/kg_test/venv

cd $REPO
source $VENV/bin/activate
export PYTHONPATH=$REPO:$PYTHONPATH   # ← this line fixes it

mkdir -p $REPO/logs $REPO/output/exp_e

python -u experiments/exp_e_llama_extraction.py \
    --index-dir /home/talbi/kg_test/output/step1/ \
    --schema    $REPO/configs/ontology_schema.json \
    --queries   $REPO/configs/descriptor_queries.jsonl \
    --qwen-ref  /home/talbi/kg_test/output/step7/raw_triples_v7.jsonl \
    --out       $REPO/output/exp_e/ \
    --model     meta-llama/Llama-3.1-8B-Instruct \
    --backend   hf

EXIT_CODE=$?
echo "======================================"
echo "EXP-E finished with exit code: $EXIT_CODE"
echo "End: $(date)"
echo "======================================"