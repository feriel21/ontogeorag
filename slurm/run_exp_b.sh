#!/bin/bash
#SBATCH --job-name=exp_b_norag
#SBATCH --partition=convergence
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/talbi/ontogeorag/logs/exp_b_%j.out
#SBATCH --error=/home/talbi/ontogeorag/logs/exp_b_%j.err

REPO=/home/talbi/ontogeorag
VENV=/home/talbi/kg_test/venv

cd $REPO
source $VENV/bin/activate
export PYTHONPATH=$REPO:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=""

mkdir -p $REPO/logs $REPO/output/exp_b

python -u experiments/exp_b_no_rag_ablation.py \
    --index-dir $REPO/output/step1/ \
    --schema    $REPO/configs/ontology_schema.json \
    --queries   $REPO/configs/descriptor_queries.jsonl \
    --out       $REPO/output/exp_b/ \
    --model     Qwen/Qwen2.5-7B-Instruct

echo "EXP-B done: $?"
