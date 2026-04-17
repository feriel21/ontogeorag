#!/bin/bash
#SBATCH --job-name=exp_d_crossmodel
#SBATCH --partition=convergence
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/talbi/ontogeorag/logs/exp_d_%j.out
#SBATCH --error=/home/talbi/ontogeorag/logs/exp_d_%j.err

REPO=/home/talbi/ontogeorag
VENV=/home/talbi/kg_test/venv

cd $REPO
source $VENV/bin/activate
export PYTHONPATH=$REPO:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=""

mkdir -p $REPO/logs $REPO/output/exp_d

python -u experiments/exp_d_cross_model.py \
    --triples   $REPO/output/step7/raw_triples_v7.jsonl \
    --index-dir $REPO/output/step1/ \
    --schema    $REPO/configs/ontology_schema.json \
    --out       $REPO/output/exp_d/ \
    --model     Qwen/Qwen2.5-7B-Instruct

echo "EXP-D done: $?"
