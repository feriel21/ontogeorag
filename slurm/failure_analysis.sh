#!/bin/bash
#SBATCH --job-name=failure_analysis
#SBATCH --partition=convergence
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/failure_analysis_%j.out
#SBATCH --error=logs/failure_analysis_%j.err

source ~/kg_test/venv/bin/activate
cd ~/ontogeorag

python pipeline/failure_analysis.py \
    --index-dir output/step1/ \
    --reference configs/lb_reference_edges.json \
    --kg-c10 output/run11_kg/tiered_kg_run11.json \
    --model Qwen/Qwen2.5-7B-Instruct \
    --rerank \
    --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --output output/diagnostics/failure_analysis.json