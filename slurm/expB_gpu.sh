#!/bin/bash
#SBATCH --job-name=expB_no_rag
#SBATCH --partition=convergence
#SBATCH --nodelist=node08
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/expB_%j.out
#SBATCH --error=logs/expB_%j.err

echo "========================================"
echo "Experiment B — No-RAG Baseline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"
echo "========================================"

cd ~/ontogeorag
source ~/kg_test/venv/bin/activate

mkdir -p logs output/expB

python3 pipeline/expB_no_rag.py \
    --queries configs/descriptor_queries.jsonl \
    --ref     configs/lb_reference_edges.json \
    --output  output/expB \
    --model   Qwen/Qwen2.5-7B-Instruct \
    --device  cuda \
    --temperature 0.0

echo ""
echo "========================================"
echo "Exp B done: $(date)"
echo "Results:"
cat output/expB/report_expB.txt
echo "========================================"