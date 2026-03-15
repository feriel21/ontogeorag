#!/bin/bash
#SBATCH --job-name=expD_cross_verifier
#SBATCH --partition=convergence
#SBATCH --nodelist=node08
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/expD_%j.out
#SBATCH --error=logs/expD_%j.err

echo "========================================"
echo "Experiment D — Cross-Model Verifier"
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"
echo "========================================"

cd ~/ontogeorag
source ~/kg_test/venv/bin/activate

mkdir -p logs output/expD

python3 pipeline/expD_cross_model.py \
    --kg        output/run11_kg/tiered_kg_run11.json \
    --index     output/step1/ \
    --output    output/expD/ \
    --model     meta-llama/Llama-3.1-8B-Instruct \
    --n-triples 100 \
    --seed      42

echo ""
echo "========================================"
echo "Exp D done: $(date)"
echo "Results:"
cat output/expD/report_expD.txt
echo "========================================"