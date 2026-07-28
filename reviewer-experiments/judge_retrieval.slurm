#!/bin/bash
#SBATCH --job-name=judge_retr
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/talbi/ontogeorag/logs/judge_retr_%j.out
#SBATCH --error=/home/talbi/ontogeorag/logs/judge_retr_%j.err
REPO=/home/talbi/ontogeorag
cd $REPO/reviewer-experiments
source $REPO/../kg_test/venv/bin/activate 2>/dev/null || source /home/talbi/kg_test/venv/bin/activate
export PYTHONPATH=$REPO:$PYTHONPATH
echo "=== $(date) node $(hostname) ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -u judge_retrieval.py \
    --kg        $REPO/reference/reference_kg.json \
    --index-dir $REPO/output/step1 \
    --model     Qwen/Qwen2.5-7B-Instruct \
    --top-n     5 \
    --outdir    ./judge_retrieval_audit
echo "=== fini $(date) ==="
