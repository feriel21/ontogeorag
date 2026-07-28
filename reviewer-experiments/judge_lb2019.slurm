#!/bin/bash
#SBATCH --job-name=judge_lb2019
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/talbi/ontogeorag/logs/judge_lb2019_%j.out
#SBATCH --error=/home/talbi/ontogeorag/logs/judge_lb2019_%j.err

REPO=/home/talbi/ontogeorag
VENV=/home/talbi/kg_test/venv
cd $REPO/reviewer-experiments
source $VENV/bin/activate
export PYTHONPATH=$REPO:$PYTHONPATH

echo "=== GPU CHECK $(date) node $(hostname) ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
python -c "import torch; print('torch CUDA:', torch.cuda.is_available(), '| n=', torch.cuda.device_count())"
echo "=== fin check, lancement juge ==="

python -u judge_lb2019.py \
    --csv    recoverability_audit_v2/edge_recoverability_v2.csv \
    --model  Qwen/Qwen2.5-7B-Instruct \
    --outdir ./judge_audit
echo "=== fini — $(date) ==="
