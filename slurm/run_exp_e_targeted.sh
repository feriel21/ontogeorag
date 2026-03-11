#!/bin/bash
#SBATCH --job-name=exp_e_targeted
#SBATCH --partition=convergence
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/talbi/ontogeorag/logs/exp_e_targeted_%j.out
#SBATCH --error=/home/talbi/ontogeorag/logs/exp_e_targeted_%j.err

REPO=/home/talbi/ontogeorag
VENV=/home/talbi/kg_test/venv

cd $REPO
source $VENV/bin/activate
export PYTHONPATH=$REPO:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=""

mkdir -p $REPO/logs

python -u experiments/exp_e_targeted_rerun.py

echo "Done: $?"
