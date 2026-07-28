#!/bin/bash
#SBATCH --job-name=m4_verify
#SBATCH  --gres=gpu:a100_3g.40gb:1
#SBATCH --time=02:00:00
#SBATCH --output=m4_verify_%j.out
#SBATCH --error=m4_verify_%j.err

 
# M4 Independent Cross-Family Verifier — SLURM job
# Llama-3.1-8B in bf16 fits comfortably on a 40GB MIG slice (node07-10)
# as well as full A100 80GB (node01-06).
 
# Fail fast: stop at the first error instead of cascading through steps
set -e
 
# CRITICAL: clear distributed-training env vars before launching Python
unset SLURM_PROCID SLURM_NTASKS RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
 
# Paths — adjust if the repo lives elsewhere
REPO="$HOME/ontogeorag"
M4DIR="$HOME/ontogeorag/m4"
VENV="$HOME/kg_test/venv"
 
source "$VENV/bin/activate"
 
cd "$M4DIR" || exit 1
echo "Working directory: $(pwd)"
 
# ── Step 1: dual-pass verification (GPU, ~1 s-3 s per triple x 2 passes)
python m4_verify.py \
    --kg     "$REPO/output/run11_kg/tiered_kg_run11.json" \
    --index  "$REPO/output/step1" \
    --output "$REPO/output/m4" \
    --model  meta-llama/Llama-3.1-8B-Instruct
 
# ── Step 2: aggregation (CPU, seconds)
python m4_aggregate.py \
    --verdicts "$REPO/output/m4/m4_verdicts.jsonl" \
    --output   "$REPO/output/m4"
 
# ── Step 3: metrics report (CPU, seconds)
python m4_metrics.py \
    --decisions "$REPO/output/m4/m4_decisions.jsonl" \
    --output    "$REPO/output/m4"
 
# ── Step 4: rebuild tiered KG with M4 verdicts (CPU, seconds)
python m4_integrate_tiers.py \
    --kg        "$REPO/output/run11_kg/tiered_kg_run11.json" \
    --decisions "$REPO/output/m4/m4_decisions.jsonl" \
    --output    "$REPO/output/m4"
 
echo "M4 pipeline complete. Outputs in $REPO/output/m4/"
 