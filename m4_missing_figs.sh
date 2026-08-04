#!/bin/bash
#SBATCH --job-name=m4_missing_figs
#SBATCH --partition=a100_3g.40gb
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/m4_missing_figs_%j.log

set -e
cd ~/ontogeorag
source ~/kg_test/venv/bin/activate
unset SLURM_PROCID SLURM_NTASKS RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export HF_HUB_OFFLINE=1 PYTHONPATH=$PWD

KG=output/run13/kg/tiered_kg_run13_enforced.json
DECISIONS=output/run13/m4_panel/m4_panel_decisions.jsonl
OUTDIR=output/run13/m4
mkdir -p "$OUTDIR"

# 1. direction check — never run on run13
python m4/m4_direction_check.py \
    --kg "$KG" \
    --decisions "$DECISIONS" \
    --output "$OUTDIR/direction_check.json"

# 2. negatives sensitivity — generate (CPU-fast) -> verify (GPU) -> report (CPU-fast)
python m4/m4_negatives.py generate \
    --kg "$KG" \
    --output "$OUTDIR/negatives" \
    --seed 13

python m4/m4_verify.py \
    --kg "$OUTDIR/negatives/controls.jsonl" \
    --output "$OUTDIR/negatives" \
    --model meta-llama/Llama-3.1-8B-Instruct

python m4/m4_negatives.py report \
    --controls "$OUTDIR/negatives/controls.jsonl" \
    --verdicts "$OUTDIR/negatives/m4_verdicts.jsonl" \
    --output "$OUTDIR/negatives"

# 3. calibration — reuses the same controls + verdicts pair from step 2
python m4/m4_calibration.py \
    --controls "$OUTDIR/negatives/controls.jsonl" \
    --verdicts "$OUTDIR/negatives/m4_verdicts.jsonl" \
    --output "$OUTDIR/calibration.json"

# regenerate the figures
# NOTE: m4_figures_v2.py takes --negatives / --direction / --panel / --output,
# but NOT --calibration. Check whether m4_calibration.py itself writes a figure
# (e.g. a .png/.pdf next to calibration.json) before assuming fig_m4_reliability
# comes from this call.
python m4/m4_figures_v2.py \
    --panel output/run13/m4_panel/m4_panel_report.json \
    --direction "$OUTDIR/direction_check.json" \
    --negatives "$OUTDIR/negatives/negatives_report.json" \
    --output figures/paper/m4_run13