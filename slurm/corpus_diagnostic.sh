#!/bin/bash
#SBATCH --job-name=corpus_diagnostic
#SBATCH --partition=convergence
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8

#SBATCH --output=logs/corpus_diagnostic_%j.out
#SBATCH --error=logs/corpus_diagnostic_%j.err

source ~/kg_test/venv/bin/activate
cd ~/ontogeorag

python pipeline/corpus_diagnostic.py \
    --index-dir output/step1/ \
    --reference configs/lb_reference_edges.json \
    --kg-c9 output/run10_kg/tiered_kg_run10_final.json \
    --kg-c10 output/run11_kg/tiered_kg_run11.json \
    --output output/diagnostics/corpus_diagnostic.json