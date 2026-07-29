#!/usr/bin/env bash
# run_full_analysis.sh — OntoGeoRAG analysis suite orchestrator.
# Additive: touches NO existing pipeline file. CPU-only by default
# (add --scibert to step 09 on a GPU node if desired).
#
# Usage:
#   bash run_full_analysis.sh <KG_JSON> <CHUNKS_JSONL> <OUTDIR>
# Example (cluster):
#   bash run_full_analysis.sh \
#       output/run12_kg/tiered_kg_enforced.json \
#       output/step1/chunks.jsonl \
#       output/analysis
set -e

KG="${1:?KG json path required}"
CHUNKS="${2:?chunks.jsonl path required}"
OUT="${3:-output/analysis}"

# SLURM hygiene (project convention)
unset SLURM_PROCID SLURM_NTASKS RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

echo "== 08 provenance =="
python 08_rebuild_provenance.py --kg "$KG" --chunks "$CHUNKS" \
    --outdir "$OUT" ${M4_DECISIONS:+--decisions "$M4_DECISIONS"}

KGP="$OUT/kg_with_provenance.json"

echo "== 09 vocabulary =="
python 09_analyze_vocabulary.py --kg "$KGP" --outdir "$OUT"

echo "== 10 descriptors =="
python 10_descriptor_analysis.py --kg "$KGP" --outdir "$OUT"

echo "== 11 relations =="
python 11_relation_analysis.py --kg "$KGP" --outdir "$OUT"

echo "== 12 graph =="
python 12_kg_analysis.py --kg "$KGP" --outdir "$OUT"

echo "== 13 robustness =="
python 13_robustness_analysis.py --kg "$KGP" --outdir "$OUT"

echo "== 14 report =="
python 14_generate_report.py --analysis-dir "$OUT"

echo "DONE — see $OUT/knowledge_report.md"