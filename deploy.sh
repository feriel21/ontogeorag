#!/bin/bash
# deploy.sh — Run this on the cluster to set up the clean ontogeorag repo
# Usage: bash deploy.sh
#
# What this does:
#   1. Creates ~/ontogeorag/ with the clean repo structure
#   2. Copies final pipeline scripts from your existing kg_test/ dirs
#   3. Copies experiment scripts from enhanc/
#   4. Copies configs and key outputs
#   5. Creates .gitignore
#   6. Initializes git repo
#
# After running: cd ~/ontogeorag && git remote add origin YOUR_URL && git push

set -e

SRC=~/kg_test
DEST=~/ontogeorag

echo "=========================================="
echo " OntoGeoRAG Repo Setup"
echo " Source: $SRC"
echo " Dest:   $DEST"
echo "=========================================="

# ── 1. Clone/create structure ─────────────────────────────────────────
if [ -d "$DEST/.git" ]; then
    echo "[!] $DEST is already a git repo. Updating files only."
else
    mkdir -p $DEST
    cd $DEST && git init
fi

mkdir -p $DEST/{pipeline/rag,experiments,evaluation,configs,slurm,logs,output}

# ── 2. Copy FINAL pipeline scripts (src__ is the latest) ─────────────
echo "[2] Copying pipeline scripts..."

# Core pipeline — use src__ versions (most recent)
cp $SRC/src__/02_rag_extract_triples_v4.py   $DEST/pipeline/02_extract_triples_v4_orig.py
cp $SRC/src__/02b_verify_triples_v5.py        $DEST/pipeline/03_verify_triples_v5_orig.py
cp $SRC/src__/03_clean_validate.py             $DEST/pipeline/04_clean_validate_orig.py
cp $SRC/src__/04_canonicalize.py               $DEST/pipeline/05_canonicalize_orig.py
cp $SRC/src__/step1_tiered_fusion.py           $DEST/pipeline/06_tiered_fusion_orig.py
cp $SRC/src__/step3_final_metrics.py           $DEST/pipeline/07_final_metrics_orig.py
cp $SRC/src__/generate_all_figures.py          $DEST/evaluation/generate_figures_orig.py

# RAG utilities
cp $SRC/src_/rag/llm_hf.py      $DEST/pipeline/rag/llm_hf_orig.py
cp $SRC/src_/rag/llm_ollama.py  $DEST/pipeline/rag/llm_ollama.py
cp $SRC/src_/rag/bm25.py        $DEST/pipeline/rag/bm25.py
cp $SRC/src_/rag/chunking.py    $DEST/pipeline/rag/chunking.py
cp $SRC/src_/rag/schema.py      $DEST/pipeline/rag/schema.py
cp $SRC/src_/rag/validate.py    $DEST/pipeline/rag/validate.py
# ── Reference data (provenance, not active pipeline) ──────────────────
echo "[+] Copying reference data..."
mkdir -p $DEST/reference
cp $SRC/reference/Table_Supplementary_1_V2.xlsx  $DEST/reference/
cp $SRC/reference/reference_kg.json              $DEST/reference/
cp $SRC/reference/build_reference_graph.py       $DEST/reference/
cp $SRC/reference/contolled_vocab.py             $DEST/reference/
cp $SRC/reference/merge_vocab.py                 $DEST/reference/

# ── 3. Copy experiment scripts ────────────────────────────────────────
echo "[3] Copying experiment scripts..."
cp $SRC/enhanc/exp_a_corpus_recoverability.py    $DEST/experiments/exp_a_recoverability.py
cp $SRC/enhanc/exp_b_no_rag_ablation.py           $DEST/experiments/exp_b_no_rag_ablation.py
cp $SRC/enhanc/exp_d_cross_model_verification.py  $DEST/experiments/exp_d_cross_model.py

# ── 4. Copy configs ───────────────────────────────────────────────────
echo "[4] Copying configs..."
cp ~/mtd-kg-pipeline/mtd-kg-pipeline/configs/lb_reference_edges.json  $DEST/configs/ 2>/dev/null || true
cp ~/mtd-kg-pipeline/mtd-kg-pipeline/configs/lexicon.json              $DEST/configs/ 2>/dev/null || true
cp ~/mtd-kg-pipeline/mtd-kg-pipeline/configs/schema_step1.json         $DEST/configs/ontology_schema.json 2>/dev/null || true

# Copy descriptor queries (Run 7)
if [ -f "$SRC/output/step7/queries_v7.jsonl" ]; then
    cp $SRC/output/step7/queries_v7.jsonl $DEST/configs/descriptor_queries.jsonl
elif [ -f "$SRC/schema_seed_output/queries_v4.jsonl" ]; then
    cp $SRC/schema_seed_output/queries_v4.jsonl $DEST/configs/descriptor_queries.jsonl
fi

# ── 5. Copy key result outputs (lightweight JSON only) ────────────────
echo "[5] Copying key outputs..."
mkdir -p $DEST/output/{exp_a,exp_b,exp_d,improved_kg,figures}
cp $SRC/output/exp_a/exp_a_recoverability_table.csv      $DEST/output/exp_a/ 2>/dev/null || true
cp $SRC/output/exp_a/exp_a_recoverability_details.json   $DEST/output/exp_a/ 2>/dev/null || true
cp $SRC/output/exp_b/exp_b_stats.json                    $DEST/output/exp_b/ 2>/dev/null || true
cp $SRC/output/exp_b_full/exp_b_stats.json               $DEST/output/exp_b/exp_b_full_stats.json 2>/dev/null || true
cp $SRC/output/exp_d/exp_d_stats.json                    $DEST/output/exp_d/ 2>/dev/null || true
cp $SRC/output/exp_d_llama/exp_d_stats.json              $DEST/output/exp_d/exp_d_llama_stats.json 2>/dev/null || true
cp $SRC/output/exp_d_mistral/exp_d_stats.json            $DEST/output/exp_d/exp_d_mistral_stats.json 2>/dev/null || true
cp $SRC/output/improved_kg/tiered_kg_final.json          $DEST/output/improved_kg/ 2>/dev/null || true


# ── Corpus placeholder (PDFs excluded — copyright) ────────────────────
echo "[+] Creating corpus data/ structure..."
mkdir -p $DEST/data/{full_corpus,small_corpus}
```

**In `.gitignore`**, add:
```
# Corpus PDFs (copyright — not distributable)
data/full_corpus/
data/extra_full_corpus/
data/small_corpus/

# Keep placeholder README
!data/README.md
!data/full_corpus/README.md

# BM25 index derived from PDFs (also not distributable)
output/step1/
output/step2/

# ── 6. Create .gitignore ──────────────────────────────────────────────
echo "[6] Creating .gitignore..."
cat > $DEST/.gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
venv/

# Large data files
output/step1/
output/step2/
output/full_corpus/
output/figures/
output/figures_v4/
*.jsonl
!output/exp_a/*.json
!output/exp_b/*.json
!output/exp_d/*.json
!output/improved_kg/*.json
!configs/*.jsonl

# Models (downloaded by HuggingFace)
~/.cache/huggingface/

# SLURM logs
logs/slurm_*.out
logs/slurm_*.err

# Jupyter
*.ipynb
EOF

# ── 7. Initial git commit ─────────────────────────────────────────────
echo "[7] Creating initial git commit..."
cd $DEST

git add -A
git commit -m "Initial commit: OntoGeoRAG pipeline v1.0

Pipeline (Run 7 final):
- 02_extract_triples: BM25 RAG + Qwen-7B extraction (226 queries)
- 03_verify_triples:  CoT GraphJudge-inspired verification
- 04_clean_validate:  Ontology-constrained filtering
- 05_canonicalize:    SciBERT entity clustering
- 06_tiered_fusion:   Tier1 (verified) + Tier2 (implied) KG
- 07_final_metrics:   LB2019 recall, descriptor coverage, hallucination

Experiments:
- EXP-A: All 26 LB2019 edges corpus-present (BM25 score >= 2.0)
- EXP-B: RAG reduces NOT_SUPPORTED rate +16.6pp (54.5% -> 37.9%)
- EXP-D: Qwen-7B calibrated; Llama never outputs NOT_SUPPORTED (degenerate verifier)
- EXP-E: Llama-3.1-8B extraction comparison (NEW)

Results (Tier1/Tier1+2): 69/105 triples, 11/12 descriptors, 23.1%/34.6% recall, 0%/2.9% hallucination
"

echo ""
echo "=========================================="
echo " Done! Next steps:"
echo "   cd ~/ontogeorag"
echo "   git remote add origin https://github.com/YOUR_USERNAME/ontogeorag.git"
echo "   git push -u origin main"
echo ""
echo "   # To run EXP-E (Llama extraction):"
echo "   sbatch slurm/run_exp_e.sh"
echo "=========================================="