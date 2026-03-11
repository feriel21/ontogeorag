# OntoGeoRAG — Ontology-Constrained Knowledge Graph Construction from Geological Literature

**OntoGeoRAG** automatically constructs geological knowledge graphs (KGs) from scientific
literature using ontology-constrained LLM extraction with BM25-based RAG and tiered
verification. Developed for mass transport deposit (MTD) interpretation in seismic data.

> *OntoGeoRAG: Unsupervised and Knowledge-Infused
> Interpretation of Seismic Patterns using Machine Learning and Knowledge Graphs.*
> Computers & Geosciences. (under review)

---

## Repository structure

```
ontogeorag/
├── pipeline/               # Core pipeline steps (run in order)
│   ├── 01_build_index.py       # Ingest PDFs → BM25 index
│   ├── 02_extract_triples.py   # RAG extraction (BM25 + LLM)
│   ├── 03_verify_triples.py    # CoT verification (GraphJudge-inspired)
│   ├── 04_clean_validate.py    # Filtering, dedup, type-checking
│   ├── 05_canonicalize.py      # Entity canonicalization (SciBERT clustering)
│   ├── 06_tiered_fusion.py     # Merge runs → tiered KG (Tier1/Tier2)
│   ├── 07_final_metrics.py     # Compute article metrics vs LB2019 ground truth
│   └── rag/                    # Shared utilities
│       ├── llm_hf.py               # HuggingFace LLM backend
│       ├── llm_ollama.py           # Ollama LLM backend
│       ├── bm25.py                 # BM25 index builder
│       ├── chunking.py             # PDF chunking
│       └── schema.py               # Ontology schema helpers
├── experiments/            # Ablation & validation experiments
│   ├── exp_a_recoverability.py     # EXP-A: corpus recoverability ceiling
│   ├── exp_b_no_rag_ablation.py    # EXP-B: RAG vs no-RAG hallucination
│   ├── exp_d_cross_model.py        # EXP-D: cross-model verification bias
│   └── exp_e_llama_extraction.py   # EXP-E: Llama-3.1-8B extraction comparison
├── evaluation/
│   ├── compute_metrics.py          # Recall, coverage, hallucination rates
│   └── generate_figures.py         # Publication-quality figures
├── configs/
│   ├── ontology_schema.json        # Relations, entity types
│   ├── lb_reference_edges.json     # LB2019 ground truth (26 edges)
│   └── descriptor_queries.jsonl    # 226 BM25 queries (Run 7)
├── slurm/
│   ├── run_pipeline.sh             # Full pipeline SLURM job
│   ├── run_exp_b.sh                # EXP-B SLURM job
│   ├── run_exp_d.sh                # EXP-D SLURM job
│   └── run_exp_e.sh                # EXP-E (Llama) SLURM job
└── setup.sh                        # One-shot environment setup
```

---

## Quick start

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/ontogeorag.git
cd ontogeorag
bash setup.sh

# 2. Run full pipeline
source venv/bin/activate

python pipeline/01_build_index.py \
    --pdf-dir data/corpus/ \
    --outdir output/step1/

python pipeline/02_extract_triples.py \
    --index-dir output/step1/ \
    --schema configs/ontology_schema.json \
    --queries configs/descriptor_queries.jsonl \
    --output output/step2/raw_triples.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct --backend hf

python pipeline/03_verify_triples.py \
    --input output/step2/raw_triples.jsonl \
    --output output/step3/verified_triples.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct --backend hf

python pipeline/04_clean_validate.py \
    --input output/step3/verified_triples.jsonl \
    --outdir output/step4/

python pipeline/05_canonicalize.py \
    --input output/step4/canonical_triples.jsonl \
    --output output/step5/canonical_triples.jsonl \
    --map output/step5/canonical_map.json

python pipeline/06_tiered_fusion.py \
    --iter-a output/step4/canonical_triples.jsonl \
    --iter-b output/step5/canonical_triples.jsonl \
    --output output/kg/tiered_kg.json

python pipeline/07_final_metrics.py \
    --kg output/kg/tiered_kg.json \
    --output output/kg/metrics.json
```

---

## Supported models

| Model | Role | Notes |
|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | Extraction + verification (primary) | Best calibration |
| `meta-llama/Llama-3.1-8B-Instruct` | Extraction comparison (EXP-E) | Over-permissive verifier |
| `Qwen/Qwen2.5-1.5B-Instruct` | Cross-verification baseline | Too lenient |
| `mistralai/Mistral-7B-Instruct-v0.3` | Cross-verification baseline | Degenerate (rejects all) |

Switch models via `--model MODEL_NAME --backend hf` on any pipeline step.

---

## Key results (Run 7, submitted manuscript)

| Metric | Tier 1 (verified) | Tier 1+2 |
|---|---|---|
| Triples | 69 | 105 |
| Descriptor coverage | 11/13 (84.6%) | 12/13 (92.3%) |
| Recall vs LB2019 | 23.1% | 34.6% |
| Hallucination rate | 0.0% | 2.9% |

**Ablation (EXP-B):** RAG reduces NOT_SUPPORTED rate by +16.6pp (54.5% → 37.9%).  
**Recoverability (EXP-A):** All 26 LB2019 edges are corpus-present (BM25 score ≥ 2.0).  
**Cross-model (EXP-D):** Qwen-7B produces calibrated 3-way verdicts; Llama-3.1-8B
never outputs NOT_SUPPORTED (verification degenerate at 0% rejection rate).

---

## Provenance & Reference Data

The `reference/` directory contains the Le Bouteiller (2019) supplementary
dataset and the scripts that bootstrapped the ontology:

| File | Role |
|---|---|
| `Table_Supplementary_1_V2.xlsx` | Primary source — LB2019 manual KG (88 nodes, 173 edges) |
| `reference_kg.json` | Machine-readable formalization of the Excel |
| `build_reference_graph.py` | Script that produced `reference_kg.json` |

These are **not part of the active pipeline**. The derived artifacts used
by the pipeline are:
- `configs/lb_reference_edges.json` — the 26 evaluation edges
- `configs/ontology_schema.json` — the ontology schema
- `pipeline/rag/constants.py` — hardcoded ground truth in code

They are included for full reproducibility and reviewer inspection.

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- HuggingFace `transformers` ≥ 4.40
- `rank-bm25`, `sentence-transformers`, `scikit-learn`
- Optional: Ollama (local inference server)

See `setup.sh` for full install.
