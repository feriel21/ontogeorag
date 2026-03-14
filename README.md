# OntoGeoRAG — Ontology-Constrained Knowledge Graph Construction from Geological Literature

**OntoGeoRAG** automatically constructs geological knowledge graphs (KGs) from scientific
literature using ontology-constrained LLM extraction with BM25-based RAG, CrossEncoder
reranking, and tiered verification. Developed for mass transport deposit (MTD) interpretation
in seismic data.

> **Feryal Talbi, John Armitage, Alain Rabaute, Jean Charlety, Jean-Noël Vittaut,
> Antoine Bouziat, Sylvie Leroy.**
> *OntoGeoRAG: Ontology-Constrained Knowledge Graph Construction from Geological Literature
> via Retrieval-Augmented Generation.*
> Computers & Geosciences. (under review)

---

## Key Results (Configuration C10 — final)

| Metric | Tier 1 | Tier 1+2 |
|---|---|---|
| Triples | 101 | **153** |
| LB2019 recall | 30.8% (8/26) | **69.2% (18/26)** |
| Descriptor coverage | 8/13 (61.5%) | 10/13 (76.9%) |
| NOT\_SUPPORTED rate $H_{T1}$ | **0.0%** (Wilson CI: [0.0, 3.6]%) | 2.9% |
| Expert relaxed precision | — | **69%** ($\kappa = 0.53$, $n = 50$) |

**CrossEncoder reranking (C10 vs C9):** recall increases from 50.0% → 69.2% (+19.2 pp),
recovering 5 reference edges that BM25 consistently failed to surface across 9 previous
configurations. Retrieval failure, not LLM reasoning, is the primary bottleneck.

---

## Pipeline Architecture

```
PDF corpus (41 papers)
      │
      ▼
01_build_index.py       →  BM25 index (3,386 chunks)
      │
      ▼
02_extract_triples.py   →  BM25 retrieval
                           → [C10 only] CrossEncoder reranking
                           → LLM extraction, Pass A (temp=0.0) + Pass B (temp=0.3)
      │
      ▼
03_verify_triples.py    →  CoT verification (STRONG / WEAK / NOT_SUPPORTED / SKIPPED)
      │
      ▼
04_clean_validate.py    →  Schema validation, lexicon filtering, deduplication
      │
      ▼
05_canonicalize.py      →  SciBERT entity canonicalization (AgglomerativeClustering)
      │
      ▼
06_tiered_fusion.py     →  Tier-1 (both passes) + Tier-2 (one pass) KG assembly
      │
      ▼
07_final_metrics.py     →  Recall, coverage, hallucination rates vs LB2019
```

---

## Repository Structure

```
ontogeorag/
├── pipeline/                   # Core pipeline steps (run in order)
│   ├── 01_build_index.py
│   ├── 02_extract_triples.py   # BM25 + optional CrossEncoder reranker (--rerank)
│   ├── 03_verify_triples.py
│   ├── 04_clean_validate.py
│   ├── 05_canonicalize.py
│   ├── 06_tiered_fusion.py
│   ├── 07_final_metrics.py
│   ├── expB_no_rag.py          # Experiment B: no-RAG parametric memory baseline
│   └── rag/
│       ├── llm_hf.py
│       ├── bm25.py
│       ├── chunking.py
│       ├── constants.py        # RELATION_MAP, ontology schema constants
│       └── schema.py
├── configs/
│   ├── ontology_schema.json        # Relations, entity types
│   ├── lb_reference_edges.json     # LB2019 ground truth (26 evaluation edges)
│   └── descriptor_queries.jsonl    # 249 ontology-guided queries (C10)
├── output/
│   ├── run11_kg/
│   │   ├── tiered_kg_run11.json    # Final C10 KG (153 triples)
│   │   └── metrics_run11.json      # C10 evaluation metrics
│   └── run10_kg/
│       ├── tiered_kg_run10_final.json  # C9 KG (103 triples, recall 50.0%)
│       └── metrics_run10.json
├── slurm/
│   ├── run11_gpu.sh            # C10 SLURM job (A100, ~20 min)
│   └── expB_gpu.sh             # Experiment B SLURM job
├── reference/
│   ├── Table_Supplementary_1_V2.xlsx   # LB2019 manual KG (88 nodes, 173 edges)
│   ├── reference_kg.json
│   └── build_reference_graph.py
└── setup.sh
```

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/feriel21/ontogeorag.git
cd ontogeorag
bash setup.sh
source venv/bin/activate

# 2. Build BM25 index from PDF corpus
python pipeline/01_build_index.py \
    --pdf-dir data/corpus/ \
    --outdir output/step1/

# 3. Extract triples — C9 (BM25 only)
python pipeline/02_extract_triples.py \
    --index-dir output/step1/ \
    --queries configs/descriptor_queries.jsonl \
    --output output/step2/raw_triples.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct --backend hf

# 3b. Extract triples — C10 (BM25 + CrossEncoder reranker)
python pipeline/02_extract_triples.py \
    --index-dir output/step1/ \
    --queries configs/descriptor_queries.jsonl \
    --output output/step2/raw_triples.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct --backend hf \
    --rerank --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --bm25-topn 20 --top-k 5

# 4. Verify triples
python pipeline/03_verify_triples.py \
    --input output/step2/raw_triples.jsonl \
    --output output/step3/verified_triples.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct --backend hf

# 5. Clean and validate
python pipeline/04_clean_validate.py \
    --input output/step3/verified_triples.jsonl \
    --outdir output/step4/

# 6. Canonicalize entities
python pipeline/05_canonicalize.py \
    --input output/step4/canonical_triples.jsonl \
    --output output/step5/canonical_triples.jsonl

# 7. Tiered fusion (dual-pass)
python pipeline/06_tiered_fusion.py \
    --iter-a output/pass_a/canonical_triples.jsonl \
    --iter-b output/pass_b/canonical_triples.jsonl \
    --output output/kg/tiered_kg.json

# 8. Compute metrics vs LB2019 benchmark
python pipeline/07_final_metrics.py \
    --kg output/kg/tiered_kg.json \
    --ref configs/lb_reference_edges.json \
    --output output/kg/metrics.json
```

---

## Pipeline Configurations

| Config | Method | Reranker | Recall T1+2 | $H_{T1}$ | Triples |
|---|---|---|---|---|---|
| C1 | SpaCy SVO | No | 7.7% | n/a | 161 |
| C4 | Qwen-7B | No | 65.4% | 78.0% | 137 |
| C5 | Qwen-7B + BM25 gate | No | 19.2% | 0.0% | 20 |
| C9 | Qwen-7B dual-pass | No | 50.0% | 0.0% | 103 |
| **C10** | **Qwen-7B dual-pass** | **CrossEncoder** | **69.2%** | **0.0%** | **153** |

Full trajectory across all 10 configurations in Appendix C of the paper.

---

## Supported Models

| Model | Role | Notes |
|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | Extraction + verification (primary) | Best calibration; used in C1–C10 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder reranker (C10) | sentence-transformers |
| `allenai/scibert_scivocab_uncased` | Entity canonicalization | AgglomerativeClustering |
| `meta-llama/Llama-3.1-8B-Instruct` | Cross-model verifier (Exp D) | Planned |

---

## Planned Extensions

| ID | Experiment | Purpose |
|---|---|---|
| Exp A | Corpus recoverability annotation | Distinguish corpus gaps from extraction failures |
| Exp B | No-RAG ablation | Quantify RAG contribution vs parametric memory |
| Exp D | Cross-model verifier (Llama-3.1-8B) | Assess self-verification bias |

Scripts: `pipeline/expB_no_rag.py`, `slurm/expB_gpu.sh`.

---

## Benchmark Reference Data

The `reference/` directory contains the Le Bouteiller (2019) supplementary dataset:

| File | Role |
|---|---|
| `Table_Supplementary_1_V2.xlsx` | LB2019 manual KG (88 nodes, 173 edges) |
| `reference_kg.json` | Machine-readable formalization |
| `build_reference_graph.py` | Script that produced `reference_kg.json` |

The 26-edge evaluation benchmark (`configs/lb_reference_edges.json`) was derived by
retaining only text-recoverable LB2019 relations: both entities present in the extraction
lexicon, predicate in the allowed schema, and relation expressible from running text
(not requiring figure or map evidence).

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- `transformers` ≥ 4.41
- `rank-bm25 == 0.2.2`
- `sentence-transformers == 2.7.0`
- `scikit-learn == 1.4.2`
- `spacy == 3.7.4`

See `setup.sh` for full install. Tested on NVIDIA A100 80 GB, CUDA 12.1.

---

## Citation

```bibtex
@article{talbi2026ontogeorag,
  title   = {{OntoGeoRAG}: Ontology-Constrained Knowledge Graph Construction
             from Geological Literature via Retrieval-Augmented Generation},
  author  = {Talbi, Feryal and Armitage, John and Rabaute, Alain and
             Charl{\'e}ty, Jean and Vittaut, Jean-No{\"e}l and
             Bouziat, Antoine and Leroy, Sylvie},
  journal = {Computers \& Geosciences},
  year    = {2026},
  note    = {Under review}
}
```
