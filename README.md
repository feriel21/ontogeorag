# OntoGeoRAG — Ontology-Constrained Knowledge Graph Construction from Geological Literature

**OntoGeoRAG** automatically constructs geological knowledge graphs (KGs) from scientific
literature using ontology-constrained LLM extraction with BM25-based RAG, CrossEncoder
reranking, and tiered verification. Developed for mass transport deposit (MTD) interpretation
in seismic data.

> **Paper:** Talbi et al. (2025). *OntoGeoRAG: Ontology-Constrained Knowledge Graph
> Construction from Geological Literature via Retrieval-Augmented Generation.*
> Submitted to *Computers & Geosciences*.

---

## Key Results (LLM-Rerank — final configuration)

| Metric | Tier 1 | Tier 1+2 |
|---|---|---|
| Triples | 101 | **153** |
| LB2019 recall (34-edge benchmark) | 47.1% (16/34) | **76.5% (26/34)** |
| LB2019 recall (26-edge original benchmark) | 50.0% (13/26) | 69.2% (18/26) |
| Descriptor coverage | 8/13 (61.5%) | 10/13 (76.9%) |
| NOT\_SUPPORTED rate $H_{T1}$ | **0.0%** (Wilson CI: [0.0, 3.6]%) | 2.9% (Wilson CI: [0.9, 7.5]%) |
| Cross-model agreement (Llama-3.1-8B, Exp D) | **96%** (4/100 disagreements) | — |
| Expert validation | pending | pending |

**Memory-only baseline (Experiment B):** 26.5% recall (9/34 edges) from parametric memory
alone, confirming that retrieval grounding is a necessary rather than incidental component
(net gain: +50.0 pp from memory-only to LLM-Rerank).

**CrossEncoder reranking (LLM-Rerank vs LLM-BM25):** recall increases from 47.1% → 76.5%
(+29.4 pp on 34-edge benchmark), recovering 8 reference edges that BM25 consistently failed
to surface. Retrieval quality, not LLM reasoning, is the primary performance bottleneck
within this sentence-level extraction framework.

**Corpus-gap corrected ceiling:** 32/34 = 94.1% (2 edges absent from the 41-paper corpus).
Correcting for corpus gaps: 26/32 recoverable edges recovered = **81.3%**.

---

## Benchmark

The pipeline is evaluated against two benchmarks derived from the Le Bouteiller et al. (2019)
expert knowledge graph:

| Benchmark | Edges | Description |
|---|---|---|
| **26-edge original** | 26 | Direct LB2019 edges: both entities in lexicon, predicate in schema, text-recoverable from a single paper |
| **34-edge extended** | 34 | 26 original + 8 corpus-grounded extended edges (co-occurrence counts: 4–74 chunks per edge) |

The 34-edge benchmark is the **primary evaluation standard** for the final LLM-Rerank
configuration. The 26-edge benchmark is retained for comparing developmental configurations
and for fully independent evaluation.

Benchmark files: `configs/lb_reference_edges.json` (34-edge),
`configs/lb_reference_edges_original26.json` (26-edge).

---

## Pipeline Architecture

```
PDF corpus (41 papers)
      │
      ▼
01_build_index.py       →  BM25 index (3,386 chunks; k1=1.5, b=0.75)
      │
      ▼
02_extract_triples.py   →  BM25 retrieval (top-5 direct, or top-20 → CrossEncoder → top-5)
                           → LLM extraction, Pass A (temp=0.0) + Pass B (temp=0.3)
                           → Qwen 2.5-7B-Instruct, 249 ontology-guided queries
      │
      ▼
03_verify_triples.py    →  CoT verification (STRONG / WEAK / NOT_SUPPORTED / SKIPPED)
                           → Same Qwen-7B model; cross-model check with Llama-3.1-8B (Exp D)
      │
      ▼
04_clean_validate.py    →  Schema validation, lexicon filtering, deduplication
      │
      ▼
05_canonicalize.py      →  SciBERT entity canonicalization (cosine distance < 0.06)
      │
      ▼
06_tiered_fusion.py     →  Tier-1 (both passes verified) + Tier-2 (one pass) KG assembly
      │
      ▼
07_final_metrics.py     →  Recall, coverage, hallucination rates vs LB2019 benchmark
```

---

## Repository Structure

```
ontogeorag/
├── pipeline/                    # Core pipeline steps (run in order)
│   ├── 01_build_index.py
│   ├── 02_extract_triples.py    # BM25 + optional CrossEncoder reranker (--rerank)
│   ├── 03_verify_triples.py
│   ├── 04_clean_validate.py
│   ├── 05_canonicalize.py
│   ├── 06_tiered_fusion.py
│   ├── 07_final_metrics.py
│   ├── expB_no_rag.py           # Experiment B: no-RAG parametric memory baseline
│   └── rag/
│       ├── llm_hf.py
│       ├── bm25.py
│       ├── chunking.py
│       ├── constants.py         # RELATION_MAP, ontology schema constants
│       └── schema.py
├── configs/
│   ├── ontology_schema.json            # Relations, entity types (11 relation types)
│   ├── lb_reference_edges.json         # 34-edge primary evaluation benchmark
│   ├── lb_reference_edges_original26.json  # 26-edge original LB2019 benchmark
│   └── descriptor_queries.jsonl        # 249 ontology-guided queries (4 strategies)
├── output/
│   ├── run11_kg/
│   │   ├── tiered_kg_run11.json        # Final LLM-Rerank KG (153 triples)
│   │   └── metrics_run11.json          # LLM-Rerank evaluation metrics
│   ├── run10_kg/
│   │   ├── tiered_kg_run10_final.json  # LLM-BM25 KG (103 triples, recall 47.1%)
│   │   └── metrics_run10.json
│   ├── expB/
│   │   ├── canonical_expB.jsonl        # Memory-only triples (53)
│   │   └── memory_only_recall_34edge.json  # 26.5% baseline figure
│   └── benchmark_circularity_report.json  # Corpus co-occurrence counts for 8 extended edges
├── slurm/
│   ├── run11_gpu.sh             # LLM-Rerank SLURM job (A100, ~20 min)
│   └── expB_gpu.sh              # Experiment B SLURM job
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

# 3a. Extract triples — LLM-BM25 (BM25 only, no reranking)
python pipeline/02_extract_triples.py \
    --index-dir output/step1/ \
    --queries configs/descriptor_queries.jsonl \
    --output output/step2/raw_triples.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct --backend hf

# 3b. Extract triples — LLM-Rerank (BM25 + CrossEncoder reranker, recommended)
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

## Pipeline Configuration Trajectory

| Config | Method | Reranker | Recall (Tier 1+2) | $H_{T1}$ | Triples |
|---|---|---|---|---|---|
| SVO-baseline | SpaCy SVO | No | 7.7% (26-edge) | n/a | 161 |
| LLM-unverif. | Qwen-7B, no verification | No | 65.4%\* (26-edge) | 78.0% | 137 |
| LLM-strict | Qwen-7B, strict verification | No | 19.2% (26-edge) | 0.0% | 20 |
| LLM-BM25 | Qwen-7B dual-pass | No | 47.1% (34-edge) | 0.0% | 103 |
| **LLM-Rerank** | **Qwen-7B dual-pass** | **CrossEncoder** | **76.5% (34-edge)** | **0.0%** | **153** |
| Memory only (Exp B) | Qwen-7B, no retrieval | — | 26.5% (34-edge) | — | 53 |

\* Raw recall before correcting for 78% NOT\_SUPPORTED rate; effective reliable recall ≈ 14%.

Full configuration trajectory (10 developmental iterations) in Appendix B of the paper.

---

## Failure Mode Analysis (26-edge benchmark)

Among the 8 unmatched edges in LLM-Rerank:

| Failure mode | Count | Description |
|---|---|---|
| Corpus gaps | 2 | Evidence absent from all 3,386 corpus chunks |
| Retrieval failures | 3 | Evidence exists but not surfaced in top-5 |
| Extraction failures | 6 | Evidence retrieved but relation expressed indirectly |

Most striking case: `pore pressure controls slope failure` — present in 84 corpus chunks,
all 5 retrieved passages relevant, but the relation is expressed through multi-step causal
chains (pore pressure → reduced shear strength → instability) rather than a direct predicate.
This is a structural limitation of sentence-level extraction, not a retrieval problem.

The 8 corpus-grounded extended edges (co-occurrence counts: 4–74 chunks) are all recovered
by LLM-Rerank and contribute no additional failure cases.

---

## Supported Models

| Model | Role | Notes |
|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | Extraction + verification (primary) | Used in all LLM configurations |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder reranker (LLM-Rerank) | sentence-transformers; no domain adaptation |
| `allenai/scibert_scivocab_uncased` | Entity canonicalization | Cosine distance threshold 0.06 |
| `meta-llama/Llama-3.1-8B-Instruct` | Cross-model verifier (Experiment D) | 96% agreement with Qwen-7B on 100 Tier-1 triples |

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- `transformers` ≥ 4.41
- `rank-bm25 == 0.2.2`
- `sentence-transformers == 2.7.0`
- `scikit-learn == 1.4.2`
- `spacy == 3.7.4`

See `setup.sh` for full install. Tested on NVIDIA A100 80 GB, CUDA 12.1 (SLURM cluster,
`convergence` partition).

---

## Citation



---

## Acknowledgements

This project is co-funded by the European Union's Horizon Europe research and innovation
programme Cofund SOUND.AI under the Marie Skłodowska-Curie Grant Agreement No. 101081674.

## Environment

The virtual environment is located at `/home/talbi/kg_test/venv` (shared across projects on the cluster). To recreate it elsewhere:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Update the `VENV` variable in `slurm/*.sh` to point to your venv location.
