# OntoGeoRAG — Ontology-Constrained Knowledge Graph Construction from Geological Literature

**OntoGeoRAG** automatically constructs geological knowledge graphs (KGs) from scientific
literature using ontology-constrained LLM extraction with BM25 retrieval, CrossEncoder
reranking, dual-pass tiering and independent cross-family verification. Developed for mass
transport deposit (MTD) interpretation in seismic data.

> **Paper:** Talbi et al. *OntoGeoRAG: Ontology-Constrained Knowledge Graph Construction
> from Geological Literature via Retrieval-Augmented Generation.*
> Submitted to *Computers & Geosciences*.

---

## Key results — run13 (canonical, frozen)

Corpus: **37 peer-reviewed papers**, 1,955 deduplicated chunks.

| Metric | Tier 1 | Tier 1+2 |
|---|---|---|
| Triples (fused, pre-panel) | 90 | **157** |
| Triples (post cross-family panel) | 64 | **151** (+6 quarantined) |
| LB2019 recall — **test split** (17 edges) | — | **82.4 % (14/17)** |
| LB2019 recall — dev split (17 edges) | — | 64.7 % (11/17) |
| LB2019 recall — 34-edge combined | 44.1 % (15/34) | 73.5 % (25/34) raw · 76.5 % (26/34) matching-artefact corrected |
| LB2019 recall — 26-edge original | 34.6 % (9/26) | 73.1 % (19/26) |
| Descriptor coverage (13-term LB2019 set) | 10/13 (76.9 %) | **11/13 (84.6 %)** |
| NOT_SUPPORTED rate H_T1 (self-verification) | **0.0 %** (Wilson CI [0.0, 3.6] %) | — |
| Evidence anchoring (triple → source sentence) | — | **94.9 %** (143/151) |
| Inter-article consensus (≥ 2 papers) | — | 39.1 % |
| Expert validation of KG triples | *pending* | *pending* |

**Evaluation protocol (frozen before the run).** The 34-edge benchmark was split 17/17 into
dev and test, declared and committed in `configs/run13_protocol_declaration.json` **before**
run13 executed. Headline recall is reported on **test**; the dev–test gap measures residual
query-design adaptation. Observed gap is negative (test > dev), i.e. no evidence of
benchmark overfitting — with the caveat that n = 17 per split gives ±5.9 pp per edge.

**Recall is measured on the fused pre-panel KG**, keeping it comparable with earlier
configurations. The cross-family panel is a precision mechanism applied afterwards; it may
quarantine benchmark edges (one in run13) and is not part of the recall protocol.

**Matching-artefact note.** Recall matching is normalized-substring based and therefore
plural-sensitive: run13's canonicalization (`submarine landslides → submarine landslide`)
broke one otherwise-correct benchmark match. Both raw (25/34) and corrected (26/34) figures
are reported.

### Retrieval is the bottleneck, not model size

| Comparison | Result |
|---|---|
| Memory-only baseline (no retrieval, Exp B) | 26.5 % recall (9/34) |
| BM25 only → BM25 + CrossEncoder | 47.1 % → 76.5 % (**+29.4 pp**), 8 reference edges recovered |
| Cross-model agreement (Llama-3.1-8B, Exp D) | 96 % (4/100 disagreements on Tier-1) |

Retrieval quality — not LLM reasoning capacity — is the primary performance bottleneck
within this sentence-level extraction framework.

### Corpus-contamination ablation (run11 → run13)

run11's index contained Jupyter `.ipynb_checkpoints` duplicates of every paper. run13
rebuilt the index from the same chunks with duplicates removed; chunking is bit-identical,
so the only variable is deduplication.

| | run11 (contaminated) | run13 (clean) |
|---|---|---|
| Index | 3,386 chunks (42.3 % duplicated) | 1,955 chunks |
| Top-5 slots occupied by duplicates | 39.8 % | 0 % |
| Effective context diversity | 3.43 / 5 unique passages | 5 / 5 |
| Raw triples per pass | 380 | **446 (+17 %)** |
| Per-pass recall (26-edge, pass A) | 17/26 | **19/26** |
| Fused Tier-1 / Tier-2 | 101 / 52 | 90 / 67 |

Cleaner contexts increase both coverage and cross-pass divergence: more raw triples, a
slightly smaller doubly-verified Tier-1 core.

### Cross-family verification findings

- **Over-interpretation index = 0.976** — of the 42 triples the verifier judges
  NOT_SUPPORTED from the text, 41 were judged PLAUSIBLE without the text. Empirical
  justification for the blind/evidence dual-pass design and for treating the LLM as a
  transducer rather than a knowledge source.
- **Machine inter-judge κ ≈ human inter-expert κ** — Llama × Mistral linear-weighted
  κ = 0.30 (evidence verdicts), against 0.30–0.37 between human experts: the agreement
  ceiling is a property of the task, not a defect of the judges.

---

## Benchmarks

Derived from the Le Bouteiller et al. (2019) expert knowledge graph (88 nodes / 173 edges,
built manually from 41 papers).

| Benchmark | Edges | File |
|---|---|---|
| 26-edge original | 26 | `configs/lb_reference_edges_original26.json` |
| 34-edge extended (primary) | 34 | `configs/lb_reference_edges.json` |
| Frozen dev split | 17 | `configs/lb_dev.json` |
| Frozen test split | 17 | `configs/lb_test.json` |

The 8 extended edges are corpus-grounded (co-occurrence 4–74 chunks each).
LB2019 is used **only** as an external benchmark — never in pipeline construction
(grep-verified). Because LB2019 and this corpus share a literature tradition, the benchmark
measures **recoverability of expert-formalized knowledge**, not discovery.

---

## Pipeline architecture

```
PDF corpus (37 papers)
      │
01_build_index.py        →  chunks.jsonl (1,955 chunks; BM25 rebuilt at query time,
      │                     k1=1.5, b=0.75 — no serialized index object)
      ▼
02_extract_triples.py    →  249 ontology-guided queries × (BM25 top-20 → CrossEncoder
      │                     → top-5) → Qwen2.5-7B-Instruct
      │                     Pass A (temp 0.0) + Pass B (temp 0.3)
      ▼
03_verify_triples.py     →  STRONG / WEAK / NOT_SUPPORTED (same model — degenerate on
      │                     survivors by design; the independent check is M4 below)
      ▼
04_clean_validate.py     →  relation normalization (RELATION_MAP), type constraints,
      │                     closed-world descriptor validation, SciBERT canonicalization
      │                     (cosine < 0.06, LB-vocabulary merge protection), dedup
      │                     → canonical_triples_v5.jsonl
      ▼
05a_rule_canonicalize.py →  deterministic variant merge (plural/hyphen/abbreviation)   [additive]
04c_lexicon_enforce.py   →  closed-world descriptor guard, imports the canonical
      │                     40-term lexicon from constants.py                          [additive]
      ▼
06_tiered_fusion.py      →  Tier-1 (both passes) + Tier-2 (one pass)
06b_attach_provenance.py →  re-attaches retrieval provenance dropped by fusion          [additive]
      ▼
07_final_metrics.py      →  recall (dev / test / 34 / original-26), coverage,
      │                     hallucination
      ▼
m4/ battery              →  Llama-3.1-8B verify (blind + evidence) → Mistral-7B verify
      │                     → two-judge conservative panel → tier reassignment
      ▼
04b_schema_enforce.py    →  entity types mapped onto the 5-type schema, self-loops,
      │                     dedup, relation-signature check                            [additive]
      ▼
analysis_suite/08–17     →  provenance, vocabulary, descriptors, relations, graph,
                            robustness, reports, expert packet
```

`pipeline/05_canonicalize.py` exists but is **not** in the canonical chain: SciBERT
canonicalization runs inside `04_clean_validate.py`, which emits `canonical_triples_v5.jsonl`
directly.

---

## Canonical run13 (frozen)

All reported results trace to one frozen run, tagged **`run13-frozen`**
(`git checkout run13-frozen` to inspect that exact state).

```bash
# full chain (from a GPU allocation, venv active)
bash analysis_suite/run13_pipeline.sh
# resume at a given stage: 2=index 3=extract 4=verify 5=clean 6=fusion+metrics 8=analysis
FROM_STAGE=5 N_PAPERS_EXPECTED=37 bash analysis_suite/run13_pipeline.sh
```

**Index note.** `01_build_index.py` currently leaks memory in this environment
(~1.7 GB / 3 s, any PDF), so run13's index was rebuilt from run11's chunks with duplicates
removed (`analysis_suite/run13_build_index_from_run11.py`). Chunking is bit-identical to
run11 by construction, which is precisely what makes the contamination ablation controlled.
Repairing the PDF parser is required before extending the corpus.

**Offline models.** `transformers`' `_patch_mistral_regex` performs a network call even
under `HF_HUB_OFFLINE=1` when given a hub ID. Always resolve to a local path first:

```bash
MODEL=$(python -c "from huggingface_hub import snapshot_download; \
  print(snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_files_only=True))")
```

**Before any change to `pipeline/`**, run the regression check:

```bash
python analysis_suite/validate_pipeline.py --quick     # CPU, ~2 min
python analysis_suite/validate_pipeline.py --gpu       # adds smoke tests
```

---

## Canonical run13 (frozen)

The results the paper reports trace to a single frozen run, tagged **`run13-frozen`** in
git (`git checkout run13-frozen` to inspect that exact state). The canonical way to
reproduce it end-to-end is:

```bash
bash analysis_suite/run13_pipeline.sh            # all stages
FROM_STAGE=5 bash analysis_suite/run13_pipeline.sh   # resume from stage 5
```

This chains dual-pass (temperature 0.0 + 0.3) extraction over the same 41-paper corpus as
run11 → verification → clean/rule-canonicalize/lexicon-enforce per pass → tiered fusion →
`07_final_metrics.py` against three benchmark splits:

- `configs/lb_dev.json` (17 edges) / `configs/lb_test.json` (17 edges) — the dev/test
  split frozen in `configs/run13_protocol_declaration.json` *before* run13 was executed,
  to guard against query-design overfitting (headline recall is reported on test; the
  dev-test gap measures residual adaptation).
- `configs/lb_extended_benchmark.json` (34-edge, dev+test combined) → `metrics_full34.json`,
  the regression baseline this repo's post-freeze cleanup work is diffed against.

Output lands under `output/run13/kg/`: `tiered_kg_run13.json`, `metrics_dev.json`,
`metrics_test.json`, `metrics_full34.json`.

Before any change to `pipeline/`, run the regression check:
`python analysis_suite/validate_pipeline.py --quick`.

## Documentation

`docs/` holds environment/reproducibility records: `parser_versions.txt` (PDF/embedding
library versions used for run13) and `requirements_frozen_YYYYMMDD.txt` (full pinned
dependency list). See `docs/README_run13.md` for the descriptor-coverage fix applied
during the post-run13 cleanup pass.

---

## Repository Structure

```
ontogeorag/
├── pipeline/                       # core stages, run in order
│   ├── 01_build_index.py           # PDFs → chunks.jsonl (BM25 source)
│   ├── 02_extract_triples.py       # RAG extraction (--reranker, --schema required)
│   ├── 03_verify_triples.py        # self-verification (STRONG/WEAK/NOT_SUPPORTED)
│   ├── 04_clean_validate.py        # normalize, validate, SciBERT canonicalize, dedup
│   ├── 05_canonicalize.py          # standalone canonicalizer (not in canonical chain)
│   ├── 06_tiered_fusion.py         # Tier-1 / Tier-2 assembly
│   ├── 07_final_metrics.py         # recall / coverage / hallucination vs LB2019
│   ├── expB_no_rag.py              # Experiment B: memory-only baseline
│   ├── plot_*.py                   # manuscript figures
│   └── rag/
│       ├── llm_hf.py
│       ├── hybrid_retriever.py  # BM25+dense+CrossEncoder (--hybrid, experimental)
│       ├── chunking.py
│       ├── constants.py         # RELATION_MAP, ontology schema constants
│       └── schema.py
├── m4/                          # Independent cross-family verifier (Llama-3.1-8B)
├── analysis_suite/              # run13 orchestration + post-hoc analysis (see below)
├── docs/                        # frozen dependency versions, parser_versions.txt
├── configs/
│   ├── ontology_schema.json, descriptor_queries.jsonl (249)
│   ├── lb_reference_edges.json (34), lb_reference_edges_original26.json (26)
│   ├── lb_dev.json, lb_test.json, run13_protocol_declaration.json   # frozen protocol
├── docs/
│   ├── PIPELINE.md                 # stage-by-stage process documentation
│   ├── VALIDATION_METRICS.md       # metrics, thresholds, justification
│   ├── KG_VALIDATION.md            # KG verification & validation process
│   ├── GEOLOGIST_GUIDE.md          # how to read the KG (no ML background needed)
│   ├── README_run13.md             # descriptor-coverage fix, run13 decisions
│   └── parser_versions.txt, requirements_frozen_*.txt
├── output/
│   ├── step1/                      # run11 chunks (source of the run13 index)
│   ├── run11_*/                    # ablation baseline
│   ├── run13/{step1,pass_a,pass_b,kg,m4,m4_mistral,m4_panel,analysis,expert_packet}
│   ├── m4*, kg_final/, analysis/   # run11-era verification outputs
│   └── _archive/                   # runs 8–12, exp*, iter1 (development trajectory)
├── reference/                      # LB2019 material (Table_Supplementary_1_V2.xlsx,
│                                   # reference_kg.json, build_reference_graph.py)
├── slurm/                          # job scripts; run11_gpu.sh is the canonical invocation
├── tools/                          # one-off analysis scripts
└── CLAUDE.md, pyproject.toml, setup.sh
```

---

## Quick start

```bash
git clone https://github.com/feriel21/ontogeorag.git && cd ontogeorag
bash setup.sh && source venv/bin/activate
export PYTHONPATH=$PWD                       # required: pipeline/ imports pipeline.rag.*

# 1. index
python pipeline/01_build_index.py --pdf-dir data/corpus/ --outdir output/step1/

# 2. extraction (canonical invocation — matches slurm/run11_gpu.sh)
python -u pipeline/02_extract_triples.py \
    --index-dir output/step1/ \
    --schema    configs/ontology_schema.json \
    --queries   configs/descriptor_queries.jsonl \
    --output    output/pass_a/raw_triples.jsonl \
    --model     Qwen/Qwen2.5-7B-Instruct --backend hf \
    --top-k 5 --bm25-topn 20 --min-bm25 2.0 --temperature 0.0 \
    --reranker  cross-encoder/ms-marco-MiniLM-L-6-v2
#    repeat with --temperature 0.3 → output/pass_b/raw_triples.jsonl

# 3. verification
python -u pipeline/03_verify_triples.py \
    --input output/pass_a/raw_triples.jsonl \
    --output output/pass_a/verified_triples.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct --backend hf

# 4. clean + canonicalize (emits canonical_triples_v5.jsonl)
python -u pipeline/04_clean_validate.py \
    --input output/pass_a/verified_triples.jsonl --outdir output/pass_a/

# 5. fusion
python pipeline/06_tiered_fusion.py \
    --iter-a output/pass_a/canonical_triples_v5.jsonl \
    --iter-b output/pass_b/canonical_triples_v5.jsonl \
    --output output/kg/tiered_kg.json

# 6. metrics (benchmark path is read from configs/lb_reference_edges.json)
python pipeline/07_final_metrics.py \
    --kg output/kg/tiered_kg.json --output output/kg/metrics.json
```

Cross-family verification (`m4/`) and the analysis suite are documented in
`docs/PIPELINE.md`.

---

## Configuration trajectory

| Config | Method | Reranker | Recall (Tier 1+2) | H_T1 | Triples |
|---|---|---|---|---|---|
| SVO baseline | spaCy SVO | no | 7.7 % (26-edge) | n/a | 161 |
| LLM unverified | Qwen-7B, no verification | no | 65.4 %* (26-edge) | 78.0 % | 137 |
| LLM strict | Qwen-7B, strict verification | no | 19.2 % (26-edge) | 0.0 % | 20 |
| LLM-BM25 | Qwen-7B dual-pass | no | 47.1 % (34-edge) | 0.0 % | 103 |
| LLM-Rerank (run11) | Qwen-7B dual-pass | CrossEncoder | 76.5 % (34-edge) | 0.0 % | 153 |
| **run13 (clean corpus)** | **Qwen-7B dual-pass** | **CrossEncoder** | **82.4 % (test split)** | **0.0 %** | **157 → 151** |
| Memory only (Exp B) | Qwen-7B, no retrieval | — | 26.5 % (34-edge) | — | 53 |

\* Raw recall before correcting for a 78 % NOT_SUPPORTED rate; effective reliable recall ≈ 14 %.

---

## Failure-mode analysis (26-edge benchmark, run11)

| Failure mode | Count | Description |
|---|---|---|
| Corpus gaps | 2 | Evidence absent from every chunk |
| Retrieval failures | 3 | Evidence present but never in top-5 |
| Extraction failures | 6 | Evidence retrieved, relation expressed indirectly |

Most instructive case: `pore pressure controls slope failure` — present in 84 chunks, all
five retrieved passages relevant, but the relation is expressed through a multi-step causal
chain (pore pressure → reduced shear strength → instability). A structural limit of
sentence-level extraction, not a retrieval problem.

---

## Validation status

| Layer | Status |
|---|---|
| Textual grounding (triple → source sentence) | **validated** — 94.9 % anchored |
| Cross-family textual support (Llama + Mistral panel) | **validated** — 84 accept / 67 uncertain / 6 quarantined |
| Recoverability vs expert ontology (LB2019, frozen split) | **validated** — 82.4 % test |
| Schema conformity (5 entity types, relation signatures) | **validated** — 0 violations |
| **Geological correctness of individual triples** | **not yet validated** — see below |

Section 4.4's existing expert verdicts were recorded against hand-written statements with no
attached triples and could not be joined to the KG. A stratified, blinded expert packet
(`analysis_suite/17_build_expert_packet.py`) samples across tier × confidence strata, carries
each triple explicitly, and withholds the pipeline's own confidence from the reviewer so that
later score validation stays non-circular. Until it returns, the KG should be described as a
**structured, traceable, textually verified index of literature assertions** — not as an
expert-validated geological model.

---

## Supported models

| Model | Role |
|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | extraction + self-verification |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder reranker (no domain adaptation) |
| `allenai/scibert_scivocab_uncased` | entity canonicalization (cosine < 0.06) |
| `meta-llama/Llama-3.1-8B-Instruct` | cross-family verifier (M4) |
| `mistralai/Mistral-7B-Instruct-v0.3` | second panel judge |

---

## Requirements

Python ≥ 3.10, PyTorch ≥ 2.1, `transformers` ≥ 4.41, `rank-bm25==0.2.2`,
`sentence-transformers==2.7.0`, `scikit-learn==1.4.2`, `spacy==3.7.4`, `networkx`.
Pinned versions: `docs/requirements_frozen_*.txt`. Tested on NVIDIA A100 (SLURM,
`convergence` partition); allocate ≥ 32 GB RAM for indexing.

---

## Reusing OntoGeoRAG for a new geological domain

The pipeline is domain-agnostic; four configuration files are MTD-specific:

| File | Role |
|---|---|
| `configs/ontology_schema.json` | relation and entity types |
| `configs/descriptor_queries.jsonl` | what the pipeline searches for (249 queries) |
| `configs/lb_reference_edges.json` | evaluation benchmark (optional — recall is skipped if absent) |
| `pipeline/rag/constants.py` | `KNOWN_DESCRIPTORS` (closed-world extraction set) and `LB2019_BENCHMARK_DESCRIPTORS` (benchmark subset) |

Replace those four and the code runs unchanged. Query files need one JSON object per line
with `query`, `strategy` (`descriptor` / `causal` / `context` / `profile`) and `focus`.

---

## Acknowledgements

Co-funded by the European Union's Horizon Europe research and innovation programme Cofund
SOUND.AI, Marie Skłodowska-Curie Grant Agreement No. 101081674.
