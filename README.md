# OntoGeoRAG — Ontology-Constrained Knowledge Graph Construction from Geological Literature

**OntoGeoRAG** automatically constructs geological knowledge graphs (KGs) from scientific
literature using ontology-constrained LLM extraction with BM25 retrieval, CrossEncoder
reranking, dual-pass tiering and independent cross-family verification. Developed for mass
transport deposit (MTD) interpretation in seismic data.



---

## Key results — run13 (canonical, frozen)

Corpus: **37 peer-reviewed papers**, 1,955 deduplicated chunks.

| Metric | Tier 1 | Tier 1+2 |
|---|---|---|
| Triples (fused, pre-panel) | 90 | **157** |
| Triples (post cross-family panel) | 64 | **151** (+6 quarantined) |
| LB2019 recall — **test split** (17 edges) | — | **82.4 % (14/17)** |
| LB2019 recall — dev split (17 edges) | — | 64.7 % (11/17) |
| LB2019 recall — 26-edge original | 34.6 % (9/26) | **73.1 % (19/26)** |
| LB2019 recall — 34-edge combined | 44.1 % (15/34) | 73.5 % (25/34) raw · 76.5 % (26/34) artefact-corrected |
| Descriptor coverage (13-term LB2019 set) | 10/13 | **11/13** — 11/12 of the *reachable* set, see note 3 |
| NOT_SUPPORTED rate H_T1 (self-verification) | **0.0 %** (Wilson CI [0.0, 3.6] %) | — |
| Evidence anchoring (triple → source sentence) | — | **94.7 %** (143/151) |
| Inter-article consensus (≥ 2 papers) | — | 39.1 % |
| Expert validation of KG triples | *pending* | *pending* |

**Evaluation protocol (frozen before the run).** The 34-edge benchmark was split 17/17 into
dev and test, declared and committed in `configs/run13_protocol_declaration.json` **before**
run13 executed. Headline recall is reported on **test**; the dev–test gap measures residual
query-design adaptation. The observed gap is negative (test > dev): no evidence of benchmark
overfitting — with the caveat that n = 17 per split gives ±5.9 pp per edge.

**Recall is measured on the fused pre-panel KG**, keeping it comparable with earlier
configurations. The cross-family panel is a precision mechanism applied afterwards; it may
quarantine benchmark edges (one in run13) and is not part of the recall protocol.

**Three known matching-protocol artefacts, reported rather than silently corrected:**
1. *Plural sensitivity* — matching is normalized-substring based, so canonicalization
   (`submarine landslides → submarine landslide`) breaks an otherwise-correct match. The
   edge is in the KG but counted as missed. Both raw (25/34) and corrected (26/34) are given.
2. *Asymmetric synonyms* — `DESCRIPTOR_SYNONYMS` is applied in `coverage()` but not in
   `recall()`. Documented rather than fixed, to preserve comparability with earlier runs.
3. *Reachable descriptor ceiling is 12, not 13* — `stratified` is mapped onto `layered`
   before counting and can never be reported as found. Coverage is therefore 11/12 (91.7 %)
   of the reachable set.

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
so the only variable is deduplication. **Both runs are measured with the same, corrected
code.**

| | run11 (contaminated) | run13 (clean) |
|---|---|---|
| Index | 3,386 chunks (42.3 % duplicated) | 1,955 chunks |
| Top-5 slots occupied by duplicates | 39.8 % | 0 % |
| Effective context diversity | 3.42 / 5 unique passages | 5 / 5 |
| Raw triples per pass | 380 | **446 (+17 %)** |
| Per-pass recall (26-edge, pass A) | 17/26 | **19/26** |
| **Recall, LB2019 original (26)** | 18/26 | **19/26** |
| Recall, 34-edge | 26/34 | 25/34 raw (26/34 corrected) |
| — of which the 8 in-house extensions | 8/8 | 6/8 |
| Descriptor coverage, Tier-1 | 11/13 | 10/13 |
| Fused Tier-1 / Tier-2 | 101 / 52 | 90 / 67 |

Deduplication improves recovery of the **independent** expert benchmark (18 → 19/26) and
raises extraction yield by 17 %, at the cost of a narrower doubly-verified core: cleaner
contexts increase coverage and cross-pass divergence simultaneously. Both edges lost on the
34-edge benchmark belong to our own extensions, one of them through the plural artefact.

### Cross-family verification findings

- **Over-interpretation index = 0.976** — of the 42 triples the verifier judges
  NOT_SUPPORTED from the text, 41 were judged PLAUSIBLE without the text. Empirical
  justification for the blind/evidence dual-pass design and for treating the LLM as a
  transducer rather than a knowledge source.
- **Machine inter-judge agreement sits at or below the human inter-expert level** —
  Llama × Mistral κ ranges 0.14–0.30 depending on the variant (only the linear-weighted
  evidence κ, 0.30, reaches the human band of 0.30–0.37). The agreement ceiling is a
  property of the task, not a defect of the judges.
- **The tier predicts an independent verifier's judgement** — 39 % of Tier-1 triples are
  SUPPORTED by the cross-family verifier against 9 % of Tier-2 (49 % of Tier-2 are
  NOT_SUPPORTED). A purely sampling-based criterion — does the triple survive both passes? —
  anticipates the verdict of a different model family.

### Causal-chain coherence

Physical mechanisms are recoverable as **paths**, not only as isolated arcs: the causal
subgraph (80 arcs over 104 nodes) contains **29 multi-arc chains, 6 of them entirely
Tier-1** (after merging one nominalization duplicate). The canonical sequence
`earthquake —triggers→ slope failure —causes→ mass transport deposit` has maximal
confidence on every arc; the longest chain spans four arcs
(`gas hydrate dissolution → excess pressure → formation stress gathering → developing MTDs`).
Composition emerges from independently extracted sentence-level assertions — no query asks
for a chain. Reproduce with `analysis_suite/19_causal_chains.py`.

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
05a_rule_canonicalize.py →  deterministic variant merge (plural/hyphen/abbrev)  [additive]
04c_lexicon_enforce.py   →  closed-world descriptor guard, imports the canonical
      │                     40-term lexicon from constants.py                   [additive]
      ▼
06_tiered_fusion.py      →  Tier-1 (both passes) + Tier-2 (one pass)
06b_attach_provenance.py →  re-attaches retrieval provenance dropped by fusion  [additive]
      ▼
07_final_metrics.py      →  recall (dev / test / 34 / original-26), coverage,
      │                     hallucination
      ▼
m4/ battery              →  Llama-3.1-8B verify (blind + evidence) → Mistral-7B verify
      │                     → two-judge conservative panel → tier reassignment
      ▼
04b_schema_enforce.py    →  entity types mapped onto the 5-type schema, self-loops,
      │                     dedup, relation-signature check                     [additive]
      ▼
analysis_suite/08–19     →  provenance, vocabulary, descriptors, relations, graph,
                            robustness, reports, expert packet, figures, causal chains
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

Outputs land under `output/run13/`: `kg/tiered_kg_run13.json`, `kg/metrics_{dev,test,full34}.json`,
`m4_panel/tiered_kg_m4.json`, `kg/tiered_kg_run13_enforced.json`, `analysis/`, `expert_packet/`.

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

It verifies artefact integrity, schema conformity and — since a merge once silently
duplicated the function computing the headline metric — that no top-level definition is
shadowed anywhere in `pipeline/`, `m4/` or `analysis_suite/`.

---

## Repository structure

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
│       ├── constants.py            # RELATION_MAP, KNOWN_DESCRIPTORS (40),
│       │                           # LB2019_BENCHMARK_DESCRIPTORS (13) — single source
│       ├── llm_hf.py, chunking.py, schema.py
│       └── hybrid_retriever.py     # BM25+dense+CrossEncoder (experimental, run12 only)
├── m4/                             # independent cross-family verifier
│   ├── m4_verify.py                # dual-pass blind + evidence, any HF model
│   ├── m4_panel.py                 # multi-judge conservative vote + inter-judge κ
│   ├── m4_integrate_tiers.py       # tier reassignment from panel decisions
│   ├── m4_aggregate.py, m4_metrics.py, m4_calibration.py, m4_negatives.py
│   ├── m4_direction_check.py, m4_quote_verify.py, m4_inspection_list.py
│   ├── m4_figures.py, m4_figures_v2.py
│   └── slurm/m4_gpu.sh
├── analysis_suite/                 # run13 orchestration + post-hoc analysis
│   ├── run13_pipeline.sh           # orchestrator (FROM_STAGE, N_PAPERS_EXPECTED)
│   ├── run13_build_index_from_run11.py, run13_01_make_devtest_split.py
│   ├── 04b_schema_enforce.py       # entity-type schema enforcement
│   ├── 04c_lexicon_enforce.py      # closed-world descriptor guard
│   ├── 05a_rule_canonicalize.py    # deterministic variant merge
│   ├── 06b_attach_provenance.py    # re-attach provenance lost at fusion
│   ├── 08_rebuild_provenance.py    # evidence anchoring + consensus + confidence
│   ├── 09_analyze_vocabulary.py    # 3-detector fragmentation audit
│   ├── 10_descriptor_analysis.py, 11_relation_analysis.py, 12_kg_analysis.py
│   ├── 13_robustness_analysis.py   # paper bootstrap + Heaps fits
│   ├── 14_generate_report.py       # knowledge_report.md + discussion.md
│   ├── 15_confidence_validation.py # confidence × expert × M4
│   ├── 16_map_statements_to_triples.py
│   ├── 17_build_expert_packet.py   # stratified blinded expert packet
│   ├── 18_paper_figures.py         # manuscript figures
│   ├── 19_causal_chains.py         # mechanisms as paths
│   ├── kg_io.py                    # format-tolerant KG I/O + figure style
│   ├── validate_pipeline.py        # stage-by-stage validation harness
│   └── run_full_analysis.sh
├── configs/
│   ├── ontology_schema.json, descriptor_queries.jsonl (249)
│   ├── lb_reference_edges.json (34), lb_reference_edges_original26.json (26)
│   └── lb_dev.json, lb_test.json, run13_protocol_declaration.json   # frozen protocol
├── docs/
│   ├── PIPELINE.md                 # stage-by-stage process documentation
│   ├── VALIDATION_METRICS.md       # metrics, thresholds, justification
│   ├── KG_VALIDATION.md            # KG verification & validation process
│   ├── GEOLOGIST_GUIDE.md          # how to read the KG (no ML background needed)
│   ├── README_run13.md             # descriptor-coverage fix, run13 decisions
│   └── parser_versions.txt, requirements_frozen_*.txt
├── figures/paper/                  # manuscript figures + figure_manifest.md
├── output/
│   ├── step1/                      # run11 chunks (source of the run13 index)
│   ├── run11_*/                    # ablation baseline
│   ├── run13/{step1,pass_a,pass_b,kg,m4,m4_mistral,m4_panel,analysis,expert_packet}
│   ├── m4*, kg_final/, analysis/   # run11-era verification outputs
│   └── _archive/                   # runs 8–12, exp*, iter1 (development trajectory)
├── reference/                      # LB2019 material
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
sentence-level extraction, not a retrieval problem — and one that the causal-chain analysis
shows is an exception rather than the rule.

---

## Validation status

| Layer | Status |
|---|---|
| Textual grounding (triple → source sentence) | **validated** — 94.7 % anchored (143/151) |
| Cross-family textual support (Llama + Mistral panel) | **validated** — 84 accept / 67 uncertain / 6 quarantined |
| Recoverability vs expert ontology (LB2019, frozen split) | **validated** — 82.4 % test |
| Schema conformity (5 entity types, relation signatures) | **validated** — 0 violations, 4 types populated |
| Structural coherence (facies associations, counter-classes, preserved ambiguities) | **validated** |
| Causal-chain coherence (mechanisms as paths) | **validated** — 29 chains, 6 all-Tier-1 |
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
