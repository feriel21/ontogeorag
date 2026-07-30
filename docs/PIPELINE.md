# PIPELINE.md — OntoGeoRAG Stage-by-Stage Process

This document describes each stage of the canonical run13 pipeline, its inputs, outputs,
and the rationale behind its position in the chain. For results and headline numbers, see
`VALIDATION_METRICS.md`. For how to read the resulting KG without an ML background, see
`GEOLOGIST_GUIDE.md`. For the six convergent checks behind "geological coherence," see
`KG_VALIDATION.md`.

## Design principles (apply to every stage)

1. **Additive, never destructive.** No script in the core chain (01–07) was modified to
   produce run13. All corrections are inserted as new stages between existing ones, or
   applied to the source corpus. This preserves run11/run13 comparability and never
   breaks what produced already-published numbers.
2. **Verify from disk, never invent a number.** `14_generate_report.py` refuses to write
   any sentence whose source figure cannot be located on disk.
3. **Freeze the protocol before measuring.** The dev/test split was drawn, declared,
   timestamped, and SHA256-hashed **before** run13 executed (`configs/run13_protocol_declaration.json`).
4. **Expose uncertainty rather than erase it.** Quarantine instead of deletion; score
   components exported alongside the scalar; hallucination reported as "verifier
   consensus" with Wilson confidence intervals — never as "absence of error."
5. **Defense in depth.** Four independent layers (Qwen self-check → Llama → Mistral panel
   → human experts): what one layer misses, another catches.

## Stage-by-stage

```
PDF corpus (37 papers)
      │
01_build_index.py        →  chunks.jsonl (1,955 deduplicated chunks; BM25 rebuilt at
      │                     query time, k1=1.5, b=0.75 — no serialized index object)
      ▼
02_extract_triples.py    →  249 ontology-guided queries × (BM25 top-20 → CrossEncoder
      │                     → top-5) → Qwen2.5-7B-Instruct
      │                     Pass A (temperature 0.0) + Pass B (temperature 0.3)
      ▼
03_verify_triples.py     →  STRONG / WEAK / NOT_SUPPORTED (same model as extraction —
      │                     degenerate on survivors by design; the independent check
      │                     is the M4 cross-family battery, below)
      ▼
04_clean_validate.py     →  relation normalization (RELATION_MAP), type constraints,
      │                     closed-world descriptor validation, SciBERT canonicalization
      │                     (cosine < 0.06, LB-vocabulary merge protection), dedup
      │                     → canonical_triples_v5.jsonl
      ▼
05a_rule_canonicalize.py →  deterministic variant merge (plural/hyphen/abbreviation)  [additive]
04c_lexicon_enforce.py   →  closed-world descriptor guard, imports the canonical
      │                     40-term lexicon from constants.py                        [additive]
      ▼
06_tiered_fusion.py      →  Tier-1 (survives both passes) + Tier-2 (survives one pass)
06b_attach_provenance.py →  re-attaches retrieval provenance dropped by fusion         [additive]
      ▼
07_final_metrics.py      →  recall (dev / test / 34-edge / original-26), descriptor
      │                     coverage, hallucination rate
      ▼
m4/ battery               →  Llama-3.1-8B verify (blind pass + evidence pass) →
      │                      Mistral-7B second judge → two-judge conservative panel
      │                      vote → tier reassignment
      ▼
04b_schema_enforce.py     →  entity types mapped onto the 5-type schema, self-loop
      │                      removal, re-dedup, relation-signature check              [additive]
      ▼
analysis_suite/08–19      →  provenance reconstruction, vocabulary audit, descriptor
                             analysis, relation analysis, graph portrait, robustness
                             (bootstrap + Heaps), report generation, expert packet,
                             manuscript figures, causal-chain extraction
```

**Note:** `pipeline/05_canonicalize.py` exists in the repository but is **not** part of
the canonical chain. SciBERT canonicalization already runs inside `04_clean_validate.py`,
which emits `canonical_triples_v5.jsonl` directly.

## Stage details

### 01_build_index.py
Builds `chunks.jsonl` from the PDF corpus. **Known issue:** the PDF parser leaks memory
in the current environment (~1.7 GB / 3 s regardless of file). For run13, the index was
instead rebuilt from run11's already-extracted chunks with `.ipynb_checkpoints` duplicates
removed (`analysis_suite/run13_build_index_from_run11.py`). This makes chunking
bit-identical to run11 by construction — the property that makes the contamination
ablation (see `VALIDATION_METRICS.md`) a controlled comparison rather than a confound.
Repairing the parser is required before extending the corpus beyond 37 papers.

### 02_extract_triples.py
249 ontology-guided descriptor/causal/context/profile queries retrieve BM25 top-20,
rerank to top-5 with `cross-encoder/ms-marco-MiniLM-L-6-v2`, then extract triples with
Qwen2.5-7B-Instruct. Two passes at different temperatures (0.0 / 0.3) give the two
independent extraction attempts that Tier-1 status depends on later.

### 03_verify_triples.py
Same model re-checks its own extractions (STRONG/WEAK/NOT_SUPPORTED). This is
**degenerate by design on survivors** — Qwen essentially always approves what it just
extracted — which is precisely why the independent cross-family M4 battery exists
downstream. Do not read `03`'s verdicts as a genuine reliability signal.

### 04_clean_validate.py
Normalizes relations against `RELATION_MAP`, enforces entity/relation type constraints,
applies closed-world descriptor validation, runs SciBERT canonicalization
(cosine similarity < 0.06, with a merge-protection rule for LB2019 vocabulary), and
deduplicates. Emits `canonical_triples_v5.jsonl`, the input to fusion.

### 05a_rule_canonicalize.py / 04c_lexicon_enforce.py *(additive)*
Deterministic variant merging (plural, hyphenation, abbreviations) and a closed-world
descriptor guard sourced from the single `constants.py` lexicon
(`KNOWN_DESCRIPTORS`, 40 terms). Added without touching `04_clean_validate.py`.

### 06_tiered_fusion.py / 06b_attach_provenance.py
Tier-1 = triple survives both extraction passes; Tier-2 = survives one. Fusion drops
retrieval provenance, which `06b` re-attaches as an additive post-step.

### 07_final_metrics.py
Computes recall against the LB2019-derived benchmarks (dev, test, 34-edge, original
26-edge), descriptor coverage, and hallucination rate. Recall is measured on the
**fused, pre-panel** KG to stay comparable across configurations; the M4 panel is a
downstream precision mechanism, not part of the recall protocol.

### m4/ — cross-family verification battery
Independent of the extraction model family entirely: Llama-3.1-8B runs a blind pass
(judge the triple with no source text) and an evidence pass (judge it against the
retrieved passage). Mistral-7B repeats both as a second judge. A two-judge conservative
panel vote reassigns tiers and quarantines triples both judges reject. This is the layer
that catches what Qwen's self-verification structurally cannot.

### 04b_schema_enforce.py *(additive)*
Maps the 25 LLM-invented entity type labels onto the declared 5-type ontology schema,
removes self-loops, re-deduplicates after remapping, and flags relation-argument
signature violations. Run as a standalone post-process — zero modifications to any
upstream file — with `--dry-run`, `--reject-long`, `--reject-bad-sig` flags for staged
adoption.

### analysis_suite/08–19
Provenance reconstruction (3-channel), vocabulary fragmentation audit (3 detectors),
descriptor/relation statistics, full graph portrait (centrality, communities), robustness
(paper-level bootstrap, Heaps'-law fits), report generation (with anti-fabrication
guard), confidence-score validation, statement-to-triple mapping, the stratified blinded
expert packet, manuscript figures, and causal-chain path extraction.

## Regression check before touching `pipeline/`

```bash
python analysis_suite/validate_pipeline.py --quick     # CPU, ~2 min, 12/13 expected
python analysis_suite/validate_pipeline.py --gpu       # adds smoke tests
```

This harness checks artefact integrity, schema conformity, and — since a branch merge
once silently duplicated the function computing the headline recall metric — that no
top-level definition is shadowed anywhere in `pipeline/`, `m4/`, or `analysis_suite/`.
The two AST-identical copies of `recall()` were logically equivalent (no number was ever
wrong), but nothing would have caught a divergent duplicate. **Never merge after a global
reformat (e.g. `ruff format`) without running this check.**

## Reusing OntoGeoRAG for a new domain

The pipeline is domain-agnostic. Four files are MTD-specific and can be swapped:

| File | Role |
|---|---|
| `configs/ontology_schema.json` | relation and entity types |
| `configs/descriptor_queries.jsonl` | what the pipeline searches for (249 queries) |
| `configs/lb_reference_edges.json` | evaluation benchmark (optional — recall is skipped if absent) |
| `pipeline/rag/constants.py` | `KNOWN_DESCRIPTORS` and `LB2019_BENCHMARK_DESCRIPTORS` |

Query files need one JSON object per line with `query`, `strategy`
(`descriptor` / `causal` / `context` / `profile`), and `focus`.