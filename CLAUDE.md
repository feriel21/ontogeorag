# Claude Code brief — OntoGeoRAG cleanup & documentation pass

## Mission

Style, documentation and consistency pass over `pipeline/`, `m4/` (or `~/m4_verifier` if not yet merged) and `analysis_suite/`. This is a **behavior-preserving** refactor: no numeric output may change. The regression check is:

`python analysis_suite/validate_pipeline.py --quick` (must stay 10/12 OK with the same two KNOWN_ISSUEs) AND

`PYTHONPATH=. python pipeline/07_final_metrics.py --kg output/run13/kg/tiered_kg_run13.json --output /tmp/regress.json`

must reproduce the committed `output/run13/kg/metrics_full34.json` numbers EXCEPT for the two deliberate fixes below.

## Ordered tasks

1. **Unify the descriptor lists (highest priority).** Three divergent lists exist: `pipeline/04_clean_validate.py` (13 terms, has hummocky/massive/stratified), `pipeline/07_final_metrics.py` line ~21 (13 terms, has `deformed`, MISSING `hummocky` — this is a bug), and `pipeline/rag/constants.py` KNOWN_DESCRIPTORS (40 terms). Fix: define in `constants.py` two exports — `KNOWN_DESCRIPTORS` (40, extraction closed-world) and `LB2019_BENCHMARK_DESCRIPTORS` (the true 13-term LB2019 list as used in 04) — and import them everywhere. Delete local copies. Descriptor coverage in 07 will change (intended; note it in the commit message).

2. **Fix 07 label bug**: the report prints "Recall vs LB2019 (n/26)" while counting over the 34-edge benchmark. Label must say n/34 (add a separate original-26 line if cheap).

3. **Docstrings (English)** for every module and public function in `pipeline/` and the m4 scripts, following the analysis_suite style: module header with WHY / WHAT / USAGE, one-line function docstrings stating inputs, outputs, side effects. Do not restructure logic.

4. **Inline comments** only where non-obvious (verification policies, RELATION_MAP rationale, SciBERT merge protection, tier logic).

5. **ruff format + ruff check --fix** across pipeline/, m4 scripts, analysis_suite/ with the provided pyproject.toml (merge with existing setup.cfg tooling config rather than duplicating). No logic edits from lint autofixes beyond imports/order.

6. **Dead code**: flag (do NOT delete without listing them first and asking) apparently-unused branches: hybrid/dense retrieval in 01/02 (`--hybrid`, `build_and_save_dense_index`) if truly unused in all slurm/ scripts; `*.bak.*` files.

7. **README.md refresh**: point to docs/, state the canonical run13 invocation chain and the run13-frozen tag.

## Hard constraints

- Never touch files under `output/`, `configs/lb_*`, or anything the frozen protocol references.
- Do not rename pipeline stage files (01–07 numbering is cited in the manuscript).
- All comments and docstrings in English.
- One commit per task, descriptive messages; run the regression check before each commit.
- Stop and ask before deleting anything.
