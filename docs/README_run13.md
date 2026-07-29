# run13 post-freeze cleanup: descriptor-coverage change

This note documents the one intentional numeric change made to `output/run13/kg/metrics_full34.json`-derived
output during the post-run13 Claude Code cleanup pass (see `CLAUDE.md` for the full brief), so the
change is traceable independently of the commit history.

## What changed and why

`pipeline/07_final_metrics.py` used to define its own local 13-term `LB2019_DESCRIPTORS` set with a bug:
it contained `"deformed"` instead of `"hummocky"`. `pipeline/04_clean_validate.py` had the *correct*
13-term list (with `"hummocky"`) in a separate local copy. The two were never compared against each
other, so the bug went unnoticed.

The fix: both files now import a single `LB2019_BENCHMARK_DESCRIPTORS` constant from
`pipeline/rag/constants.py` (see commit "pipeline: unify descriptor lists into pipeline/rag/constants.py").
`07_final_metrics.py`'s descriptor coverage is therefore now computed against the correct 13-term list.

## Coverage-change decomposition

Re-running `PYTHONPATH=. python pipeline/07_final_metrics.py --kg output/run13/kg/tiered_kg_run13.json`
against the frozen `output/run13/kg/metrics_full34.json` baseline, diffed field by field:

| Field | Before (buggy) | After (fixed) | Change |
|---|---|---|---|
| `descriptor_coverage.tier1.n_found` | 9 | 10 | +1 |
| `descriptor_coverage.tier1.coverage` | 69.2% | 76.9% | +7.7 pp |
| `descriptor_coverage.tier12.n_found` | 10 | 11 | +1 |
| `descriptor_coverage.tier12.coverage` | 76.9% | 84.6% | +7.7 pp |

**Cause of the +1**: `"hummocky"` is now correctly counted as a found descriptor. It was previously
invisible to `coverage()` because the buggy list checked for `"deformed"` (not an LB2019 descriptor at
all) in its place. This is the entire delta — no other descriptor's found/missing status changed.

**Everything else is unchanged**: `recall_vs_lb2019` (the 34-edge recall block), `hallucination`,
`summary`, `relation_distribution_by_tier`, and `triples_final` are byte-for-byte identical to the
frozen baseline. Recall matching (`recall()` in `07_final_metrics.py`) never depended on the descriptor
list — it matches directly against `configs/lb_reference_edges.json` / `configs/lb_reference_edges_original26.json`
— so it was never affected by this bug.

The only *addition* (not a change to an existing number) is a new `recall_vs_lb2019_orig26` block,
added in the same pass to fix a separate label bug — see below.

## Coverage is counted post-synonym-mapping

A detail worth knowing before reading the "Descriptors found" / "Missing" lists in `07_final_metrics.py`'s
output: `coverage()` checks descriptors *after* `norm_desc()` applies `DESCRIPTOR_SYNONYMS`, not the raw
extracted text. `DESCRIPTOR_SYNONYMS` maps `"stratified"` → `"layered"` (they are treated as the same
seismic-facies concept). This means:

- An extracted triple whose object is literally `"stratified"` is counted as coverage of `"layered"`,
  not `"stratified"`.
- `"stratified"` therefore shows up in `coverage()`'s `missing` list even on a KG that does contain the
  word "stratified" somewhere in a `hasDescriptor` object — that's expected, not a bug. The 13-term
  LB2019 benchmark still nominally includes `"stratified"` as its own category, but the synonym mapping
  means it can never independently appear as "found" under the current normalization; any hit on
  `"stratified"` is absorbed into `"layered"`'s count instead.

If a future accuracy fix wants `"stratified"` and `"layered"` reported separately, `coverage()` would
need to check pre-synonym text, which would be a genuine behavior change (out of scope for this
behavior-preserving cleanup pass).

## Separately: the n/26 → n/34 label fix

Unrelated to descriptor coverage, `07_final_metrics.py` also mislabeled its LB2019 recall line as
`"(n/26)"` while actually counting hits against the 34-edge `configs/lb_reference_edges.json` benchmark.
The label now reads `"(n/34)"` (computed dynamically from `total_reference`, not hardcoded), and a
separate `"Recall vs LB2019-orig26 (n/26)"` line was added — computed against
`configs/lb_reference_edges_original26.json` — for readers who want the original 26-edge figure
specifically. See commit "pipeline/07: fix n/26 label bug, add original-26 recall line".
