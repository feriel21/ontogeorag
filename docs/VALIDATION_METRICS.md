# VALIDATION_METRICS.md — Metrics, Thresholds, and Justification

Canonical run: **run13** (tag `run13-frozen`). Corpus: 37 peer-reviewed papers,
1,955 deduplicated chunks. All numbers below are traced to
`output/run13/kg/tiered_kg_run13_enforced.json` and `output/run13/analysis/`.

## Validation status summary

| Layer | Status | Evidence |
|---|---|---|
| Textual grounding (triple → source sentence) | **validated** | 94.7 % anchored (143/151) |
| Cross-family textual support (Llama + Mistral panel) | **validated** | 84 accept / 67 uncertain / 6 quarantined |
| Recoverability vs. expert ontology (LB2019, frozen split) | **validated** | 82.4 % test recall |
| Schema conformity | **validated** | 0 relation-signature violations, 4/5 entity types populated |
| Structural coherence (facies, counter-classes, ambiguities) | **validated** | see `KG_VALIDATION.md` §7.1, 7.4, 7.5 |
| Causal-chain coherence | **validated** | 29 chains, 6 all-Tier-1 |
| **Geological correctness of individual triples** | **not yet validated** | expert packet sent, pending return |

**Defensible one-line summary:** the KG is a *structured, traceable, textually verified
index of literature assertions* — not yet an expert-validated geological model.

## Headline numbers (run13)

| Metric | Tier-1 | Tier-1+2 |
|---|---|---|
| Triples (fused, pre-panel) | 90 | 157 |
| Triples (post cross-family panel) | 64 | 151 (+6 quarantined) |
| LB2019 recall — **test split** (17 edges) | — | **82.4 % (14/17)** |
| LB2019 recall — dev split (17 edges) | — | 64.7 % (11/17) |
| LB2019 recall — 26-edge original | 34.6 % (9/26) | 73.1 % (19/26) |
| LB2019 recall — 34-edge combined | 44.1 % (15/34) | 73.5 % raw (25/34) · 76.5 % artefact-corrected (26/34) |
| Descriptor coverage (13-term LB2019 set) | 10/13 | 11/13 (11/12 of the *reachable* set) |
| NOT_SUPPORTED rate, self-verification | 0.0 % (Wilson CI [0.0, 3.6] %) | — |
| Evidence anchoring | — | 94.7 % (143/151) |
| Inter-article consensus (≥ 2 papers) | — | 39.1 % |

## Why the dev/test split matters, and how it was frozen

The 34-edge benchmark was split 17/17 (stratified), drawn, hashed (SHA256), and
committed in `configs/run13_protocol_declaration.json` **before** run13 executed. This
directly answers the objection that the KG's "rescue" queries were tuned against the
benchmark: if query design had overfit the benchmark, dev recall would exceed test
recall. The observed gap runs the opposite way (test 82.4 % > dev 64.7 %) — no evidence
of overfitting. **Caveat:** n = 17 per split means one edge = ±5.9 pp; the finding is the
*direction* of the gap, not the precision of either number.

Recall is measured on the **fused, pre-panel** KG for comparability across
configurations — the cross-family panel is a precision mechanism applied afterwards and
can remove benchmark-matching edges (it quarantined one in run13); it is deliberately
excluded from the recall protocol.

## Three known matching-protocol artefacts (reported, not silently corrected)

These are declared rather than fixed, to preserve comparability with earlier runs.

1. **Plural sensitivity.** Matching is normalized-substring based; canonicalization
   (`submarine landslides → submarine landslide`) breaks an otherwise-correct match. The
   edge is present in the KG but counted as missed. Both raw (25/34) and
   plural-corrected (26/34) figures are reported.
2. **Asymmetric synonym handling.** `DESCRIPTOR_SYNONYMS` is applied inside
   `coverage()` but not inside `recall()`. Consequence: an edge like `amapá megaslide
   complex hasDescriptor stratified` does not match `megaslide hasDescriptor layered`
   even though the KG contains the semantically equivalent statement.
3. **Reachable descriptor ceiling is 12, not 13.** `stratified` is mapped onto `layered`
   before counting and can structurally never be reported as "found." True coverage
   against the reachable set is 11/12 = 91.7 %, not 11/13.

## Corpus-contamination ablation (run11 → run13)

run11's index contained `.ipynb_checkpoints` duplicates of every paper (42.3 % of
chunks). run13 rebuilt the index from the same underlying chunks with duplicates
removed — chunking is bit-identical, so deduplication is the only variable. Both runs
are measured with the same, corrected code.

| Measure | run11 (contaminated) | run13 (clean) |
|---|---|---|
| Index | 3,386 chunks (42.3 % duplicated) | 1,955 chunks, 0 duplicated |
| Top-5 retrieval slots occupied by duplicates | 39.8 % | 0 % |
| Effective context diversity | 3.42 / 5 unique passages | 5 / 5 |
| Raw triples per pass | 380 | 446 (+17 %) |
| Recall, pass A (26-edge) | 17/26 | 19/26 |
| Recall, LB2019 original (26) | 18/26 = 69.2 % | 19/26 = 73.1 % |
| Recall, 34-edge | 26/34 = 76.5 % | 25/34 raw · 26/34 corrected |
| — of which the 8 in-house extensions | 8/8 | 6/8 (7/8 corrected) |
| Descriptor coverage, Tier-1 | 11/13 | 10/13 (loses `undeformed`) |
| Fused Tier-1 / Tier-2 | 101 / 52 | 90 / 67 |
| Post-panel Tier-1 / Tier-2 / Quarantine | 74 / 74 / 5 | 64 / 87 / 6 |
| Evidence anchoring | 89.5 % | 94.7 % |
| Consensus ≥ 2 papers | 36.6 % | 39.1 % |

**Reading.** Decontamination improves recovery of the independent expert benchmark
(18 → 19/26) and raw extraction yield (+17 %), at the cost of a narrower Tier-1 core
(101 → 90 fused triples, 11 → 10 descriptors, Tier-1 recall 47.1 % → 44.1 %). Both edges
lost on the 34-edge benchmark belong to the team's own extensions; one is purely the
plural-matching artefact above. Cleaner contexts increase both coverage and cross-pass
divergence simultaneously — this is expected, not a regression.

## Retrieval-quality findings

| Comparison | Result |
|---|---|
| Memory-only baseline (no retrieval, Exp B) | 26.5 % recall (9/34) |
| BM25 only → BM25 + CrossEncoder | 47.1 % → 76.5 % (**+29.4 pp**), 8 reference edges recovered |
| Cross-model agreement (Llama-3.1-8B, Exp D) | 96 % (4/100 disagreements on Tier-1) |

Retrieval quality — not LLM reasoning capacity — is the primary bottleneck within this
sentence-level extraction framework. This is the paper's first pillar result.

## Cross-family verification findings

- **Over-interpretation index = 0.976.** Of 42 triples the verifier judges
  NOT_SUPPORTED from the source text, 41 are judged PLAUSIBLE without seeing the text.
  Empirical justification for the blind/evidence dual-pass design and for the
  "LLM as transducer, not knowledge source" framing.
- **Machine inter-judge agreement sits within the human range.** Llama × Mistral κ
  ranges 0.14–0.30 depending on the variant; only linear-weighted evidence κ (0.30)
  reaches the human inter-expert band (0.30–0.37, Elia+Alain vs. Sara). The agreement
  ceiling reflects task difficulty, not annotator or judge quality.
- **Tier predicts an independent verifier's judgement.** 39 % of Tier-1 triples are
  SUPPORTED by the cross-family verifier vs. 9 % of Tier-2 (49 % of Tier-2 are
  NOT_SUPPORTED). Dual-pass survival — a purely sampling-based criterion — anticipates
  the verdict of a different model family.

## Causal-chain coherence

The causal subgraph (80 arcs, 104 nodes; sources/sinks derived structurally, no domain
lexicon) contains 29 multi-arc chains, 6 entirely Tier-1, after merging one
nominalization duplicate (`formation of bipartite flow` ≡ `bipartite flow`).
Chain-level scoring follows the weak-link rule: chain tier = max(arc tiers), chain
confidence = min(arc confidences). The canonical chain `earthquake —triggers→ slope
failure —causes→ mass transport deposit` has maximal confidence on every arc. One chain
(`earthquake → slope failure → incision of lateral ramps`) has minimum confidence 0.0
and is geologically questionable — flagged for the inspection list rather than silently
dropped. Reproduce with `analysis_suite/19_causal_chains.py`.

## Expert validation (§4.4) — current state

- Round-1 (Elia + Alain, joint): 19 Yes / 10 Partial / 0 No on 29 Tier-1 statements.
- Sara (independent second pass): overlapping verdicts; introduced 2 Partial→No
  divergences (statements 6, 18), attributed to terminological framing rather than
  geological falsity.
- Cohen's κ (Elia+Alain vs. Sara): unweighted 0.30 ("fair"), linear-weighted 0.37.
- **Antoine's independent pass: not yet provided** — critical gate for §4.4
  finalization.
- **Structural limitation discovered:** the 29 §4.4 statements are hand-written, with no
  triple ever attached, and cannot be joined to the KG (three join attempts — by ID, by
  text, by assisted mapping — failed or produced arbitrary matches; an apparent negative
  result, τ = −0.22, turned out to be a mixed-join artefact). **No triple in the KG
  currently carries an expert verdict.**
- **Response:** a new stratified, blinded expert packet
  (`analysis_suite/17_build_expert_packet.py`) samples 36 items across tier × confidence
  strata (6 strata + priority group), each item carrying its actual triple, with tier and
  confidence withheld from the reviewer to keep later score validation non-circular.
  This packet — not the §4.4 statements — is the path to triple-level geological
  validation.

## Reproduce these numbers

```bash
cd ~/ontogeorag && source ~/kg_test/venv/bin/activate
unset SLURM_PROCID SLURM_NTASKS RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export HF_HUB_OFFLINE=1 PYTHONPATH=$PWD

python analysis_suite/validate_pipeline.py --quick
FROM_STAGE=8 N_PAPERS_EXPECTED=37 bash analysis_suite/run13_pipeline.sh
```

Frozen protocol files: `configs/lb_dev.json`, `configs/lb_test.json`,
`configs/run13_protocol_declaration.json`. Tag: `run13-frozen`
(`git checkout run13-frozen` to inspect the exact evaluated state).