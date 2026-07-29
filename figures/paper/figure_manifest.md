# Manuscript figure manifest

| Figure | Generator | Status |
|---|---|---|
| Pipeline overview | `pipeline/plot_pipeline_overview.py` | existing |
| Retrieval comparison | `pipeline/plot_retrieval_comparison.py` | existing (hard-coded values — check against run13) |
| Corpus diagnostic / failure modes | `pipeline/plot_corpus_diagnostic.py` | existing |
| KG subgraph, vignette | `pipeline/plot_kg_subgraph.py`, `plot_vignette_subgraph.py` | existing (hand-curated triples — verify they still exist in run13) |
| KG portrait, descriptors, relations, graph, growth | `analysis_suite/10–13` via `run_full_analysis.sh` | regenerated per run |
| M4 verdicts, blind-vs-evidence, panel agreement, tier flow | `m4/m4_figures.py`, `m4/m4_figures_v2.py` | **regenerate for run13** |
| Contamination ablation | `analysis_suite/18_paper_figures.py` | new |
| Frozen protocol recall | `analysis_suite/18_paper_figures.py` | new |
| Confidence & consensus | `analysis_suite/18_paper_figures.py` | new |
| Evidence anchoring | `analysis_suite/18_paper_figures.py` | new |

## Regenerate the M4 figures for run13

```bash
python m4/m4_figures.py \
    --decisions output/run13/m4_panel/m4_panel_decisions.jsonl \
    --output    figures/paper/m4_run13
python m4/m4_figures.py \
    --decisions output/run13/m4/m4_decisions.jsonl \
    --output    figures/paper/m4_run13_llama
python m4/m4_figures_v2.py \
    --panel  output/run13/m4_panel/m4_panel_report.json \
    --output figures/paper/m4_run13
```

Generated: fig_paper_contamination, fig_paper_frozen_protocol, fig_paper_confidence_anatomy, fig_paper_provenance