# Pipeline Validation Report
*Generated 2026-08-04T10:21:46 on node10*

| Stage | Status | Detail |
|---|---|---|
| frozen eval protocol | **OK** | dev/test/declaration committed |
| duplicate definitions | **OK** | no shadowed top-level definitions |
| 01 index artifact | **OK** | papers=37, chunks=1955, dup=0 |
| 01 PDF parser | **KNOWN_ISSUE** | memory leak (~1.7GB/3s) in current env; run13 bypassed via chunk-level dedup of run11 index |
| 06 fused KG | **OK** | loads via kg_io, 157 triples |
| 06b provenance KG | **OK** | loads via kg_io, 157 triples |
| M4 panel KG | **OK** | loads via kg_io, 151 triples |
| final enforced KG | **OK** | loads via kg_io, 151 triples |
| 04b schema enforcement | **OK** | 4 types |
| 05a rule-canonicalize (smoke) | **OK** | exit 0 |
| 04c lexicon guard (smoke) | **OK** | exit 0 |
| 07 metrics (full34) | **OK** | exit 0 |
| 08 provenance (analysis) | **OK** | exit 0 |

## Failures / known issues & recommendations
### 01 PDF parser — KNOWN_ISSUE
- Detail: memory leak (~1.7GB/3s) in current env; run13 bypassed via chunk-level dedup of run11 index
- Recommendation: pin/repair the PDF parsing dependency before any corpus extension; document versions with pip freeze
