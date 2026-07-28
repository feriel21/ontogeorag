# Robustness report (auto-generated)

Corpus: 30 papers resolved from provenance (3 triples without paper_ids, excluded).
Bootstrap draws per size: 200 (seed 42).

## nodes
- Heaps fit: V(n) ≈ 17.9 · n^0.64 — CONVERGING (sub-linear growth — the fixed query set saturates; additional papers densify support rather than expand coverage).
- Extrapolation (EXTRAPOLATED, not observed): n=80: ≈296, n=130: ≈404, n=530: ≈995.

## edges
- Heaps fit: V(n) ≈ 14.5 · n^0.69 — CONVERGING (sub-linear growth — the fixed query set saturates; additional papers densify support rather than expand coverage).
- Extrapolation (EXTRAPOLATED, not observed): n=80: ≈298, n=130: ≈417, n=530: ≈1100.

## descriptors
- Heaps fit: V(n) ≈ 5.8 · n^0.36 — CONVERGING (sub-linear growth — the fixed query set saturates; additional papers densify support rather than expand coverage).
- Extrapolation (EXTRAPOLATED, not observed): n=80: ≈28, n=130: ≈34, n=530: ≈56.

## Stability
- Hub stability (Jaccard, top-10 degree, 60% subsample vs full): **0.68** — hubs depend on corpus composition; interpret hub-based claims cautiously.
- Relation-distribution stability (1 − L1/2): **0.92**.

## Stated limitation
`hasDescriptor` is a closed-world relation: descriptor growth is upper-bounded by the size of the canonical descriptor vocabulary. Truncate the descriptor extrapolation at that bound in the manuscript; only the node/edge extrapolations are meaningful beyond it.
Subsampling measures redundancy inside the current corpus and query design. It cannot anticipate genuinely new terminology from unseen basins; the extrapolations are lower bounds on novelty and must be labeled as such in the manuscript.