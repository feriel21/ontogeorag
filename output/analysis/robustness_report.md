# Robustness report (auto-generated)

Corpus: 28 papers resolved from provenance (7 triples without paper_ids, excluded).
Bootstrap draws per size: 200 (seed 42).

## nodes
- Heaps fit: V(n) ≈ 16.3 · n^0.67 — CONVERGING (sub-linear growth — the fixed query set saturates; additional papers densify support rather than expand coverage).
- Extrapolation (EXTRAPOLATED, not observed): n=78: ≈297, n=128: ≈414, n=528: ≈1063.

## edges
- Heaps fit: V(n) ≈ 14.1 · n^0.71 — slowly growing.
- Extrapolation (EXTRAPOLATED, not observed): n=78: ≈315, n=128: ≈448, n=528: ≈1232.

## descriptors
- Heaps fit: V(n) ≈ 5.0 · n^0.33 — CONVERGING (sub-linear growth — the fixed query set saturates; additional papers densify support rather than expand coverage).
- Extrapolation (EXTRAPOLATED, not observed): n=78: ≈21, n=128: ≈25, n=528: ≈41.

## Stability
- Hub stability (Jaccard, top-10 degree, 60% subsample vs full): **0.63** — hubs depend on corpus composition; interpret hub-based claims cautiously.
- Relation-distribution stability (1 − L1/2): **0.91**.

## Stated limitation
`hasDescriptor` is a closed-world relation: descriptor growth is upper-bounded by the size of the canonical descriptor vocabulary. Truncate the descriptor extrapolation at that bound in the manuscript; only the node/edge extrapolations are meaningful beyond it.
Subsampling measures redundancy inside the current corpus and query design. It cannot anticipate genuinely new terminology from unseen basins; the extrapolations are lower bounds on novelty and must be labeled as such in the manuscript.