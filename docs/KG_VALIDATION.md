# KG_VALIDATION.md — How Geological Coherence Is Established

Geological coherence cannot be demonstrated by a single metric. It is established by
**six convergent checks**, five of which are complete; the sixth (triple-level
geological correctness) is in progress via the blinded expert packet. See
`VALIDATION_METRICS.md` for the numeric backing of each check, and
`GEOLOGIST_GUIDE.md` for how to present these to a domain expert.

## 7.1 Reproduction of known facies associations *(done — `fig_object_descriptor_heatmap`)*

The most direct check: the object × descriptor matrix must reproduce associations any
seismic interpreter already knows, **without having been given them**. Verified row by
row: MTD → chaotic, transparent, hummocky, blocky, discontinuous; turbidite → parallel,
continuous, layered; hemipelagite → parallel, continuous; slide → blocky, undeformed.
That the **counter-classes** (turbidite, hemipelagite) receive descriptors opposite to
the MTD's is a strong signal: the graph also encodes what an MTD is *not*.

## 7.2 Recoverability of an independent expert ontology *(done — 82.4 % test recall)*

LB2019 (Le Bouteiller et al. 2019) was built by hand by a geologist, from a related but
distinct corpus, with no contact with this pipeline (grep-verified: LB2019 never enters
pipeline construction, only evaluation). Recovering 82.4 % of its extractable edges on a
frozen test split is an indirect but solid geological validation — it means the pipeline
independently arrives at conclusions a domain expert had already formalized by hand.

## 7.3 Causal-chain coherence *(done — check passed)*

Verified with `analysis_suite/19_causal_chains.py`: the causal subgraph (80 arcs,
104 nodes, sources/sinks derived structurally, no hardcoded domain lexicon) contains
29 multi-arc chains, 6 entirely Tier-1 (after merging one nominalization duplicate).
Distribution: 17 two-arc chains, 13 three-arc chains.

**Reference chain:** `earthquake —triggers→ slope failure —causes→ mass transport
deposit`, minimum confidence 1.0 on every arc — the only chain maximal on all three
channels (tier, evidence, consensus). This is the domain's canonical sequence,
reconstructed purely by composing sentence-level assertions extracted independently.

**Longest chain (4 arcs):** `gas hydrate dissolution → excess pressure → formation
stress gathering → developing MTDs` — a complete and physically sound mechanism.

**Two other defensible chains:** `wave-loading effects → decrease in effective stress →
sediment approaching liquefaction`, and the `water intrusion and fluidisation →
particle segregation → …` family.

**One chain to flag rather than trust:** `earthquake → slope failure → incision of
lateral ramps` — minimum confidence 0.0 (no paper support) and geologically doubtful:
lateral-ramp incision is a basal-surface geometry, not a triggering product. Candidate
for the inspection list, not for the manuscript's positive examples.

**Weak-link rule** (enforced by the script): chain tier = worst arc tier, chain
confidence = minimum of arc confidences. A three-arc Tier-2 chain is weaker than any of
its arcs taken alone — chains do not average out weakness.

**Note on the prediction this check overturned:** the expectation going in was a thin
result ("a few two-arc chains, nothing at three or more"), based on `controls` having
only 4 triples and the known failure of the `pore pressure` case (see
`VALIDATION_METRICS.md`, failure-mode analysis). The actual result contradicts that:
mechanism composition does emerge from assembling sentence-level assertions. The
`pore pressure` case is therefore a documented exception, not the governing pattern.

## 7.4 Preservation of genuine ambiguity *(done — to be foregrounded, not hidden)*

The KG lists both `high-amplitude` and `low-amplitude` for MTDs. A system that "cleaned"
this contradiction away would produce a tidier but less true graph: amplitude genuinely
is non-diagnostic for MTDs, and the literature says so inconsistently because the
phenomenon is inconsistent. Preserving an attested ambiguity is evidence of fidelity, not
a defect to explain away.

## 7.5 Descriptor discriminance *(done — cooccurrence analysis)*

A descriptor attached to a single object class separates that class perfectly; a
descriptor shared across three or more object classes is a poor discriminator. This is
the only centrality measure in the KG portrait that is directly geologically actionable,
and it feeds forward into Part II's descriptor-to-facies matching.

## 7.6 Absence of geologically false statements *(in progress — expert packet)*

The five checks above establish that the graph is **structurally** coherent. Only a
geologist can establish that it is **factually** correct. Three examples from the
statement inspection illustrate why no automatic metric can substitute for this:

- `rugged upper surface overlies stable paleo-lakebed` — the source text actually says
  "rugose **and elevated with respect to**": extraction converted a comparison of
  elevation into a stratigraphic relation.
- `excess pore pressure causes slope failure` — the source passage frames this as a
  *precondition*, with earthquakes as the actual trigger. `causes` overstates the role.
- `amapá megaslide complex is described as stratified` — the source text says the
  **incised layers** are stratified, not the complex as a whole. An attribution error,
  not a fabrication: the words are genuinely in the passage.

All three pass textual verification (the words are present in the source sentence).
Only a domain expert catches the semantic drift. **This is the central argument for why
expert validation is necessary and cannot be automated away** — and the reason the
blinded expert packet (36 items, stratified across tier × confidence, each item carrying
its actual triple) is the critical remaining gate before the KG can be described as
geologically validated rather than merely structurally validated.

## What to say about validation status today

Without hedging: *"Textual traceability and structural coherence are established and
measured; triple-by-triple geological correctness is currently under expert review — here
is the 36-item packet we have submitted."* Claiming an expert validation that has not
happened is the one real risk to this work's credibility; claiming none of the six checks
have been attempted would be equally inaccurate.

## Where these checks live in code

| Check | Script |
|---|---|
| 7.1 Facies heatmap | `analysis_suite/10_descriptor_analysis.py` |
| 7.2 LB2019 recoverability | `pipeline/07_final_metrics.py` |
| 7.3 Causal chains | `analysis_suite/19_causal_chains.py` |
| 7.4 Ambiguity preservation | `analysis_suite/10_descriptor_analysis.py` (manual read of descriptor lists) |
| 7.5 Descriptor discriminance | `analysis_suite/10_descriptor_analysis.py` (cooccurrence matrix) |
| 7.6 Statement-level correctness | `analysis_suite/17_build_expert_packet.py` |