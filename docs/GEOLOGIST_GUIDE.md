# GEOLOGIST_GUIDE.md — How to Read and Present This KG (No ML Background Needed)

This guide is for presenting OntoGeoRAG's knowledge graph to geologists — reviewers,
collaborators, or domain experts asked to validate it. It assumes no familiarity with
retrieval, embeddings, or LLMs.

## Principle: enter through the geology, not the metric

A geologist evaluates an interpretation system on what it says about the rock and the
seismic signature, not on its recall score. Never open a conversation with recall, κ, or
F1 — those come last, and only as sorting tools. Recommended order:

1. **The facies table** (`figures/paper/... fig_object_descriptor_heatmap`) — a familiar
   object. *"Here is what 37 papers say about the seismic signatures of five geological
   objects. Does this match what you already know?"* This opens the discussion on
   substance immediately, and lets the reviewer's own expertise do the first check.
2. **A causal chain**, drawn from the graph: earthquake → slope failure → MTD, with
   pore-water overpressure as a contributing factor. *"The system reconstructed this
   chain without being told to look for one — no query asked for a chain."*
3. **A triple with its evidence**: the assertion, the exact source sentence, the paper.
   This is what distinguishes the work from a chatbot, and the moment the audience
   realizes every claim in the graph can be audited individually.
4. **An admitted error.** Show `amapá megaslide complex is described as stratified` and
   explain why it is wrong (see `KG_VALIDATION.md` §7.6). Surfacing an error before
   someone else finds it builds more credibility than any number could.
5. **Only then**: tiers, confidence, recall — framed as *sorting tools* ("where to start
   if you only have an hour"), never as a performance claim.

## Vocabulary — what to say, what to avoid

| Avoid | Say instead |
|---|---|
| "the model learned" | "the system extracted this from sentence X in paper Y" |
| "82 % precision" | "it recovers 14 of the 17 relations an expert had already formalized" |
| "embeddings, cosine, tokens" | "it groups terms that refer to the same thing" |
| "Tier-1" (unexplained) | "extracted independently twice **and** confirmed by a second, different system" |
| "the KG detects MTDs" | "the KG describes MTDs; it does not detect them" (see below) |

## The sentence to say every time, before anyone asks

> The graph says "an MTD is often chaotic." It never says "whatever is chaotic is an
> MTD." A turbidite channel, a gas chimney, or plain noise can all look chaotic too.
> Downstream, this produces a **candidate mask**, never a detector.

This is the epistemological guarantee of the entire project. Saying it before the
question is asked prevents the misunderstanding that would otherwise cause the whole
approach to be dismissed.

## Answering the three expected objections

**"Your graph only has 151 relations — that's very small."**
The right comparison is not Wikidata but LB2019: 173 edges, built by hand by a team, from
41 papers. Here: ~150 automatically extracted edges from 37 papers, each with provenance,
recovering ~82 % of the extractable portion of that hand-built graph. This is a measured
**human–machine equivalence**, not a small graph.

**"An LLM hallucinates. How do you know this isn't fabricated?"**
Three answers, in this order: (1) every arc traces back to a specific sentence in a
specific paper — check one; (2) two models from different architecture families
re-read the passage independently and can quarantine a triple; (3) the scale of the
problem has been **measured directly** — of 42 assertions the text does not support, the
model judged 41 of them "plausible" from memory alone. That number is exactly why the
pipeline never trusts a model's memory over the retrieved text.

**"Does this replace an interpreter?"**
No, and that is not the goal. It replaces the days spent re-reading 37 papers to find who
said what about MTD facies. Interpretation itself remains human.

## What to say about validation, today, without overstating

State it plainly: *"Textual traceability and structural coherence are established and
measured; triple-by-triple geological correctness is currently under expert review — here
is the 36-item packet we have submitted."* Claiming an expert validation pass that has
not happened yet is the single largest credibility risk in presenting this work — larger
than any weakness the honest numbers themselves reveal.

## Reading the KG file directly

`output/run13/kg/tiered_kg_run13_enforced.json` is a JSON object with keys
`{meta, tier1, tier2, quarantine}`. Each triple carries:

- `subject`, `relation`, `object` — the assertion itself
- `tier` — 1 (survived both extraction passes and cross-family review) or 2 (survived
  one pass)
- a **confidence score** in [0, 1] — composite of tier weight, cross-family panel
  decision, and inter-article consensus (not the degenerate Qwen self-verification score)
- **evidence** — the source sentence and paper, when anchored (94.7 % of triples are)

A triple in `quarantine` was flagged by the cross-family panel and excluded from the
main graph, but kept (not deleted) so its rejection is itself auditable.

## Where to point people for more

- Facies table and full descriptor analysis: `KG_VALIDATION.md` §7.1
- Numeric backing for every claim in this guide: `VALIDATION_METRICS.md`
- Full stage-by-stage pipeline: `PIPELINE.md`