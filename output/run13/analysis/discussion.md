# Discussion (AUTO-DRAFT — scaffolding to be rewritten by the author; every claim cites its source file)

## Major concepts
The graph is organized around **mass transport deposit** (degree 33, 17 papers), **slope failure** (degree 9, 12 papers), **submarine landslide** (degree 7, 21 papers), **turbidite** (degree 6, 13 papers), **slide** (degree 5, 20 papers). [node_statistics.csv] Degree here measures assertion coverage, not geological importance; the topology partly reflects the object-centered query design.

## Processes and mechanisms
Causal assertions present in the graph: *slope failure* —causes→ *mass transport deposit* (T1, 6 papers); *earthquake* —triggers→ *slope failure* (T1, 6 papers); *excess pore pressure* —causes→ *slope failure* (T2, 5 papers); *rise of hydrostatic pressure* —causes→ *instabilities* (T1, 3 papers); *seismic loading* —triggers→ *slope failure* (T2, 3 papers); *gas hydrates destabilization* —causes→ *mass movement initiation* (T1, 2 papers); *higher sedimentation rates* —causes→ *mass movement initiation* (T1, 2 papers); *rapid sediment accumulation* —triggers→ *failure* (T1, 2 papers). [provenance_report.csv] Multi-step mechanisms (e.g. pore-pressure chains) are known to be under-represented by sentence-level extraction; absence here is a formalism limit, not evidence of geological absence.

## Geological controls
3 `controls` assertions. The controls layer is the thinnest of the causal family and should be flagged as under-populated relative to the literature. [relation_statistics.csv]

## Environments
Depositional settings asserted: **continental slope** (2), **mediterranean continental margin** (2), **passive margin** (2), **continental margin** (1), **tectonically active margin** (1), **upper slope** (1), **amazon fan** (1), **deep-water basinal settings** (1), **ebro margin** (1), **basin floor** (1), **deltaic wedge** (1), **espírito santo basin** (1). [provenance_report.csv] Granularity is coarse (margin/basin scale); basin-specific provenance is available per triple via paper_ids and should be exposed if multi-basin corpora are added.

## MTD seismic descriptors
Descriptors attached to the MTD node, by paper support: **undeformed** (14p, T2), **continuous** (13p, T1), **transparent** (12p, T1), **discontinuous** (12p, T1), **chaotic** (10p, T1), **high-amplitude** (9p, T1), **hummocky** (7p, T1), **blocky** (6p, T1), **low-amplitude** (4p, T1), **layered** (3p, T2), **faulted** (3p, T1), **folded** (2p, T1), **stratified** (2p, T2), **wedge-shaped** (2p, T1), **variable-amplitude** (1p, T2). [descriptor_statistics.csv] Where interpreter-canonical descriptors (e.g. chaotic) appear only at Tier-2 while others reach Tier-1, this mismatch between textual and perceptual salience is a finding to report, not an error to fix.

## Under-documented concepts
100 concepts are supported by ≤1 paper [vocabulary_report.csv] — single-source assertions; the confidence score already down-weights them (w_consensus).

## Pivot concepts
Highest-betweenness nodes: **mass transport deposit**, **slope failure**, **continental slope**. [node_statistics.csv] On this small, star-shaped graph betweenness mostly restates the query design; pivots should only be discussed when they sit on causal chains (cross-check relation paths).