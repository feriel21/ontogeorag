# The knowledge graph, catalogued

*Everything the graph contains, in plain terms: every concept, which names mean the same thing, every relation and what it is worth, and what would change with more papers.*

---

## 1. What is in the graph

- **159 concepts** (nodes)
- **151 statements** (arcs), plus 0 set aside in quarantine
- **8 kinds of relation** actually used

Concepts by kind:
- **Process** — 61
- **Geological_Object** — 40
- **Environmental_Control** — 39
- **Descriptor** — 19

Every concept is listed in `nodes_inventory.csv` with the sentence it came from and the paper that sentence is in.

---

## 2. Names that may mean the same thing

13 pairs of names were flagged as possibly denoting the same concept. **They have not been merged**: deciding whether two names are the same thing is a geological judgement, not a text-processing one. Please mark each pair in `synonym_candidates.csv` (column `DECISION_same_concept_YN`).

| Name A | Name B | Why flagged |
|---|---|---|
| `bipartite flow` | `formation of bipartite flow` | nominalization: 'formation of X' vs 'X' |
| `development of excess pore-water pressures` | `excess pore-water pressures` | nominalization: 'development of X' vs 'X' |
| `excess pore pressure` | `excess pore-water pressures` | similar spelling (ratio 0.85) |
| `excess pore pressure` | `excess pressure` | similar spelling (ratio 0.86) |
| `excess pore pressure` | `high excess pore pressure` | similar spelling (ratio 0.89) |
| `fluid overpressure` | `pore-fluid overpressure` | similar spelling (ratio 0.88) |
| `gas hydrate disassociation` | `gas hydrate dissolution` | similar spelling (ratio 0.86) |
| `gas hydrate disassociation` | `gas-hydrate dissociation` | similar spelling (ratio 0.92) |
| `gas hydrate dissolution` | `gas-hydrate dissociation` | similar spelling (ratio 0.85) |
| `instabilities` | `instability` | similar spelling (ratio 0.83) |
| `rapid rate of sedimentation` | `rapid sedimentation` | similar spelling (ratio 0.83) |
| `rapid sediment accumulation` | `rapid sedimentation` | similar spelling (ratio 0.83) |
| `translational domain` | `translational sliding` | similar spelling (ratio 0.83) |

### Pairs that look alike but are OPPOSITES

These were caught by the same spelling test and **must not be merged** — they differ by a negation prefix, which reverses the meaning. They are listed so that nobody merges them by mistake, and because their co-existence in the graph is expected and correct.

| Name A | Name B |
|---|---|
| `bore-hole instability` | `bore-hole stability` |
| `continuous` | `discontinuous` |
| `disequilibrium conditions` | `equilibrium conditions` |
| `slope instability` | `slope stability` |


### These name variants are not scattered at random

The flagged pairs cluster into a few families:

- **pore pressure / overpressure** — 7 spellings: `development of excess pore-water pressures`, `excess pore pressure`, `excess pore-water pressures`, `excess pressure`, `fluid overpressure`, `high excess pore pressure`, `pore-fluid overpressure`
- **gas hydrate dissociation** — 3 spellings: `gas hydrate disassociation`, `gas hydrate dissolution`, `gas-hydrate dissociation`
- **sedimentation rate** — 3 spellings: `rapid rate of sedimentation`, `rapid sediment accumulation`, `rapid sedimentation`

These are the principal triggering mechanisms discussed in the literature. The graph fragments precisely where knowledge is densest, because the concepts most authors write about are the ones written in the most different ways. Fragmentation here is a sign of how much a concept is discussed, not of carelessness.


**100 of the 159 concepts appear in only one statement.** The graph is a dense core surrounded by many single mentions — normal for a literature graph, but worth knowing before reading any network statistic.


Note on one case worth an explicit decision: `formation of X` versus `X`. These may be the same thing, or the *process* may be genuinely distinct from the *state*. That is your call, not ours.

---

## 3. The relations, and what each is worth

Importance is deliberately shown as **four separate numbers**, because they disagree — and where they disagree is where the interesting cases are.

| Relation | Reads as | Statements | Papers | % Tier-1 | % accepted by two independent verifiers |
|---|---|---|---|---|---|
| `causes` | produces | 49 | 19 | 49 % | 60 % |
| `hasDescriptor` | is described in seismic data as | 46 | 27 | 39 % | 50 % |
| `affects` | influences | 15 | 17 | 47 % | 53 % |
| `occursIn` | is found in | 15 | 23 | 40 % | 53 % |
| `triggers` | initiates | 14 | 12 | 50 % | 85 % |
| `partOf` | is a component of | 6 | 4 | 17 % | 17 % |
| `overlies` | lies above | 3 | 2 | 33 % | 67 % |
| `controls` | modulates | 3 | 2 | 0 % | 0 % |

**How to read these columns.**
- *Statements* — how often the relation is used. Frequent does not mean reliable.
- *Papers* — in how many different articles. Breadth of support.
- *% Tier-1* — share extracted consistently under two independent sampling conditions.
- *% accepted* — share that two independent language models, from different families, confirmed against the source passage.


**A finding that concerns you directly.** `controls` — the relation carrying the *conditions* under which failure happens — is the weakest in the graph: 3 statements, none extracted consistently, none confirmed by the independent verifiers. Conditions are typically stated across several sentences ('rapid loading raises pore pressure, which in turn reduces effective stress, so the slope fails'), and sentence-level extraction cannot compose them into one subject–verb–object statement. **The layer most useful to an interpreter is the one this method captures worst**, and we report it rather than hide it.

**Conversely, `triggers` is the most reliable** (85 % confirmed). Triggers tend to be stated directly in the literature ('earthquakes trigger slope failures'), which is exactly the sentence shape this method handles best.
**Thinly populated relations** (controls, overlies) rest on very few statements; treat any conclusion drawn from them as provisional.

---

## 4. Would more papers change this?

Measured by repeatedly rebuilding the graph from random subsets of the corpus and fitting the growth curve.

- **nodes**: growth exponent 0.64 → still growing — more papers would still add new ones
- **edges**: growth exponent 0.69 → still growing — more papers would still add new ones
- **descriptors**: growth exponent 0.36 → close to saturation — more papers would mostly add support, not new items

**In plain terms.**
- **New concepts and new statements: yes, they would keep appearing.** The graph is not saturated at 37 papers.
- **New seismic descriptors: essentially no.** The descriptor vocabulary is a closed list of 40 agreed terms, and it has plateaued — adding papers adds *support* for the descriptors already there, not new ones.
- **Would the main conclusions change?** The core concepts are moderately stable to corpus composition (hub stability ≈ 0.68 measured by resampling). Statements resting on a single paper are the ones most likely to move; that is why each statement carries its paper count.
- **One caveat we cannot measure**: resampling tells us about redundancy *inside this corpus*. It cannot anticipate genuinely new terminology from a different basin or a different school of thought.

---

## 5. What we are asking you to check

1. **The concept list** (`nodes_inventory.csv`) — are any of these not real geological concepts? Any obviously missing?
2. **The name pairs** (`synonym_candidates.csv`) — same thing or different things? Your `DECISION` column drives whether we merge them.
3. **The relation glossary above** — does `controls` mean to you what we say it means? Is the distinction between `causes` and `triggers` one you would make?
4. **The statements themselves** — handled separately, in the 36-item review packet.

Every statement in this catalogue can be traced to a sentence in a named article; ask for any of them and we will show you the passage.
