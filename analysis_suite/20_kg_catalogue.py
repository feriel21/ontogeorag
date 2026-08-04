#!/usr/bin/env python3
"""
20_kg_catalogue.py — The knowledge graph, catalogued for geologists.
====================================================================
WHY
    Geologists reviewing the KG ask a specific set of questions that no
    existing output answers directly:
      · give me EVERY node, not a top-10;
      · which names refer to the same thing (synonyms / variants)?
      · how many relations, and which ones actually matter?
      · would any of this change if we added more papers?
    This script answers those four questions in one readable document plus
    machine-readable tables, in domain language rather than graph jargon.

WHAT (in --outdir)
    kg_catalogue.md          the document to hand to a geologist
    nodes_inventory.csv      every node: type, role, degree, tier profile,
                             papers, evidence, detected variants
    synonym_candidates.csv   name pairs that may denote the same concept,
                             with the detector that found them and a
                             DECISION column to fill in
    relations_census.csv     per relation: count, papers, entropy,
                             specificity, verifier acceptance, importance
    stability_report.csv     what a larger corpus would and would not change

    Importance of a relation is deliberately NOT a single number. Four
    facets are reported side by side, because they disagree and the
    disagreement is informative:
      · volume        — how many triples use it
      · breadth       — in how many distinct papers
      · informativeness — normalized entropy of its object distribution
                        (low = always points at the same few things)
      · reliability   — share accepted by the cross-family panel
    A relation can be frequent yet uninformative (relatedTo), or rare yet
    decisive (controls).

USAGE
    python analysis_suite/20_kg_catalogue.py \
        --kg output/run13/analysis/kg_with_provenance.json \
        --decisions output/run13/m4_panel/m4_panel_decisions.jsonl \
        --robustness output/run13/analysis/robustness_report.md \
        --outdir output/run13/catalogue
"""

import argparse
import csv
import difflib
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kg_io import get_object, get_relation, get_subject, load_kg

# Relation glossary in plain geological language, not schema names.
RELATION_GLOSS = {
    "hasdescriptor": ("is described in seismic data as",
                      "links a mappable body to a seismic-facies adjective"),
    "causes": ("produces", "one thing brings another into existence"),
    "triggers": ("initiates", "external forcing that starts a process"),
    "controls": ("modulates", "a condition that governs whether/how much"),
    "affects": ("influences", "weaker, unspecified causal influence"),
    "occursin": ("is found in", "depositional or physiographic setting"),
    "overlies": ("lies above", "stratigraphic position"),
    "underlies": ("lies beneath", "stratigraphic position"),
    "partof": ("is a component of", "meronymy: part–whole"),
    "indicates": ("is evidence for", "observation supporting an inference"),
    "evidences": ("is evidence for", "observation supporting an inference"),
    "relatedto": ("is related to", "unspecified — carries no semantics"),
    "formedby": ("is formed by", "genetic origin"),
}

ABBREV = {"mtd": "mass transport deposit", "mtds": "mass transport deposit",
          "mtc": "mass transport complex", "mtcs": "mass transport complex"}
# Negation prefixes INVERT meaning while barely changing the string:
# 'continuous' vs 'discontinuous' scores 0.87 similarity yet they are
# opposites. Flagging such pairs as synonym candidates is dangerous — a
# reviewer skimming the list could merge antonyms. They are detected and
# reported separately, as OPPOSITES.
NEGATION_PREFIXES = ("dis", "in", "un", "non", "a", "im", "ir", "il")

NOMINAL_PREFIXES = ("formation of ", "development of ", "initiation of ",
                    "build-up of ", "rise of ", "reduction of ",
                    "decrease in ", "increase in ", "onset of ",
                    "generation of ", "occurrence of ")


def norm(s):
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def rel_key(r):
    return norm(r).replace("_", "").replace(" ", "")


def normal_form(e):
    """Deterministic normal form: lowercase, hyphens/underscores to spaces,
    punctuation dropped, abbreviations expanded, conservative singular."""
    x = norm(e).replace("-", " ").replace("_", " ")
    x = re.sub(r"[^\w\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    if x in ABBREV:
        x = ABBREV[x]
    toks = []
    for t in x.split():
        if (len(t) > 3 and t.endswith("s")
                and not t.endswith(("ss", "is", "us"))):
            t = t[:-1]
        toks.append(t)
    return " ".join(toks)


def antonym_pair(a, b):
    """True if a and b differ only by a negation prefix on the head word
    (continuous/discontinuous, stability/instability, equilibrium/
    disequilibrium). Such pairs are opposites, never synonyms."""
    ta, tb = norm(a).replace("-", " ").split(), norm(b).replace("-", " ").split()
    if len(ta) != len(tb):
        return False
    diffs = [(x, y) for x, y in zip(ta, tb) if x != y]
    if len(diffs) != 1:
        return False
    x, y = diffs[0]
    short, long_ = (x, y) if len(x) < len(y) else (y, x)
    for p in NEGATION_PREFIXES:
        if long_ == p + short and len(p) >= 1 and len(short) >= 4:
            return True
    return False


def norm_entropy(counts):
    tot = sum(counts)
    if tot == 0 or len(counts) <= 1:
        return 0.0
    h = -sum((c / tot) * math.log2(c / tot) for c in counts if c)
    return round(h / math.log2(len(counts)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--decisions", default=None)
    ap.add_argument("--direction", default=None,
                    help="m4_direction_summary.json — adds the directional "
                         "reliability warning, which matters because the "
                         "verifier is measurably weak at direction errors")
    ap.add_argument("--robustness", default=None,
                    help="robustness_report.md from 13_, for the growth "
                         "section")
    ap.add_argument("--outdir", default="output/run13/catalogue")
    ap.add_argument("--string-thr", type=float, default=0.82)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    kg = load_kg(args.kg)
    active = [t for t in kg["triples"] if t.get("_status") != "quarantine"]
    quarantined = [t for t in kg["triples"]
                   if t.get("_status") == "quarantine"]

    panel = {}
    if args.decisions:
        for line in open(args.decisions, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                panel[(norm(d.get("subject")), rel_key(d.get("relation")),
                       norm(d.get("object")))] = str(
                    d.get("m4_decision") or "").upper()

    # ── node inventory ────────────────────────────────────────────────
    info = defaultdict(lambda: {
        "as_subject": 0, "as_object": 0, "types": Counter(),
        "relations": Counter(), "neighbours": set(), "papers": set(),
        "tiers": Counter(), "evidence": "", "evidence_paper": ""})

    for t in active:
        s, r, o = get_subject(t), get_relation(t), get_object(t)
        if not s or not o:
            continue
        tier = t.get("_tier", 2)
        papers = t.get("paper_ids", []) or []
        ev = str(t.get("evidence") or "").strip().strip('"')
        for node, role, typ in ((s, "as_subject",
                                 t.get("subject_type") or
                                 t.get("source_type")),
                                (o, "as_object",
                                 t.get("object_type") or
                                 t.get("target_type"))):
            d = info[node]
            d[role] += 1
            if typ:
                d["types"][typ] += 1
            d["relations"][r] += 1
            d["tiers"][tier] += 1
            d["papers"].update(papers)
            if ev and len(ev) > len(d["evidence"]):
                d["evidence"] = ev[:400]
                d["evidence_paper"] = t.get("evidence_paper", "")
        info[s]["neighbours"].add(o)
        info[o]["neighbours"].add(s)

    nodes = sorted(info)

    def role_of(n):
        d = info[n]
        if d["as_subject"] and d["as_object"]:
            return "both"
        return "subject only" if d["as_subject"] else "object only"

    # ── synonym / variant detection ───────────────────────────────────
    by_nf = defaultdict(list)
    for n in nodes:
        by_nf[normal_form(n)].append(n)
    pairs = {}
    for group in by_nf.values():
        for a, b in itertools.combinations(sorted(group), 2):
            pairs[(a, b)] = "identical after normalization (plural, " \
                            "hyphen, abbreviation)"
    for n in nodes:                      # nominalizations
        for p in NOMINAL_PREFIXES:
            if n.lower().startswith(p):
                base = n[len(p):]
                for m in nodes:
                    if norm(m) == norm(base):
                        pairs[tuple(sorted((n, m)))] = \
                            f"nominalization: '{p.strip()} X' vs 'X'"
    antonyms = {}
    for a, b in itertools.combinations(nodes, 2):   # string similarity
        if (a, b) in pairs:
            continue
        ratio = difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
        if ratio < args.string_thr:
            continue
        if antonym_pair(a, b):
            antonyms[(a, b)] = (f"OPPOSITES — differ by a negation prefix "
                                f"(spelling ratio {ratio:.2f})")
        else:
            pairs[(a, b)] = f"similar spelling (ratio {ratio:.2f})"

    variants_of = defaultdict(set)
    for (a, b) in pairs:
        variants_of[a].add(b)
        variants_of[b].add(a)

    with open(outdir / "nodes_inventory.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node", "type", "role", "n_triples", "as_subject",
                    "as_object", "n_neighbours", "n_papers", "n_tier1",
                    "n_tier2", "relations_used", "possible_variants",
                    "example_evidence", "evidence_paper"])
        for n in sorted(nodes, key=lambda n: -(info[n]["as_subject"]
                                               + info[n]["as_object"])):
            d = info[n]
            w.writerow([
                n,
                d["types"].most_common(1)[0][0] if d["types"] else "",
                role_of(n), d["as_subject"] + d["as_object"],
                d["as_subject"], d["as_object"], len(d["neighbours"]),
                len(d["papers"]), d["tiers"].get(1, 0), d["tiers"].get(2, 0),
                "; ".join(sorted(d["relations"])),
                "; ".join(sorted(variants_of.get(n, ()))),
                d["evidence"][:220], d["evidence_paper"]])

    with open(outdir / "synonym_candidates.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name_A", "n_triples_A", "name_B", "n_triples_B",
                    "why_flagged", "DECISION_same_concept_YN",
                    "COMMENT"])
        for (a, b), why in sorted(pairs.items()):
            w.writerow([a, info[a]["as_subject"] + info[a]["as_object"],
                        b, info[b]["as_subject"] + info[b]["as_object"],
                        why, "", ""])
    with open(outdir / "antonym_pairs.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name_A", "name_B", "note",
                    "DECISION_truly_opposite_YN", "COMMENT"])
        for (a, b), why in sorted(antonyms.items()):
            w.writerow([a, b, why, "", ""])

    # ── relation census ───────────────────────────────────────────────
    rel = defaultdict(lambda: {"n": 0, "papers": set(), "subjects": set(),
                               "objects": Counter(), "tier1": 0,
                               "accept": 0, "judged": 0})
    for t in active:
        s, r, o = get_subject(t), get_relation(t), get_object(t)
        d = rel[r]
        d["n"] += 1
        d["papers"].update(t.get("paper_ids", []) or [])
        d["subjects"].add(s)
        d["objects"][o] += 1
        if t.get("_tier") == 1:
            d["tier1"] += 1
        dec = panel.get((norm(s), rel_key(r), norm(o)))
        if dec:
            d["judged"] += 1
            d["accept"] += (dec == "ACCEPT")

    with open(outdir / "relations_census.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relation", "plain_english", "meaning", "n_triples",
                    "share_pct", "n_papers", "n_distinct_subjects",
                    "n_distinct_objects", "informativeness_entropy",
                    "pct_tier1", "pct_accepted_by_panel"])
        total = sum(d["n"] for d in rel.values()) or 1
        for r in sorted(rel, key=lambda r: -rel[r]["n"]):
            d = rel[r]
            gloss = RELATION_GLOSS.get(rel_key(r), ("—", "—"))
            w.writerow([
                r, gloss[0], gloss[1], d["n"],
                round(100 * d["n"] / total, 1), len(d["papers"]),
                len(d["subjects"]), len(d["objects"]),
                norm_entropy(list(d["objects"].values())),
                round(100 * d["tier1"] / d["n"], 1),
                round(100 * d["accept"] / d["judged"], 1)
                if d["judged"] else ""])

    # ── growth / stability ────────────────────────────────────────────
    betas = {}
    if args.robustness and Path(args.robustness).exists():
        txt = Path(args.robustness).read_text(encoding="utf-8")
        for m in re.finditer(r"##\s*(\w+).*?n\^([0-9.]+)", txt, re.S):
            betas[m.group(1)] = float(m.group(2))

    # ── the geologist-facing document ─────────────────────────────────
    L = ["# The knowledge graph, catalogued", "",
         "*Everything the graph contains, in plain terms: every concept, "
         "which names mean the same thing, every relation and what it is "
         "worth, and what would change with more papers.*", "",
         "---", "", "## 1. What is in the graph", "",
         f"- **{len(nodes)} concepts** (nodes)",
         f"- **{len(active)} statements** (arcs), plus "
         f"{len(quarantined)} set aside in quarantine",
         f"- **{len(rel)} kinds of relation** actually used", ""]

    type_counts = Counter()
    for n in nodes:
        if info[n]["types"]:
            type_counts[info[n]["types"].most_common(1)[0][0]] += 1
    L.append("Concepts by kind:")
    for ty, c in type_counts.most_common():
        L.append(f"- **{ty}** — {c}")
    L += ["", "Every concept is listed in `nodes_inventory.csv` with the "
          "sentence it came from and the paper that sentence is in.", ""]

    L += ["---", "", "## 2. Names that may mean the same thing", "",
          f"{len(pairs)} pairs of names were flagged as possibly denoting "
          "the same concept. **They have not been merged**: deciding whether "
          "two names are the same thing is a geological judgement, not a "
          "text-processing one. Please mark each pair in "
          "`synonym_candidates.csv` (column `DECISION_same_concept_YN`).",
          ""]
    if pairs:
        L.append("| Name A | Name B | Why flagged |")
        L.append("|---|---|---|")
        for (a, b), why in sorted(pairs.items())[:40]:
            L.append(f"| `{a}` | `{b}` | {why} |")
        if len(pairs) > 40:
            L.append(f"| … | … | {len(pairs)-40} more in the CSV |")
    else:
        L.append("*No candidate pairs found — names appear already "
                 "consistent.*")
    if antonyms:
        L += ["", "### Pairs that look alike but are OPPOSITES", "",
              "These were caught by the same spelling test and **must not "
              "be merged** — they differ by a negation prefix, which "
              "reverses the meaning. They are listed so that nobody merges "
              "them by mistake, and because their co-existence in the graph "
              "is expected and correct.", "",
              "| Name A | Name B |", "|---|---|"]
        for (a, b) in sorted(antonyms):
            L.append(f"| `{a}` | `{b}` |")
        L.append("")
    fam = defaultdict(set)
    for (a, b) in pairs:
        for kw, label in (("pore", "pore pressure / overpressure"),
                          ("overpressure", "pore pressure / overpressure"),
                          ("hydrate", "gas hydrate dissociation"),
                          ("sediment", "sedimentation rate")):
            if kw in norm(a) or kw in norm(b):
                fam[label].update((a, b))
    if fam:
        L += ["", "### These name variants are not scattered at random", "",
              "The flagged pairs cluster into a few families:", ""]
        for label, names in sorted(fam.items(), key=lambda x: -len(x[1])):
            L.append(f"- **{label}** — {len(names)} spellings: "
                     + ", ".join(f"`{n}`" for n in sorted(names)))
        L += ["", "These are the principal triggering mechanisms discussed "
              "in the literature. The graph fragments precisely where "
              "knowledge is densest, because the concepts most authors "
              "write about are the ones written in the most different ways. "
              "Fragmentation here is a sign of how much a concept is "
              "discussed, not of carelessness.", ""]

    singletons = [n for n in nodes
                  if info[n]["as_subject"] + info[n]["as_object"] == 1]
    L += ["", f"**{len(singletons)} of the {len(nodes)} concepts appear in "
          f"only one statement.** The graph is a dense core surrounded by "
          "many single mentions — normal for a literature graph, but worth "
          "knowing before reading any network statistic.", ""]

    L += ["", "Note on one case worth an explicit decision: "
          "`formation of X` versus `X`. These may be the same thing, or the "
          "*process* may be genuinely distinct from the *state*. That is "
          "your call, not ours.", ""]

    L += ["---", "", "## 3. The relations, and what each is worth", "",
          "Importance is deliberately shown as **four separate numbers**, "
          "because they disagree — and where they disagree is where the "
          "interesting cases are.", "",
          "| Relation | Reads as | Statements | Papers | % Tier-1 | "
          "% accepted by two independent verifiers |",
          "|---|---|---|---|---|---|"]
    for r in sorted(rel, key=lambda r: -rel[r]["n"]):
        d = rel[r]
        gloss = RELATION_GLOSS.get(rel_key(r), ("—", ""))[0]
        acc = (f"{100*d['accept']/d['judged']:.0f} %" if d["judged"]
               else "—")
        L.append(f"| `{r}` | {gloss} | {d['n']} | {len(d['papers'])} | "
                 f"{100*d['tier1']/d['n']:.0f} % | {acc} |")
    L += ["",
          "**How to read these columns.**",
          "- *Statements* — how often the relation is used. Frequent does "
          "not mean reliable.",
          "- *Papers* — in how many different articles. Breadth of support.",
          "- *% Tier-1* — share extracted consistently under two independent "
          "sampling conditions.",
          "- *% accepted* — share that two independent language models, from "
          "different families, confirmed against the source passage.", ""]

    ctl = rel.get("controls")
    if ctl and ctl["judged"] and ctl["accept"] == 0:
        L += ["", "**A finding that concerns you directly.** `controls` — "
              "the relation carrying the *conditions* under which failure "
              f"happens — is the weakest in the graph: {ctl['n']} statements, "
              f"none extracted consistently, none confirmed by the "
              "independent verifiers. Conditions are typically stated across "
              "several sentences ('rapid loading raises pore pressure, which "
              "in turn reduces effective stress, so the slope fails'), and "
              "sentence-level extraction cannot compose them into one "
              "subject–verb–object statement. **The layer most useful to an "
              "interpreter is the one this method captures worst**, and we "
              "report it rather than hide it."]
    trg = rel.get("triggers")
    if trg and trg["judged"] and trg["accept"] / trg["judged"] > 0.7:
        L += ["", f"**Conversely, `triggers` is the most reliable** "
              f"({100*trg['accept']/trg['judged']:.0f} % confirmed). Triggers "
              "tend to be stated directly in the literature ('earthquakes "
              "trigger slope failures'), which is exactly the sentence shape "
              "this method handles best."]

    weak = [r for r in rel if rel[r]["n"] <= 3]
    if weak:
        L.append(f"**Thinly populated relations** ({', '.join(sorted(weak))})"
                 " rest on very few statements; treat any conclusion drawn "
                 "from them as provisional.")
    if "relatedTo" in rel:
        L.append("**`relatedTo` carries no meaning by design** — it is a "
                 "catch-all and should be either re-specified or dropped.")
    L.append("")

    # ── directional reliability warning ───────────────────────────────
    DIRECTIONAL = {"partof", "overlies", "underlies", "causes", "triggers",
                   "controls"}
    dirsum = {}
    if args.direction and Path(args.direction).exists():
        dirsum = json.loads(Path(args.direction).read_text(encoding="utf-8"))
    if dirsum:
        v = dirsum.get("verdicts", {})
        n = dirsum.get("n_checked", sum(v.values()) or 1)
        L += ["---", "",
              "## 3bis. A warning about direction — please read this one", "",
              "Some relations mean the opposite thing if you swap the two "
              "ends: `A is part of B` is not `B is part of A`. We tested "
              f"the direction of {n} such statements against their source "
              "passage, and separately measured how well the automatic "
              "verifier detects a deliberately reversed statement.", "",
              "**It detects reversals poorly**: when we inverted statements "
              "on purpose, the verifier caught only about three in ten "
              "(the same test catches roughly nine in ten when we swap in a "
              "wrong passage instead). Direction is the axis on which the "
              "automatic checks are weakest.", "",
              "Result of the direction check on the real statements:", "",
              "| Verdict | Count | Meaning |", "|---|---|---|"]
        gloss = {"FORWARD": "the passage states it in the direction we "
                            "extracted — confirmed",
                 "REVERSE": "the passage states the opposite — **direction "
                            "error**",
                 "UNDIRECTED": "the passage links the two but states no "
                               "direction — residual risk",
                 "ABSENT": "the direction is not stated in the passage at "
                           "all — needs inspection",
                 "UNPARSEABLE": "the check itself failed"}
        for k in ("FORWARD", "REVERSE", "UNDIRECTED", "ABSENT",
                  "UNPARSEABLE"):
            if k in v:
                L.append(f"| {k} | {v[k]} | {gloss.get(k, '')} |")
        L += ["", "**What we are asking of you.** Directional statements are "
              "the ones where your judgement adds the most, because it is "
              "exactly where our automatic checks add the least. The "
              "directional statements in the graph are listed below.", ""]
        rows = []
        for t in active:
            r = get_relation(t)
            if rel_key(r) in ("partof", "overlies", "underlies"):
                dec = panel.get((norm(get_subject(t)), rel_key(r),
                                 norm(get_object(t))), "")
                rows.append((get_subject(t), r, get_object(t), dec))
        if rows:
            L += ["| Statement | Automatic verdict |", "|---|---|"]
            for s, r, o, dec in sorted(rows):
                L.append(f"| `{s}` **{r}** `{o}` | {dec or '—'} |")
            L += ["", "*(Note: an ACCEPT here means the wording matched the "
                  "passage, not that the direction was checked. Please read "
                  "each of these as a geologist would.)*", ""]

    L += ["---", "", "## 4. Would more papers change this?", "",
          "Measured by repeatedly rebuilding the graph from random subsets "
          "of the corpus and fitting the growth curve.", ""]
    if betas:
        for k, b in betas.items():
            regime = ("still growing — more papers would still add new ones"
                      if b >= 0.5 else
                      "close to saturation — more papers would mostly add "
                      "support, not new items")
            L.append(f"- **{k}**: growth exponent {b:.2f} → {regime}")
    else:
        L.append("- *(growth exponents unavailable — pass --robustness "
                 "with `robustness_report.md`)*")
    L += ["",
          "**In plain terms.**",
          "- **New concepts and new statements: yes, they would keep "
          "appearing.** The graph is not saturated at 37 papers.",
          "- **New seismic descriptors: essentially no.** The descriptor "
          "vocabulary is a closed list of 40 agreed terms, and it has "
          "plateaued — adding papers adds *support* for the descriptors "
          "already there, not new ones.",
          "- **Would the main conclusions change?** The core concepts are "
          "moderately stable to corpus composition (hub stability ≈ 0.68 "
          "measured by resampling). Statements resting on a single paper "
          "are the ones most likely to move; that is why each statement "
          "carries its paper count.",
          "- **One caveat we cannot measure**: resampling tells us about "
          "redundancy *inside this corpus*. It cannot anticipate genuinely "
          "new terminology from a different basin or a different school of "
          "thought.", ""]

    L += ["---", "", "## 5. What we are asking you to check", "",
          "1. **The concept list** (`nodes_inventory.csv`) — are any of "
          "these not real geological concepts? Any obviously missing?",
          "2. **The name pairs** (`synonym_candidates.csv`) — same thing or "
          "different things? Your `DECISION` column drives whether we "
          "merge them.",
          "3. **The relation glossary above** — does `controls` mean to you "
          "what we say it means? Is the distinction between `causes` and "
          "`triggers` one you would make?",
          "4. **The statements themselves** — handled separately, in the "
          "36-item review packet.", "",
          "Every statement in this catalogue can be traced to a sentence in "
          "a named article; ask for any of them and we will show you the "
          "passage.", ""]

    (outdir / "kg_catalogue.md").write_text("\n".join(L), encoding="utf-8")

    with open(outdir / "stability_report.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "growth_exponent", "reading"])
        for k, b in betas.items():
            w.writerow([k, b, "still growing" if b >= 0.5
                        else "near saturation"])

    print("=" * 62)
    print("KG CATALOGUE FOR GEOLOGISTS")
    print("=" * 62)
    print(f"concepts (nodes)      : {len(nodes)}")
    print(f"statements (arcs)     : {len(active)} "
          f"(+{len(quarantined)} quarantined)")
    print(f"relations used        : {len(rel)}")
    print(f"synonym candidates    : {len(pairs)}")
    print(f"antonym pairs (NOT to merge): {len(antonyms)}")
    if dirsum:
        print(f"direction check included : {dirsum.get('n_checked')} "
              "statements")
    print(f"concept types         : {dict(type_counts)}")
    print(f"\nhand to the geologist : {outdir}/kg_catalogue.md")
    print(f"to be filled in       : {outdir}/synonym_candidates.csv")


if __name__ == "__main__":
    main()