#!/usr/bin/env python3
"""
11_relation_analysis.py — Relation-level statistics.
====================================================
WHY
    The review identified three untested claims: (a) causes/triggers/controls
    may be a mapping artifact rather than a corpus distinction, (b) relatedTo
    carries no information, (c) partOf direction may be inconsistent. This
    script quantifies all three instead of arguing them.

METRICS (documented for the paper)
    informativeness(r) = normalized Shannon entropy of the object distribution
        given r, H(O|r)/log2(|O_r|). LOW entropy = the relation always points
        to the same few objects (predictable, low information); HIGH = spread.
        Reported together with n_pairs so tiny relations aren't over-read.
    redundancy: (subject, object) pairs that appear under >=2 distinct
        relations — direct evidence of relation-level ambiguity (e.g. the
        causes/triggers overlap).
    direction check: pairs (a, r, b) AND (b, r, a) both present — flags
        inconsistent direction for partOf / overlies / underlies.

OUTPUTS
    relation_statistics.csv, relation_object_matrix.csv,
    relation_redundancy.csv, fig_relation_distribution.png,
    relation_findings.md

USAGE
    python 11_relation_analysis.py \
        --kg output/analysis/kg_with_provenance.json --outdir output/analysis
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from kg_io import (
    BLUE,
    TERRACOTTA,
    apply_style,
    get_object,
    get_relation,
    get_subject,
    load_kg,
)


def norm_entropy(counts):
    tot = sum(counts)
    if tot == 0 or len(counts) <= 1:
        return 0.0
    h = -sum((c / tot) * math.log2(c / tot) for c in counts if c)
    return round(h / math.log2(len(counts)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--outdir", default="output/analysis")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt = apply_style()

    kg = load_kg(args.kg)
    active = [t for t in kg["triples"] if t["_status"] == "active"]

    r_triples = defaultdict(int)
    r_papers = defaultdict(set)
    r_objects = defaultdict(lambda: defaultdict(int))
    r_pairs = defaultdict(set)
    pair_rels = defaultdict(set)
    directed = set()
    for t in active:
        s, r, o = get_subject(t), get_relation(t), get_object(t)
        r_triples[r] += 1
        r_objects[r][o] += 1
        r_pairs[r].add((s, o))
        pair_rels[(s, o)].add(r)
        directed.add((s, r, o))
        for p in t.get("paper_ids", []):
            r_papers[r].add(p)

    rels = sorted(r_triples, key=lambda r: -r_triples[r])

    with open(
        outdir / "relation_statistics.csv", "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "relation",
                "n_triples",
                "n_papers",
                "n_unique_pairs",
                "n_unique_objects",
                "object_entropy_norm",
            ]
        )
        for r in rels:
            w.writerow(
                [
                    r,
                    r_triples[r],
                    len(r_papers[r]),
                    len(r_pairs[r]),
                    len(r_objects[r]),
                    norm_entropy(list(r_objects[r].values())),
                ]
            )

    # relation x object matrix
    all_objects = sorted({o for r in rels for o in r_objects[r]})
    with open(
        outdir / "relation_object_matrix.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow(["relation"] + all_objects)
        for r in rels:
            w.writerow([r] + [r_objects[r].get(o, 0) for o in all_objects])

    # redundancy: same pair, several relations
    red_rows = [
        (s, o, ";".join(sorted(rs)))
        for (s, o), rs in sorted(pair_rels.items())
        if len(rs) >= 2
    ]
    with open(
        outdir / "relation_redundancy.csv", "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(["subject", "object", "relations"])
        w.writerows(red_rows)

    # direction inconsistencies
    bidir = sorted(
        {
            (min(s, o), r, max(s, o))
            for (s, r, o) in directed
            if (o, r, s) in directed
        }
    )

    # figure
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(rels))
    ax.bar(
        x - 0.2, [r_triples[r] for r in rels], 0.4, color=BLUE, label="triples"
    )
    ax.bar(
        x + 0.2,
        [len(r_papers[r]) for r in rels],
        0.4,
        color=TERRACOTTA,
        label="papers",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(rels, rotation=45, ha="right")
    ax.set_title("Relation distribution", fontsize=18)
    ax.legend(frameon=False)
    fig.savefig(outdir / "fig_relation_distribution.png")
    plt.close(fig)

    # findings (data-conditioned)
    lines = ["# Relation findings (auto-generated)", ""]
    lines.append(
        f"- **Most used**: `{rels[0]}` ({r_triples[rels[0]]} "
        f"triples, {len(r_papers[rels[0]])} papers)."
    )
    informative = sorted(
        [r for r in rels if len(r_pairs[r]) >= 3],
        key=lambda r: -norm_entropy(list(r_objects[r].values())),
    )
    if informative:
        lines.append(
            f"- **Most informative** (highest normalized object "
            f"entropy among relations with ≥3 pairs): "
            f"`{informative[0]}`."
        )
    if "relatedTo" in r_triples:
        lines.append(
            f"- `relatedTo`: {r_triples['relatedTo']} triples — "
            "candidate for purge/requalification (no semantic "
            "content)."
        )
    causal = [
        r for r in rels if r.lower() in ("causes", "triggers", "controls")
    ]
    overlap = [
        row
        for row in red_rows
        if set(row[2].split(";")) & {"causes", "triggers", "controls"}
        and len(set(row[2].split(";")) & {"causes", "triggers", "controls"})
        >= 2
    ]
    if causal:
        lines.append(
            f"- **causal family overlap**: {len(overlap)} "
            "(subject, object) pairs asserted under ≥2 of "
            "{causes, triggers, controls}. "
            + (
                "If >0, the causal distinction is partly a mapping "
                "artifact — run the merge test before defending it."
                if overlap
                else "No overlapping pair: the distinction is at least "
                "internally consistent."
            )
        )
    lines.append(
        f"- **Bidirectional pairs** (direction inconsistency "
        f"candidates): {len(bidir)}"
        + (
            " — " + "; ".join(f"{a}↔{b} [{r}]" for a, r, b in bidir[:10])
            if bidir
            else "."
        )
    )
    (outdir / "relation_findings.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"[11] {len(rels)} relations analyzed — outputs in {outdir}")


if __name__ == "__main__":
    main()
