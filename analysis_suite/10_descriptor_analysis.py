#!/usr/bin/env python3
"""
10_descriptor_analysis.py — Seismic descriptor statistics and figures.
======================================================================
WHY
    Descriptors are the only part of the KG consumed downstream (Part II
    matching, MTD -> descriptors, one-way). Their support, discriminance and
    ambiguity must be measured, not asserted. Frequencies are computed on
    PAPER support (rebuilt in step 08), not on triple counts, because at
    ~150 triples raw counts are single-digit and statistically anecdotal.

WHAT
    descriptor_statistics.csv  — per descriptor: n_triples, n_papers,
                                 n_objects, tier profile, discriminance
    fig_descriptor_support.png — bar chart (papers + triples), palette
                                 bleu/terracotta, DejaVu Sans
    fig_object_descriptor_heatmap.png — object x descriptor matrix
                                 (colored by max tier of support)
    fig_descriptor_cooccurrence.png   — descriptor x descriptor shared-object
                                 counts
    descriptor_findings.md     — AUTO-INTERPRETED findings; every sentence is
                                 conditioned on computed values (no fabricated
                                 claims), each with the numbers inline.

    Discriminance definition used (documented for the paper):
        discriminance(d) = 1 / n_objects(d)
    A descriptor attached to a single geological object separates classes
    perfectly inside this KG; one attached to MTD + debris flow + slide is a
    poor class separator — directly relevant to Part II clustering labels.

    The descriptor list is DERIVED from hasDescriptor objects in the KG
    itself (no hardcoded lexicon).

USAGE
    python 10_descriptor_analysis.py \
        --kg output/analysis/kg_with_provenance.json --outdir output/analysis
"""

import argparse
import csv
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--outdir", default="output/analysis")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt = apply_style()

    kg = load_kg(args.kg)
    hd = [
        t
        for t in kg["triples"]
        if t["_status"] == "active"
        and get_relation(t).lower() == "hasdescriptor"
    ]
    if not hd:
        raise SystemExit(
            "No hasDescriptor triples found — check relation "
            "field detection in kg_io.py"
        )

    d_triples = defaultdict(int)
    d_papers = defaultdict(set)
    d_objects = defaultdict(set)
    d_tiers = defaultdict(lambda: defaultdict(int))
    obj_desc_tier = {}  # (obj, desc) -> best tier
    for t in hd:
        d, o = get_object(t), get_subject(t)
        d_triples[d] += 1
        d_objects[d].add(o)
        d_tiers[d][t["_tier"]] += 1
        for p in t.get("paper_ids", []):
            d_papers[d].add(p)
        best = obj_desc_tier.get((o, d), 99)
        obj_desc_tier[(o, d)] = min(best, t["_tier"] if t["_tier"] else 3)

    descs = sorted(d_triples, key=lambda d: (-len(d_papers[d]), -d_triples[d]))
    objects = sorted({o for (o, _) in obj_desc_tier})

    # ── CSV ───────────────────────────────────────────────────────────
    with open(
        outdir / "descriptor_statistics.csv", "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "descriptor",
                "n_triples",
                "n_papers",
                "n_objects",
                "objects",
                "n_tier1",
                "n_tier2",
                "discriminance",
            ]
        )
        for d in descs:
            w.writerow(
                [
                    d,
                    d_triples[d],
                    len(d_papers[d]),
                    len(d_objects[d]),
                    ";".join(sorted(d_objects[d])),
                    d_tiers[d][1],
                    d_tiers[d][2],
                    round(1.0 / max(len(d_objects[d]), 1), 3),
                ]
            )

    # ── fig 1: support bar chart ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(descs))
    ax.bar(
        x - 0.2,
        [len(d_papers[d]) for d in descs],
        0.4,
        color=BLUE,
        label="papers (consensus)",
    )
    ax.bar(
        x + 0.2,
        [d_triples[d] for d in descs],
        0.4,
        color=TERRACOTTA,
        label="triples",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(descs, rotation=45, ha="right")
    ax.set_ylabel("count")
    ax.set_title("Seismic descriptor support", fontsize=18)
    ax.legend(frameon=False)
    fig.savefig(outdir / "fig_descriptor_support.png")
    plt.close(fig)

    # ── fig 2: object x descriptor heatmap (by tier) ──────────────────
    mat = np.zeros((len(objects), len(descs)))
    for (o, d), tier in obj_desc_tier.items():
        mat[objects.index(o), descs.index(d)] = {1: 2.0, 2: 1.0}.get(tier, 0.5)
    fig, ax = plt.subplots(
        figsize=(max(8, 0.7 * len(descs)), max(4, 0.45 * len(objects)))
    )
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(range(len(descs)))
    ax.set_xticklabels(descs, rotation=45, ha="right")
    ax.set_yticks(range(len(objects)))
    ax.set_yticklabels(objects)
    ax.set_title("Object × descriptor (2=Tier-1, 1=Tier-2)", fontsize=18)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(outdir / "fig_object_descriptor_heatmap.png")
    plt.close(fig)

    # ── fig 3: descriptor co-occurrence (shared objects) ──────────────
    co = np.zeros((len(descs), len(descs)))
    for i, a in enumerate(descs):
        for j, b in enumerate(descs):
            co[i, j] = len(d_objects[a] & d_objects[b])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(co, cmap="Oranges")
    ax.set_xticks(range(len(descs)))
    ax.set_xticklabels(descs, rotation=45, ha="right")
    ax.set_yticks(range(len(descs)))
    ax.set_yticklabels(descs)
    ax.set_title("Descriptor co-occurrence (shared objects)", fontsize=18)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(outdir / "fig_descriptor_cooccurrence.png")
    plt.close(fig)

    # ── auto-interpretation (data-conditioned only) ───────────────────
    lines = ["# Descriptor findings (auto-generated, data-conditioned)", ""]
    top = descs[0]
    lines.append(
        f"- **Most supported descriptor**: `{top}` "
        f"({len(d_papers[top])} papers, {d_triples[top]} triples, "
        f"objects: {', '.join(sorted(d_objects[top]))})."
    )
    mtd_like = [
        o
        for o in objects
        if "mass transport" in o.lower() or o.lower() == "mtd"
    ]
    if mtd_like:
        mo = mtd_like[0]
        mtd_desc = sorted(
            [d for d in descs if mo in d_objects[d]],
            key=lambda d: -len(d_papers[d]),
        )

        def best_tier(d):
            ks = [k for k, v in d_tiers[d].items() if v > 0 and k > 0]
            return f"T{min(ks)}" if ks else "T?"

        lines.append(
            f"- **Descriptors of `{mo}`** (by paper support): "
            + ", ".join(
                f"`{d}` ({len(d_papers[d])}p, {best_tier(d)})"
                for d in mtd_desc
            )
            + "."
        )
        t1 = [d for d in mtd_desc if d_tiers[d][1] > 0]
        t2only = [d for d in mtd_desc if d_tiers[d][1] == 0]
        if t1:
            lines.append(
                f"- Tier-1 MTD descriptors: "
                + ", ".join(f"`{d}`" for d in t1)
                + "."
            )
        if t2only:
            lines.append(
                f"- Tier-2-only MTD descriptors: "
                + ", ".join(f"`{d}`" for d in t2only)
                + " — extracted textual salience does not always "
                "match interpreter-perceived salience; report as "
                "a finding, not an error."
            )
    discr = [d for d in descs if len(d_objects[d]) == 1]
    ambig = [d for d in descs if len(d_objects[d]) >= 3]
    if discr:
        lines.append(
            f"- **Discriminant descriptors** (single object): "
            + ", ".join(f"`{d}`→{next(iter(d_objects[d]))}" for d in discr)
            + "."
        )
    if ambig:
        lines.append(
            f"- **Ambiguous descriptors** (≥3 objects — poor class "
            f"separators for Part II): "
            + ", ".join(f"`{d}`" for d in ambig)
            + "."
        )
    weak = [d for d in descs if len(d_papers[d]) <= 1]
    if weak:
        lines.append(
            f"- **Weakly documented** (≤1 paper): "
            + ", ".join(f"`{d}`" for d in weak)
            + "."
        )
    (outdir / "descriptor_findings.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        f"[10] {len(descs)} descriptors, {len(objects)} objects — "
        f"outputs in {outdir}"
    )


if __name__ == "__main__":
    main()
