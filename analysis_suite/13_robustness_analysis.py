#!/usr/bin/env python3
"""
13_robustness_analysis.py — Growth & stability via paper subsampling.
=====================================================================
WHY
    "What happens with +50/+100/+500 papers?" cannot be answered by
    simulation of fictitious articles — that would be fabrication. What CAN
    be done rigorously with 37 papers is the standard corpus-growth method:
    bootstrap subsampling of the EXISTING papers (using the provenance
    rebuilt in step 08), fitting Heaps' law V(n) = K * n^beta to the
    accumulation curves, and EXTRAPOLATING with the fitted exponent —
    explicitly labeled as an extrapolation, never as an observation.

    beta < 1 with a flattening curve  -> the vocabulary/graph is CONVERGING
    (the query set saturates: more papers densify support, not coverage);
    beta close to 1 -> still growing linearly (unlikely here, and if
    observed it contradicts the fixed-query-set ceiling — interesting either
    way).

WHAT
    robustness_curves.csv   — mean & CI of nodes/edges/descriptors at each
                              subsample size (B bootstrap draws)
    robustness_report.md    — fitted Heaps exponents, extrapolations to
                              n+50/+100/+500, hub stability (Jaccard of the
                              top-10 degree nodes between 60% subsamples and
                              the full corpus), relation-distribution
                              stability (L1 distance)
    fig_growth_curves.png   — accumulation curves with fits, blue/terracotta

LIMITATION (stated in output): subsampling measures redundancy INSIDE the
    current corpus/query design; it cannot anticipate genuinely new
    terminology from unseen basins. The report says so explicitly.

USAGE
    python 13_robustness_analysis.py \
        --kg output/analysis/kg_with_provenance.json \
        --outdir output/analysis [--boot 200] [--seed 42]
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


def heaps_fit(ns, vs):
    """log-log OLS fit of V = K * n^beta. Returns (K, beta) or None."""
    ns, vs = np.asarray(ns, float), np.asarray(vs, float)
    m = (ns > 0) & (vs > 0)
    if m.sum() < 3:
        return None
    b, a = np.polyfit(np.log(ns[m]), np.log(vs[m]), 1)
    return float(np.exp(a)), float(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--outdir", default="output/analysis")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt = apply_style()
    rng = np.random.default_rng(args.seed)

    kg = load_kg(args.kg)
    active = [t for t in kg["triples"] if t["_status"] == "active"]

    paper2triples = defaultdict(list)
    n_no_prov = 0
    for t in active:
        ps = t.get("paper_ids", [])
        if not ps:
            n_no_prov += 1
        for p in ps:
            paper2triples[p].append(t)
    papers = sorted(paper2triples)
    P = len(papers)
    if P < 5:
        raise SystemExit(
            f"Only {P} papers resolved from provenance "
            f"({n_no_prov} triples without paper_ids). Run "
            "08_rebuild_provenance.py first and check chunk-id resolution."
        )

    def kg_counts(paper_subset):
        nodes, edges, descs, rels = set(), set(), set(), defaultdict(int)
        seen = set()
        for p in paper_subset:
            for t in paper2triples[p]:
                key = id(t)
                if key in seen:
                    continue
                seen.add(key)
                s, r, o = get_subject(t), get_relation(t), get_object(t)
                nodes.update((s, o))
                edges.add((s, r, o))
                rels[r] += 1
                if r.lower() == "hasdescriptor":
                    descs.add(o)
        return nodes, edges, descs, rels

    sizes = sorted(
        {
            max(2, int(P * f))
            for f in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        }
    )
    curves = {"nodes": {}, "edges": {}, "descriptors": {}}
    for n in sizes:
        buf = {k: [] for k in curves}
        for _ in range(args.boot if n < P else 1):
            sub = rng.choice(papers, size=n, replace=False)
            nodes, edges, descs, _ = kg_counts(sub)
            buf["nodes"].append(len(nodes))
            buf["edges"].append(len(edges))
            buf["descriptors"].append(len(descs))
        for k in curves:
            arr = np.array(buf[k])
            curves[k][n] = (
                arr.mean(),
                np.percentile(arr, 2.5),
                np.percentile(arr, 97.5),
            )

    with open(
        outdir / "robustness_curves.csv", "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "n_papers",
                "nodes_mean",
                "nodes_lo",
                "nodes_hi",
                "edges_mean",
                "edges_lo",
                "edges_hi",
                "desc_mean",
                "desc_lo",
                "desc_hi",
            ]
        )
        for n in sizes:
            row = [n]
            for k in ("nodes", "edges", "descriptors"):
                row += [round(v, 1) for v in curves[k][n]]
            w.writerow(row)

    # ── Heaps fits + extrapolation ────────────────────────────────────
    fits, extrap = {}, {}
    for k in curves:
        fit = heaps_fit(sizes, [curves[k][n][0] for n in sizes])
        fits[k] = fit
        if fit:
            K, beta = fit
            extrap[k] = {P + d: K * (P + d) ** beta for d in (50, 100, 500)}

    # ── hub stability ─────────────────────────────────────────────────
    def top10(paper_subset):
        deg = defaultdict(int)
        _, edges, _, _ = kg_counts(paper_subset)
        for s, _, o in sorted(edges):
            deg[s] += 1
            deg[o] += 1
        return set(sorted(deg, key=lambda x: -deg[x])[:10])

    full_top = top10(papers)
    jac = []
    for _ in range(args.boot):
        sub = rng.choice(papers, size=max(2, int(0.6 * P)), replace=False)
        t10 = top10(sub)
        jac.append(len(t10 & full_top) / len(t10 | full_top))
    hub_stab = float(np.mean(jac))

    # relation distribution stability (L1, 60% vs full)
    _, _, _, full_rels = kg_counts(papers)
    tot = sum(full_rels.values())
    l1 = []
    for _ in range(args.boot):
        sub = rng.choice(papers, size=max(2, int(0.6 * P)), replace=False)
        _, _, _, r = kg_counts(sub)
        st = sum(r.values()) or 1
        keys = set(full_rels) | set(r)
        l1.append(
            sum(
                abs(full_rels.get(k, 0) / tot - r.get(k, 0) / st) for k in keys
            )
            / 2
        )
    rel_stab = 1.0 - float(np.mean(l1))

    # ── figure ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for k, color in (
        ("nodes", BLUE),
        ("edges", TERRACOTTA),
        ("descriptors", "#6B8F71"),
    ):
        means = [curves[k][n][0] for n in sizes]
        lo = [curves[k][n][1] for n in sizes]
        hi = [curves[k][n][2] for n in sizes]
        ax.plot(sizes, means, "-o", color=color, label=k, ms=4)
        ax.fill_between(sizes, lo, hi, color=color, alpha=0.15)
        if fits[k]:
            K, beta = fits[k]
            xs = np.linspace(sizes[0], P, 100)
            ax.plot(
                xs,
                K * xs**beta,
                "--",
                color=color,
                lw=1,
                label=f"{k} fit β={beta:.2f}",
            )
    ax.set_xlabel("number of papers")
    ax.set_ylabel("count")
    ax.set_title("KG growth under paper subsampling", fontsize=18)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outdir / "fig_growth_curves.png")
    plt.close(fig)

    # ── report ────────────────────────────────────────────────────────
    lines = [
        "# Robustness report (auto-generated)",
        "",
        f"Corpus: {P} papers resolved from provenance "
        f"({n_no_prov} triples without paper_ids, excluded).",
        f"Bootstrap draws per size: {args.boot} (seed {args.seed}).",
        "",
    ]
    for k in ("nodes", "edges", "descriptors"):
        if fits[k]:
            K, beta = fits[k]
            regime = (
                "CONVERGING (sub-linear growth — the fixed query set "
                "saturates; additional papers densify support rather "
                "than expand coverage)"
                if beta < 0.7
                else "slowly growing"
                if beta < 0.9
                else "near-linear growth (unexpected under a fixed query "
                "set — inspect)"
            )
            lines.append(f"## {k}")
            lines.append(
                f"- Heaps fit: V(n) ≈ {K:.1f} · n^{beta:.2f} — {regime}."
            )
            lines.append(
                "- Extrapolation (EXTRAPOLATED, not observed): "
                + ", ".join(f"n={n}: ≈{v:.0f}" for n, v in extrap[k].items())
                + "."
            )
            lines.append("")
    lines.append(f"## Stability")
    lines.append(
        f"- Hub stability (Jaccard, top-10 degree, 60% subsample "
        f"vs full): **{hub_stab:.2f}** — "
        + (
            "hubs are robust to corpus composition."
            if hub_stab >= 0.7
            else "hubs depend on corpus composition; interpret hub-based "
            "claims cautiously."
        )
    )
    lines.append(
        f"- Relation-distribution stability (1 − L1/2): **{rel_stab:.2f}**."
    )
    lines.append("")
    lines.append("## Stated limitation")
    lines.append(
        "`hasDescriptor` is a closed-world relation: descriptor "
        "growth is upper-bounded by the size of the canonical "
        "descriptor vocabulary. Truncate the descriptor "
        "extrapolation at that bound in the manuscript; only the "
        "node/edge extrapolations are meaningful beyond it."
    )
    lines.append(
        "Subsampling measures redundancy inside the current corpus "
        "and query design. It cannot anticipate genuinely new "
        "terminology from unseen basins; the extrapolations are "
        "lower bounds on novelty and must be labeled as such in "
        "the manuscript."
    )
    (outdir / "robustness_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        f"[13] robustness done — hub stability {hub_stab:.2f} — "
        f"outputs in {outdir}"
    )


if __name__ == "__main__":
    main()
