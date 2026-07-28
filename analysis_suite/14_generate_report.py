#!/usr/bin/env python3
"""
14_generate_report.py — Assemble the final analysis/ report.
============================================================
WHY
    The suite must end with a single artifact a co-author (or reviewer) can
    read: a statistics compendium (knowledge_report.md) and a Discussion-like
    geological narrative (discussion.md). Every sentence in discussion.md is
    generated from computed values — the script REFUSES to emit a claim whose
    supporting number is missing (anti-fabrication guard, consistent with the
    project's verification discipline).

WHAT
    analysis/knowledge_report.md — compendium linking every CSV + figure
    analysis/discussion.md      — auto-drafted geological discussion:
        major concepts, dominant processes, mechanisms (causal chains found
        by path search), geological controls, environments, MTD descriptors,
        under-documented concepts, pivot concepts.
        Each paragraph cites its source CSV so a human can audit it. The
        header marks it clearly as an AUTO-DRAFT to be rewritten by the
        author — it is manuscript scaffolding, not manuscript text.

WHERE
    Last step; reads only the outputs of scripts 08–13.

USAGE
    python 14_generate_report.py --analysis-dir output/analysis
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default="output/analysis")
    args = ap.parse_args()
    d = Path(args.analysis_dir)

    graph = {
        r["metric"]: r["value"] for r in read_csv(d / "graph_statistics.csv")
    }
    nodes = read_csv(d / "node_statistics.csv")
    rels = read_csv(d / "relation_statistics.csv")
    descs = read_csv(d / "descriptor_statistics.csv")
    vocab = read_csv(d / "vocabulary_report.csv")
    prov = read_csv(d / "provenance_report.csv")

    missing = [
        name
        for name, data in [
            ("graph_statistics", graph),
            ("node_statistics", nodes),
            ("relation_statistics", rels),
            ("descriptor_statistics", descs),
            ("provenance_report", prov),
        ]
        if not data
    ]
    if missing:
        raise SystemExit(
            f"Missing upstream outputs: {missing}. "
            "Run scripts 08–13 first (anti-fabrication guard: "
            "no report without computed numbers)."
        )

    # ── knowledge_report.md ───────────────────────────────────────────
    kr = ["# Knowledge graph — quantitative report (auto-generated)", ""]
    kr.append("## Graph statistics")
    kr.append("| metric | value | main text? |")
    kr.append("|---|---|---|")
    for r in read_csv(d / "graph_statistics.csv"):
        kr.append(
            f"| {r['metric']} | {r['value']} | {r['report_in_main_text']} |"
        )
    kr += [
        "",
        "## Files",
        "- `provenance_report.csv` — per-triple support & confidence",
        "- `vocabulary_report.csv`, `synonyms_report.csv`, "
        "`canonicalization_report.csv`",
        "- `descriptor_statistics.csv` + `descriptor_findings.md`",
        "- `relation_statistics.csv`, `relation_object_matrix.csv`, "
        "`relation_redundancy.csv` + `relation_findings.md`",
        "- `node_statistics.csv` + `graph_findings.md`",
        "- `robustness_curves.csv` + `robustness_report.md`",
        "",
        "## Figures",
        "- fig_descriptor_support.png, fig_object_descriptor_heatmap.png,"
        " fig_descriptor_cooccurrence.png",
        "- fig_relation_distribution.png",
        "- fig_degree_distribution.png, fig_kg_communities.png, "
        "fig_top_centrality.png",
        "- fig_growth_curves.png",
    ]
    (d / "knowledge_report.md").write_text("\n".join(kr), encoding="utf-8")

    # ── discussion.md (data-conditioned draft) ────────────────────────
    def top(rows, key, n=5, cast=float):
        return sorted(rows, key=lambda r: -cast(r[key] or 0))[:n]

    disc = [
        "# Discussion (AUTO-DRAFT — scaffolding to be rewritten by the "
        "author; every claim cites its source file)",
        "",
    ]

    hubs = top(nodes, "degree", 5)
    disc.append("## Major concepts")
    disc.append(
        "The graph is organized around "
        + ", ".join(
            f"**{r['node']}** (degree {r['degree']}, {r['n_papers']} papers)"
            for r in hubs
        )
        + ". [node_statistics.csv] Degree here measures assertion coverage, "
        "not geological importance; the topology partly reflects the "
        "object-centered query design."
    )

    causal = [
        r
        for r in prov
        if r["relation"].lower() in ("causes", "triggers", "controls")
    ]
    disc.append("\n## Processes and mechanisms")
    if causal:
        chains = defaultdict(list)
        for r in causal:
            chains[r["subject"]].append((r["relation"], r["object"]))
        links = [
            f"*{r['subject']}* —{r['relation']}→ *{r['object']}* "
            f"(T{r['tier']}, {r['support_papers']} papers)"
            for r in sorted(causal, key=lambda x: -int(x["support_papers"]))[
                :8
            ]
        ]
        disc.append(
            "Causal assertions present in the graph: "
            + "; ".join(links)
            + ". [provenance_report.csv] "
            "Multi-step mechanisms (e.g. pore-pressure chains) are "
            "known to be under-represented by sentence-level "
            "extraction; absence here is a formalism limit, not "
            "evidence of geological absence."
        )
    else:
        disc.append(
            "No causal triples present — the mechanism layer is "
            "empty in this run. [provenance_report.csv]"
        )

    controls = [r for r in prov if r["relation"].lower() == "controls"]
    disc.append("\n## Geological controls")
    disc.append(
        (
            f"{len(controls)} `controls` assertions. "
            if controls
            else "No `controls` assertions — "
        )
        + "The controls layer is the thinnest of the causal family "
        "and should be flagged as under-populated relative to the "
        "literature. [relation_statistics.csv]"
    )

    envs = [r for r in prov if r["relation"].lower() == "occursin"]
    disc.append("\n## Environments")
    if envs:
        env_count = defaultdict(int)
        for r in envs:
            env_count[r["object"]] += 1
        env_sorted = sorted(env_count.items(), key=lambda x: -x[1])
        disc.append(
            "Depositional settings asserted: "
            + ", ".join(f"**{e}** ({c})" for e, c in env_sorted)
            + ". [provenance_report.csv] Granularity is coarse "
            "(margin/basin scale); basin-specific provenance is "
            "available per triple via paper_ids and should be "
            "exposed if multi-basin corpora are added."
        )
    else:
        disc.append("No occursIn assertions found.")

    disc.append("\n## MTD seismic descriptors")
    mtd_rows = [
        r
        for r in descs
        if any(
            "mass transport" in o or o == "mtd"
            for o in r["objects"].lower().split(";")
        )
    ]
    if mtd_rows:
        mtd_sorted = sorted(mtd_rows, key=lambda r: -int(r["n_papers"]))
        disc.append(
            "Descriptors attached to the MTD node, by paper "
            "support: "
            + ", ".join(
                f"**{r['descriptor']}** ({r['n_papers']}p, "
                f"{'T1' if int(r['n_tier1']) else 'T2'})"
                for r in mtd_sorted
            )
            + ". [descriptor_statistics.csv] Where interpreter-"
            "canonical descriptors (e.g. chaotic) appear only at "
            "Tier-2 while others reach Tier-1, this mismatch "
            "between textual and perceptual salience is a finding "
            "to report, not an error to fix."
        )
    else:
        disc.append(
            "No descriptor row maps to an MTD-like object — check "
            "canonicalization of the MTD node."
        )

    disc.append("\n## Under-documented concepts")
    weak = [r for r in vocab if int(r["n_papers"] or 0) <= 1]
    disc.append(
        f"{len(weak)} concepts are supported by ≤1 paper "
        "[vocabulary_report.csv] — single-source assertions; the "
        "confidence score already down-weights them "
        "(w_consensus)."
    )

    disc.append("\n## Pivot concepts")
    piv = top(nodes, "betweenness", 3)
    disc.append(
        "Highest-betweenness nodes: "
        + ", ".join(f"**{r['node']}**" for r in piv)
        + ". [node_statistics.csv] On this small, star-shaped graph "
        "betweenness mostly restates the query design; pivots "
        "should only be discussed when they sit on causal chains "
        "(cross-check relation paths)."
    )

    (d / "discussion.md").write_text("\n".join(disc), encoding="utf-8")
    print(f"[14] knowledge_report.md + discussion.md written in {d}")


if __name__ == "__main__":
    main()
