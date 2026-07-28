#!/usr/bin/env python3
"""
12_kg_analysis.py — Graph-level statistics, centralities, communities.
======================================================================
WHY
    The manuscript needs a quantitative KG portrait, and the review requires
    it to be reported honestly: on a ~150-edge, query-design-shaped graph,
    several network metrics are numerically defined but scientifically weak.
    Every metric is therefore exported WITH an explicit caveat flag
    (report_in_main_text: yes/annex) so the paper does not over-read them.

WHAT
    graph_statistics.csv  — nodes, edges, components, density, avg degree,
                            diameter & avg path length (largest component),
                            clustering, assortativity, modularity,
                            n_communities, n_types, n_relations, n_cycles
    node_statistics.csv   — degree / in / out / betweenness / closeness /
                            eigenvector / pagerank per node + community id
                            + support_papers (from step 08)
    fig_degree_distribution.png (log-log), fig_kg_communities.png,
    fig_top_centrality.png
    graph_findings.md     — hubs, authorities, isolated concepts, pivots,
                            with the geological reading rules from the review
                            (descriptor degree = seismic polysemy; causal-
                            chain betweenness = mechanistic pivot).

USAGE
    python 12_kg_analysis.py \
        --kg output/analysis/kg_with_provenance.json --outdir output/analysis
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from kg_io import (load_kg, get_subject, get_object, get_relation,
                   apply_style, BLUE, TERRACOTTA, PALETTE)


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

    G = nx.MultiDiGraph()
    node_papers = defaultdict(set)
    types = set()
    for t in active:
        s, r, o = get_subject(t), get_relation(t), get_object(t)
        G.add_edge(s, o, relation=r, tier=t["_tier"])
        for k in ("source_type", "target_type", "subject_type",
                  "object_type"):
            if t.get(k):
                types.add(str(t[k]))
        for p in t.get("paper_ids", []):
            node_papers[s].add(p)
            node_papers[o].add(p)

    U = nx.Graph(G.to_undirected())          # simple undirected for metrics
    D = nx.DiGraph(G)                        # simple directed
    comps = list(nx.connected_components(U))
    largest = U.subgraph(max(comps, key=len))

    # communities (greedy modularity on largest component)
    try:
        communities = list(
            nx.community.greedy_modularity_communities(largest))
        modularity = nx.community.modularity(largest, communities)
    except Exception:
        communities, modularity = [], float("nan")
    node2comm = {}
    for ci, comm in enumerate(communities):
        for n in comm:
            node2comm[n] = ci

    def safe(fn, *a, **k):
        try:
            return fn(*a, **k)
        except Exception:
            return float("nan")

    stats = [
        ("n_nodes", U.number_of_nodes(), "yes"),
        ("n_edges_multi", G.number_of_edges(), "yes"),
        ("n_edges_simple", U.number_of_edges(), "yes"),
        ("n_relation_types",
         len({d["relation"] for _, _, d in G.edges(data=True)}), "yes"),
        ("n_entity_types_observed", len(types), "yes"),
        ("n_components", len(comps), "yes"),
        ("largest_component_size", largest.number_of_nodes(), "yes"),
        ("n_communities", len(communities), "annex"),
        ("modularity", round(modularity, 3) if modularity == modularity
         else "nan", "annex"),
        ("density", round(nx.density(U), 4), "yes"),
        ("avg_degree",
         round(2 * U.number_of_edges() / max(U.number_of_nodes(), 1), 2),
         "yes"),
        ("diameter_largest_cc", safe(nx.diameter, largest), "annex"),
        ("avg_path_length_largest_cc",
         round(safe(nx.average_shortest_path_length, largest), 2), "annex"),
        ("avg_clustering", round(safe(nx.average_clustering, U), 3),
         "annex"),
        ("degree_assortativity",
         round(safe(nx.degree_assortativity_coefficient, U), 3), "annex"),
        ("n_cycles_directed",
         len(list(nx.simple_cycles(D))) if D.number_of_nodes() < 500
         else "skipped", "annex"),
    ]
    with open(outdir / "graph_statistics.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "report_in_main_text"])
        w.writerows(stats)

    # ── centralities ──────────────────────────────────────────────────
    deg = dict(U.degree())
    btw = safe(nx.betweenness_centrality, U) or {}
    clo = safe(nx.closeness_centrality, U) or {}
    try:
        eig = nx.eigenvector_centrality(U, max_iter=1000)
    except Exception:
        eig = {n: float("nan") for n in U}
    pr = nx.pagerank(D)
    indeg, outdeg = dict(D.in_degree()), dict(D.out_degree())

    with open(outdir / "node_statistics.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node", "degree", "in_degree", "out_degree",
                    "betweenness", "closeness", "eigenvector", "pagerank",
                    "community", "n_papers"])
        for n in sorted(U, key=lambda n: -deg[n]):
            w.writerow([n, deg[n], indeg.get(n, 0), outdeg.get(n, 0),
                        round(btw.get(n, float("nan")), 4),
                        round(clo.get(n, float("nan")), 4),
                        round(eig.get(n, float("nan")), 4),
                        round(pr.get(n, float("nan")), 4),
                        node2comm.get(n, -1), len(node_papers[n])])

    # ── figures ───────────────────────────────────────────────────────
    degs = sorted(deg.values(), reverse=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(range(1, len(degs) + 1), degs, "o", color=BLUE, ms=5)
    ax.set_xlabel("rank")
    ax.set_ylabel("degree")
    ax.set_title("Degree distribution (rank plot)", fontsize=18)
    fig.savefig(outdir / "fig_degree_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(U, seed=42, k=0.6)
    colors = [PALETTE[node2comm.get(n, -1) % len(PALETTE)] for n in U]
    sizes = [80 + 40 * deg[n] for n in U]
    nx.draw_networkx_edges(U, pos, ax=ax, alpha=0.25)
    nx.draw_networkx_nodes(U, pos, ax=ax, node_color=colors,
                           node_size=sizes, alpha=0.9)
    labels = {n: n for n in U if deg[n] >= max(2, np.percentile(degs, 80))}
    nx.draw_networkx_labels(U, pos, labels, ax=ax, font_size=8,
                            font_family="DejaVu Sans")
    ax.set_title("Knowledge graph — communities", fontsize=20)
    ax.axis("off")
    fig.savefig(outdir / "fig_kg_communities.png")
    plt.close(fig)

    topN = sorted(U, key=lambda n: -deg[n])[:15]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(topN))[::-1], [deg[n] for n in topN],
            color=TERRACOTTA)
    ax.set_yticks(range(len(topN))[::-1])
    ax.set_yticklabels(topN)
    ax.set_title("Top-15 nodes by degree", fontsize=18)
    fig.savefig(outdir / "fig_top_centrality.png")
    plt.close(fig)

    # ── findings ──────────────────────────────────────────────────────
    lines = ["# Graph findings (auto-generated)", "",
             "**Caveat (do not remove):** on a query-design-shaped graph of "
             f"{U.number_of_edges()} edges, betweenness/closeness/eigenvector"
             " mainly restate the star topology induced by object-centered "
             "queries. Degree and n_papers are the metrics with geological "
             "content; the rest belongs in an annex.", ""]
    hubs = topN[:5]
    lines.append("- **Hubs (degree)**: "
                 + ", ".join(f"`{n}` ({deg[n]})" for n in hubs) + ".")
    auth = sorted(U, key=lambda n: -indeg.get(n, 0))[:5]
    lines.append("- **Authorities (in-degree — typically descriptors/"
                 "settings)**: "
                 + ", ".join(f"`{n}` ({indeg.get(n,0)})" for n in auth) + ".")
    if btw:
        pivots = sorted(U, key=lambda n: -btw.get(n, 0))[:5]
        lines.append("- **Mechanistic pivots (betweenness)**: "
                     + ", ".join(f"`{n}`" for n in pivots)
                     + " — nodes bridging causal chains to the descriptor "
                       "fan (e.g. intermediate processes).")
    isolated = [n for n in U if deg[n] <= 1]
    lines.append(f"- **Peripheral concepts (degree ≤1)**: {len(isolated)} — "
                 "each is either a semi-open entity or a canonicalization "
                 "fragment; cross-check against synonyms_report.csv.")
    lines.append(f"- **Components**: {len(comps)} "
                 f"(largest: {largest.number_of_nodes()} nodes). Components "
                 "beyond the main one indicate assertions disconnected from "
                 "the MTD core — inspection candidates.")
    (outdir / "graph_findings.md").write_text("\n".join(lines),
                                              encoding="utf-8")
    print(f"[12] graph: {U.number_of_nodes()} nodes / "
          f"{G.number_of_edges()} edges, {len(comps)} components — "
          f"outputs in {outdir}")


if __name__ == "__main__":
    main()