#!/usr/bin/env python3
"""
19_causal_chains.py — Are physical mechanisms recoverable as PATHS?
===================================================================
WHY
    A knowledge graph can have every arc correct and still tell no coherent
    physical story. Geological mechanisms are sequences — preconditioning,
    trigger, failure, transport, deposition — so the sixth coherence control
    (§7.3 of the project notes) asks whether those sequences exist as PATHS
    in the graph, not merely as isolated arcs.

    Expectation is deliberately low, and that is the point: the failure-mode
    analysis already showed that `pore pressure controls slope failure` is
    stated across 84 chunks yet never extracted as a chain, and the relation
    census gives `controls` only a handful of triples. A thin result here
    converts an intuition ("sentence-level extraction misses mechanisms")
    into a measured limitation, which is the strongest way to report a
    weakness.

WHAT
    Builds the causal subgraph (triggers / causes / controls / affects, plus
    any relation given with --relations) and enumerates simple paths from
    *sources* (nodes that only cause, never are caused) to *sinks* (nodes
    that are only caused). No domain lexicon is hardcoded: sources and sinks
    are derived from the graph's own degree structure.

    For each chain it reports the WEAKEST LINK — a chain is only as reliable
    as its least reliable arc, so chain tier = max(edge tiers) and chain
    confidence = min(edge confidences). A 3-step chain built from Tier-2
    arcs is not a Tier-2 finding; it is weaker than any of its parts.

OUTPUTS (in --outdir)
    causal_chains.csv      one row per path: nodes, relations, length,
                           weakest tier, min confidence, papers involved
    causal_chains.md       readable listing grouped by length, ready to send
                           for expert review
    causal_chains_report.json  counts, length distribution, sources/sinks
    fig_causal_chains.png  chain-length distribution + arc census

USAGE
    python analysis_suite/19_causal_chains.py \
        --kg output/run13/analysis/kg_with_provenance.json \
        --outdir output/run13/analysis
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))
from kg_io import (BLUE, TERRACOTTA, apply_style, get_object, get_relation,
                   get_subject, load_kg)

DEFAULT_CAUSAL = ["triggers", "causes", "controls", "affects", "formedby",
                  "leads_to", "results_in"]


def norm_rel(r):
    return str(r or "").strip().lower().replace("_", "").replace(" ", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--outdir", default="output/run13/analysis")
    ap.add_argument("--relations", nargs="*", default=DEFAULT_CAUSAL,
                    help="relations forming the causal subgraph")
    ap.add_argument("--max-len", type=int, default=5,
                    help="maximum path length in arcs")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    kg = load_kg(args.kg)
    active = [t for t in kg["triples"] if t.get("_status") != "quarantine"]
    causal = {norm_rel(r) for r in args.relations}

    G = nx.DiGraph()
    arc_meta = {}
    rel_census = Counter()
    for t in active:
        s, r, o = get_subject(t), get_relation(t), get_object(t)
        if norm_rel(r) not in causal or not s or not o or s == o:
            continue
        rel_census[r] += 1
        # keep the most reliable arc if the same pair appears twice
        prev = arc_meta.get((s, o))
        tier = t.get("_tier", 2) or 2
        conf = float(t.get("confidence") or 0)
        if prev is None or tier < prev["tier"]:
            arc_meta[(s, o)] = {"relation": r, "tier": tier,
                                "confidence": conf,
                                "papers": t.get("paper_ids", [])}
            G.add_edge(s, o)

    # sources = only cause; sinks = only are caused (derived, not hardcoded)
    sources = [n for n in G if G.in_degree(n) == 0 and G.out_degree(n) > 0]
    sinks = [n for n in G if G.out_degree(n) == 0 and G.in_degree(n) > 0]

    chains = []
    for s in sources:
        for k in sinks:
            if s == k:
                continue
            try:
                for path in nx.all_simple_paths(G, s, k,
                                                cutoff=args.max_len):
                    if len(path) < 3:      # < 2 arcs is not a chain
                        continue
                    arcs = [arc_meta[(path[i], path[i + 1])]
                            for i in range(len(path) - 1)]
                    papers = set()
                    for a in arcs:
                        papers.update(a["papers"])
                    chains.append({
                        "n_arcs": len(arcs),
                        "path": " -> ".join(path),
                        "relations": " | ".join(a["relation"] for a in arcs),
                        "weakest_tier": max(a["tier"] for a in arcs),
                        "min_confidence": round(
                            min(a["confidence"] for a in arcs), 4),
                        "all_tier1": all(a["tier"] == 1 for a in arcs),
                        "n_papers_involved": len(papers),
                        "start": path[0], "end": path[-1],
                    })
            except nx.NetworkXNoPath:
                continue

    chains.sort(key=lambda c: (-c["n_arcs"], c["weakest_tier"],
                              -c["min_confidence"]))

    # ── outputs ───────────────────────────────────────────────────────
    if chains:
        with open(outdir / "causal_chains.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(chains[0].keys()))
            w.writeheader()
            w.writerows(chains)

    by_len = defaultdict(list)
    for c in chains:
        by_len[c["n_arcs"]].append(c)

    lines = ["# Causal chains in the knowledge graph", "",
             f"Causal subgraph: {G.number_of_nodes()} nodes, "
             f"{G.number_of_edges()} arcs "
             f"({', '.join(f'{r}={n}' for r, n in rel_census.most_common())})",
             f"Sources (never caused): {len(sources)} · "
             f"Sinks (never cause): {len(sinks)}",
             f"Multi-arc chains found: **{len(chains)}**", "",
             "A chain is only as reliable as its weakest arc: chain tier = "
             "max(arc tiers), chain confidence = min(arc confidences). A "
             "three-step chain of Tier-2 arcs is weaker than any single one "
             "of them.", ""]
    for k in sorted(by_len, reverse=True):
        lines.append(f"## Chains of {k} arcs ({len(by_len[k])})")
        for c in by_len[k][:40]:
            flag = " ✓all-Tier-1" if c["all_tier1"] else ""
            lines.append(f"- `{c['path']}`  \n"
                         f"  relations: {c['relations']} · weakest tier "
                         f"T{c['weakest_tier']} · min conf "
                         f"{c['min_confidence']}{flag}")
        if len(by_len[k]) > 40:
            lines.append(f"- … and {len(by_len[k]) - 40} more (see CSV)")
        lines.append("")
    if not chains:
        lines += ["## No multi-arc chain found", "",
                  "Every causal assertion in the graph is an isolated arc: "
                  "the corpus states mechanisms across sentences, and "
                  "sentence-level extraction cannot compose them. This is a "
                  "measured limitation of the extraction granularity, not "
                  "evidence that the mechanisms are absent from the "
                  "literature."]
    (outdir / "causal_chains.md").write_text("\n".join(lines),
                                             encoding="utf-8")

    report = {
        "causal_relations": sorted(causal),
        "relation_census": dict(rel_census),
        "subgraph_nodes": G.number_of_nodes(),
        "subgraph_arcs": G.number_of_edges(),
        "n_sources": len(sources), "n_sinks": len(sinks),
        "sources": sorted(sources)[:40], "sinks": sorted(sinks)[:40],
        "n_chains": len(chains),
        "chain_length_distribution": {str(k): len(v)
                                      for k, v in sorted(by_len.items())},
        "n_all_tier1_chains": sum(1 for c in chains if c["all_tier1"]),
        "longest_chain": chains[0]["path"] if chains else None,
        "interpretation": (
            "Chains are enumerated between structurally derived sources "
            "(never caused) and sinks (never causing); no domain lexicon is "
            "used. A low count is expected and is reported as a limitation "
            "of sentence-level extraction, not as absence of the mechanism "
            "in the literature."),
    }
    with open(outdir / "causal_chains_report.json", "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── figure ────────────────────────────────────────────────────────
    try:
        plt = apply_style()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        ax = axes[0]
        if by_len:
            ks = sorted(by_len)
            ax.bar([str(k) for k in ks], [len(by_len[k]) for k in ks],
                   color=BLUE)
            for i, k in enumerate(ks):
                ax.text(i, len(by_len[k]), str(len(by_len[k])),
                        ha="center", va="bottom", fontsize=10)
        else:
            ax.text(0.5, 0.5, "no multi-arc chain", ha="center",
                    va="center", transform=ax.transAxes, fontsize=13)
        ax.set_xlabel("arcs per chain")
        ax.set_ylabel("chains")
        ax.set_title("A. Causal chain lengths", fontsize=15, loc="left")

        ax = axes[1]
        rels = [r for r, _ in rel_census.most_common()]
        ax.bar(rels, [rel_census[r] for r in rels], color=TERRACOTTA)
        ax.set_ylabel("arcs")
        ax.set_title("B. Causal arcs by relation", fontsize=15, loc="left")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(outdir / "fig_causal_chains.pdf")
        fig.savefig(outdir / "fig_causal_chains.png", dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] figure skipped: {e}")

    print("=" * 62)
    print("CAUSAL CHAIN COHERENCE CHECK")
    print("=" * 62)
    print(f"causal subgraph      : {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} arcs")
    print(f"relations            : {dict(rel_census)}")
    print(f"sources / sinks      : {len(sources)} / {len(sinks)}")
    print(f"multi-arc chains     : {len(chains)}")
    if by_len:
        print(f"length distribution  : "
              f"{ {k: len(v) for k, v in sorted(by_len.items())} }")
        print(f"all-Tier-1 chains    : {report['n_all_tier1_chains']}")
        print(f"longest              : {chains[0]['path']}")
    else:
        print("  -> every causal assertion is an isolated arc; report as a "
              "limitation of sentence-level extraction")
    print(f"\noutputs in: {outdir}")


if __name__ == "__main__":
    main()