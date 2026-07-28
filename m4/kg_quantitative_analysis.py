#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KG Quantitative Analysis — OntoGeoRAG
======================================
Self-contained script producing three deliverables from the final KG:

  1. KG Portrait (for figure C3 and §3 of the manuscript)
     - Node count by entity type
     - Edge count by relation type
     - Tier distribution per relation
     - Degree distribution (in/out/total)
     - Support-count distribution
     - Top-10 highest-degree nodes

  2. MTD Ego-Subgraph (for figure C5 vignette)
     - All triples within 2 hops of "mass transport deposit"
     - Exported as edge list + GraphML for networkx

  3. Duplicate / Near-Duplicate Entity Scan (for limitations section)
     - Pairwise cosine similarity of all unique entity names
     - Flags pairs above threshold (default 0.85)
     - Cross-references with known duplicate pairs

Usage (on cluster):
  python kg_quantitative_analysis.py --kg ~/ontogeorag/output/kg_final/tiered_kg_m4.json
  python kg_quantitative_analysis.py --kg ~/ontogeorag/output/kg_final/tiered_kg_m4.json --no-embeddings

Output: ./kg_analysis/ directory with tables (CSV), stats (JSON), subgraph (GraphML), figures (PDF+PNG).
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ─── Lazy imports for optional dependencies ───────────────────────────
def _import_networkx():
    try:
        import networkx as nx
        return nx
    except ImportError:
        print("ERROR: networkx not installed. pip install networkx --break-system-packages")
        sys.exit(1)

def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
KNOWN_DUPLICATES = [
    (15, 16), (38, 39), (40, 41), (90, 91), (110, 111), (112, 115),
]

MTD_CANONICAL = "mass transport deposit"

# Okabe-Ito palette (colourblind safe)
OI = {
    "orange":    "#E69F00",
    "skyblue":   "#56B4E9",
    "green":     "#009E73",
    "yellow":    "#F0E442",
    "blue":      "#0072B2",
    "vermilion": "#D55E00",
    "purple":    "#CC79A7",
    "black":     "#000000",
    "grey":      "#999999",
}

TYPE_COLORS = {
    "SeismicObject": OI["blue"],
    "Process":       OI["orange"],
    "Descriptor":    OI["green"],
    "Setting":       OI["purple"],
    "Property":      OI["vermilion"],
    "":              OI["grey"],
}

TIER_COLORS = {
    1: OI["blue"],
    2: OI["orange"],
    "quarantine": OI["vermilion"],
}


# ═══════════════════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════════════════
def load_kg(path: str) -> list[dict]:
    """Load KG JSON, handling both dict-with-triples and list-of-pairs formats."""
    raw = json.loads(Path(path).read_text())

    # dict format: {"metadata": {...}, "triples": [...]}
    if isinstance(raw, dict):
        if "triples" in raw:
            return raw["triples"]
        # list-of-pairs: dict(json.load(...)) then access 'triples'
        return list(raw.values()) if not any(isinstance(v, list) for v in raw.values()) else raw.get("triples", [])

    # list-of-pairs format: [[key, value], ...]
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
        d = dict(raw)
        return d.get("triples", [])

    # direct list of triples
    if isinstance(raw, list):
        return raw

    raise ValueError(f"Unrecognized KG format in {path}")


def norm(s: str) -> str:
    """Lowercase strip normalisation for entity matching."""
    import re
    return re.sub(r"\s+", " ", (s or "").lower().strip()).rstrip(".,;:")


# ═══════════════════════════════════════════════════════════════════════
# 1. KG PORTRAIT
# ═══════════════════════════════════════════════════════════════════════
def kg_portrait(triples: list[dict], outdir: Path):
    """Compute and export all KG portrait statistics."""
    print("\n" + "=" * 65)
    print("1. KG PORTRAIT")
    print("=" * 65)

    # ── Filter out quarantined if flagged ──────────────────────────────
    active = [t for t in triples if not t.get("quarantine", False)]
    quarantined = [t for t in triples if t.get("quarantine", False)]
    print(f"  Active triples:      {len(active)}")
    print(f"  Quarantined triples: {len(quarantined)}")

    # ── Tier distribution ─────────────────────────────────────────────
    tier_counts = Counter(t.get("tier", "?") for t in active)
    print(f"\n  Tier distribution (active):")
    for tier in sorted(tier_counts):
        print(f"    Tier {tier}: {tier_counts[tier]}")

    # ── Nodes by entity type ──────────────────────────────────────────
    nodes = {}  # name -> type
    for t in active:
        s, o = norm(t["subject"]), norm(t["object"])
        st = t.get("subject_type", "")
        ot = t.get("object_type", "")
        if s and s not in nodes:
            nodes[s] = st
        if o and o not in nodes:
            nodes[o] = ot

    type_counts = Counter(nodes.values())
    print(f"\n  Unique nodes: {len(nodes)}")
    print(f"  Nodes by entity type:")
    for tp, cnt in type_counts.most_common():
        label = tp if tp else "(untyped)"
        print(f"    {label:20s}: {cnt}")

    # Export node table
    node_csv = outdir / "nodes_by_type.csv"
    with open(node_csv, "w") as f:
        f.write("entity,entity_type\n")
        for name, tp in sorted(nodes.items()):
            f.write(f'"{name}","{tp}"\n')
    print(f"  -> {node_csv}")

    # ── Edges by relation ─────────────────────────────────────────────
    rel_counts = Counter(t.get("relation", "?") for t in active)
    print(f"\n  Edges by relation:")
    for rel, cnt in rel_counts.most_common():
        print(f"    {rel:22s}: {cnt}")

    # ── Tier × Relation cross-tab ─────────────────────────────────────
    tier_rel = defaultdict(lambda: defaultdict(int))
    for t in active:
        tier_rel[t.get("relation", "?")][t.get("tier", "?")] += 1

    print(f"\n  Tier distribution per relation:")
    print(f"    {'Relation':22s}  {'T1':>4s}  {'T2':>4s}  {'Other':>5s}")
    print(f"    {'-'*22}  {'----':>4s}  {'----':>4s}  {'-----':>5s}")
    rows_tier_rel = []
    for rel in sorted(tier_rel.keys()):
        t1 = tier_rel[rel].get(1, 0)
        t2 = tier_rel[rel].get(2, 0)
        other = sum(v for k, v in tier_rel[rel].items() if k not in (1, 2))
        print(f"    {rel:22s}  {t1:4d}  {t2:4d}  {other:5d}")
        rows_tier_rel.append((rel, t1, t2, other))

    tier_rel_csv = outdir / "tier_by_relation.csv"
    with open(tier_rel_csv, "w") as f:
        f.write("relation,tier1,tier2,other\n")
        for rel, t1, t2, other in rows_tier_rel:
            f.write(f'"{rel}",{t1},{t2},{other}\n')
    print(f"  -> {tier_rel_csv}")

    # ── Degree distribution ───────────────────────────────────────────
    in_deg = Counter()
    out_deg = Counter()
    for t in active:
        s, o = norm(t["subject"]), norm(t["object"])
        out_deg[s] += 1
        in_deg[o] += 1

    all_nodes_set = set(in_deg.keys()) | set(out_deg.keys())
    total_deg = {n: in_deg.get(n, 0) + out_deg.get(n, 0) for n in all_nodes_set}

    print(f"\n  Degree statistics:")
    degs = list(total_deg.values())
    print(f"    Mean:   {sum(degs)/len(degs):.2f}")
    print(f"    Max:    {max(degs)}")
    print(f"    Median: {sorted(degs)[len(degs)//2]}")

    # Top-10 by total degree
    top10 = sorted(total_deg.items(), key=lambda x: -x[1])[:10]
    print(f"\n  Top-10 highest-degree nodes:")
    print(f"    {'Node':40s}  {'Type':18s}  {'In':>3s}  {'Out':>3s}  {'Tot':>3s}")
    print(f"    {'-'*40}  {'-'*18}  {'---':>3s}  {'---':>3s}  {'---':>3s}")
    top10_rows = []
    for name, deg in top10:
        tp = nodes.get(name, "")
        ind = in_deg.get(name, 0)
        outd = out_deg.get(name, 0)
        print(f"    {name:40s}  {tp:18s}  {ind:3d}  {outd:3d}  {deg:3d}")
        top10_rows.append((name, tp, ind, outd, deg))

    top10_csv = outdir / "top10_degree.csv"
    with open(top10_csv, "w") as f:
        f.write("entity,entity_type,in_degree,out_degree,total_degree\n")
        for name, tp, ind, outd, deg in top10_rows:
            f.write(f'"{name}","{tp}",{ind},{outd},{deg}\n')
    print(f"  -> {top10_csv}")

    # ── Support-count distribution ────────────────────────────────────
    sup_counts = [t.get("support_count", 0) for t in active]
    sup_dist = Counter(sup_counts)
    print(f"\n  Support-count distribution:")
    for sc in sorted(sup_dist.keys()):
        print(f"    support_count={sc}: {sup_dist[sc]} triples")

    # ── Summary JSON ──────────────────────────────────────────────────
    summary = {
        "total_triples": len(triples),
        "active_triples": len(active),
        "quarantined_triples": len(quarantined),
        "unique_nodes": len(nodes),
        "tier_distribution": {str(k): v for k, v in sorted(tier_counts.items())},
        "nodes_by_type": dict(type_counts.most_common()),
        "edges_by_relation": dict(rel_counts.most_common()),
        "tier_by_relation": {rel: dict(tiers) for rel, tiers in tier_rel.items()},
        "degree_stats": {
            "mean": round(sum(degs) / len(degs), 2),
            "max": max(degs),
            "median": sorted(degs)[len(degs) // 2],
        },
        "support_count_distribution": {str(k): v for k, v in sorted(sup_dist.items())},
        "top10_nodes": [
            {"entity": n, "type": tp, "in": ind, "out": outd, "total": deg}
            for n, tp, ind, outd, deg in top10_rows
        ],
    }
    summary_json = outdir / "kg_portrait.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  -> {summary_json}")

    # ── Figures ───────────────────────────────────────────────────────
    _plot_portrait(active, nodes, type_counts, rel_counts, tier_rel,
                   total_deg, sup_dist, outdir)

    return active, nodes, total_deg


def _plot_portrait(active, nodes, type_counts, rel_counts, tier_rel,
                   total_deg, sup_dist, outdir):
    """Generate publication-quality portrait figures."""
    plt = _import_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("OntoGeoRAG — Knowledge Graph Portrait", fontsize=14, fontweight="bold")

    # (a) Nodes by entity type
    ax = axes[0, 0]
    types = list(type_counts.keys())
    counts = [type_counts[t] for t in types]
    colors = [TYPE_COLORS.get(t, OI["grey"]) for t in types]
    labels = [t if t else "(untyped)" for t in types]
    ax.barh(labels, counts, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xlabel("Count")
    ax.set_title("(a) Nodes by entity type", fontweight="bold")
    ax.invert_yaxis()
    for i, v in enumerate(counts):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=9)

    # (b) Edges by relation (stacked T1/T2)
    ax = axes[0, 1]
    rels_sorted = sorted(tier_rel.keys(), key=lambda r: sum(tier_rel[r].values()), reverse=True)
    t1_vals = [tier_rel[r].get(1, 0) for r in rels_sorted]
    t2_vals = [tier_rel[r].get(2, 0) for r in rels_sorted]
    y_pos = range(len(rels_sorted))
    ax.barh(y_pos, t1_vals, color=TIER_COLORS[1], edgecolor="black",
            linewidth=0.8, label="Tier-1")
    ax.barh(y_pos, t2_vals, left=t1_vals, color=TIER_COLORS[2],
            edgecolor="black", linewidth=0.8, label="Tier-2")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(rels_sorted, fontsize=8)
    ax.set_xlabel("Count")
    ax.set_title("(b) Edges by relation (stacked by tier)", fontweight="bold")
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower right")

    # (c) Degree distribution histogram
    ax = axes[1, 0]
    degs = list(total_deg.values())
    max_deg = max(degs)
    bins = range(1, max_deg + 2)
    ax.hist(degs, bins=bins, color=OI["skyblue"], edgecolor="black", linewidth=0.8,
            align="left")
    ax.set_xlabel("Total degree")
    ax.set_ylabel("Number of nodes")
    ax.set_title("(c) Node degree distribution", fontweight="bold")
    ax.set_xticks(range(1, max_deg + 1, max(1, max_deg // 10)))

    # (d) Support-count distribution
    ax = axes[1, 1]
    sc_keys = sorted(sup_dist.keys())
    sc_vals = [sup_dist[k] for k in sc_keys]
    ax.bar([str(k) for k in sc_keys], sc_vals, color=OI["green"],
           edgecolor="black", linewidth=0.8)
    ax.set_xlabel("Support count (# source papers)")
    ax.set_ylabel("Number of triples")
    ax.set_title("(d) Evidence breadth distribution", fontweight="bold")
    for i, v in enumerate(sc_vals):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=8)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fpath = outdir / f"kg_portrait.{ext}"
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outdir}/kg_portrait.pdf|png")


# ═══════════════════════════════════════════════════════════════════════
# 2. MTD EGO-SUBGRAPH
# ═══════════════════════════════════════════════════════════════════════
def mtd_ego_subgraph(triples: list[dict], outdir: Path):
    """Extract 2-hop ego-subgraph around 'mass transport deposit'."""
    print("\n" + "=" * 65)
    print("2. MTD EGO-SUBGRAPH (2-hop)")
    print("=" * 65)

    nx = _import_networkx()

    # Build full graph
    active = [t for t in triples if not t.get("quarantine", False)]
    G = nx.DiGraph()
    for t in active:
        s, o = norm(t["subject"]), norm(t["object"])
        G.add_node(s, entity_type=t.get("subject_type", ""))
        G.add_node(o, entity_type=t.get("object_type", ""))
        G.add_edge(s, o,
                   relation=t.get("relation", ""),
                   tier=t.get("tier", "?"),
                   support_count=t.get("support_count", 0))

    # Find MTD node (fuzzy match)
    mtd_node = None
    for n in G.nodes():
        if "mass transport deposit" in n or n in ("mtd", "mass-transport deposit"):
            mtd_node = n
            break

    if mtd_node is None:
        print("  WARNING: 'mass transport deposit' node not found in graph!")
        print(f"  Available nodes containing 'mass': "
              f"{[n for n in G.nodes() if 'mass' in n]}")
        return

    print(f"  MTD node: '{mtd_node}' (degree {G.degree(mtd_node)})")

    # BFS 2-hop on undirected view
    G_undirected = G.to_undirected()
    hop1 = set(G_undirected.neighbors(mtd_node))
    hop2 = set()
    for n in hop1:
        hop2.update(G_undirected.neighbors(n))
    ego_nodes = {mtd_node} | hop1 | hop2

    # Extract subgraph (directed)
    sub = G.subgraph(ego_nodes).copy()
    print(f"  Hop-1 neighbors: {len(hop1)}")
    print(f"  Hop-2 neighbors: {len(hop2 - hop1 - {mtd_node})}")
    print(f"  Subgraph: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges")

    # Export edge list CSV
    edge_csv = outdir / "mtd_ego_edges.csv"
    with open(edge_csv, "w") as f:
        f.write("subject,relation,object,tier,support_count,hop_from_mtd\n")
        for s, o, data in sub.edges(data=True):
            # Determine hop distance of the edge
            if s == mtd_node or o == mtd_node:
                hop = 1
            elif s in hop1 or o in hop1:
                hop = 2
            else:
                hop = 3  # shouldn't happen with 2-hop
            f.write(f'"{s}","{data.get("relation", "")}","{o}",'
                    f'{data.get("tier", "?")},{data.get("support_count", 0)},{hop}\n')
    print(f"  -> {edge_csv}")

    # Export node list with hop distance
    node_csv = outdir / "mtd_ego_nodes.csv"
    with open(node_csv, "w") as f:
        f.write("entity,entity_type,hop_from_mtd,degree_in_subgraph\n")
        for n in sorted(sub.nodes()):
            if n == mtd_node:
                hop = 0
            elif n in hop1:
                hop = 1
            else:
                hop = 2
            tp = sub.nodes[n].get("entity_type", "")
            deg = sub.degree(n)
            f.write(f'"{n}","{tp}",{hop},{deg}\n')
    print(f"  -> {node_csv}")

    # Export GraphML for external visualization
    graphml_path = outdir / "mtd_ego_subgraph.graphml"
    # Add hop attribute to nodes for coloring
    for n in sub.nodes():
        if n == mtd_node:
            sub.nodes[n]["hop"] = 0
        elif n in hop1:
            sub.nodes[n]["hop"] = 1
        else:
            sub.nodes[n]["hop"] = 2
    nx.write_graphml(sub, str(graphml_path))
    print(f"  -> {graphml_path}")

    # Print subgraph triples for quick inspection
    print(f"\n  Subgraph triples:")
    for s, o, data in sorted(sub.edges(data=True), key=lambda x: x[2].get("tier", 99)):
        rel = data.get("relation", "?")
        tier = data.get("tier", "?")
        print(f"    T{tier}  ({s}, {rel}, {o})")

    # Summary JSON
    ego_summary = {
        "mtd_node": mtd_node,
        "mtd_degree": G.degree(mtd_node),
        "hop1_count": len(hop1),
        "hop2_count": len(hop2 - hop1 - {mtd_node}),
        "subgraph_nodes": sub.number_of_nodes(),
        "subgraph_edges": sub.number_of_edges(),
        "edges": [
            {"subject": s, "relation": data.get("relation", ""),
             "object": o, "tier": data.get("tier", "?")}
            for s, o, data in sub.edges(data=True)
        ],
    }
    (outdir / "mtd_ego_summary.json").write_text(
        json.dumps(ego_summary, indent=2, ensure_ascii=False))
    print(f"  -> {outdir}/mtd_ego_summary.json")

    return sub


# ═══════════════════════════════════════════════════════════════════════
# 3. DUPLICATE / NEAR-DUPLICATE ENTITY SCAN
# ═══════════════════════════════════════════════════════════════════════
def duplicate_scan(triples: list[dict], outdir: Path,
                   model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                   threshold: float = 0.85,
                   use_embeddings: bool = True):
    """Scan all entity names for near-duplicates via embedding cosine similarity."""
    print("\n" + "=" * 65)
    print("3. DUPLICATE / NEAR-DUPLICATE ENTITY SCAN")
    print("=" * 65)

    # Collect all unique entities
    active = [t for t in triples if not t.get("quarantine", False)]
    entities = set()
    entity_info = {}  # name -> {type, indices}
    for i, t in enumerate(active):
        s, o = norm(t["subject"]), norm(t["object"])
        if s:
            entities.add(s)
            entity_info.setdefault(s, {"type": t.get("subject_type", ""), "indices": []})
            entity_info[s]["indices"].append(i)
        if o:
            entities.add(o)
            entity_info.setdefault(o, {"type": t.get("object_type", ""), "indices": []})
            entity_info[o]["indices"].append(i)

    entities = sorted(entities)
    print(f"  Unique entities: {len(entities)}")

    # ── Known duplicates check ────────────────────────────────────────
    # Map index-based known pairs to entity names where possible
    print(f"\n  Known duplicate pairs (by KG index): {KNOWN_DUPLICATES}")
    print(f"  (These are triple indices from run11; entity names must be")
    print(f"   verified from the actual KG file.)")

    if not use_embeddings:
        print("\n  Skipping embedding-based scan (--no-embeddings).")
        # Fall back to simple string overlap heuristics
        _string_heuristic_scan(entities, entity_info, outdir, threshold=0.8)
        return

    # ── Embedding-based scan ──────────────────────────────────────────
    print(f"\n  Loading model: {model_name}")
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("  ERROR: sentence-transformers not installed.")
        print("  Falling back to string heuristic scan.")
        _string_heuristic_scan(entities, entity_info, outdir, threshold=0.8)
        return

    model = SentenceTransformer(model_name)
    print(f"  Encoding {len(entities)} entity names...")
    embeddings = model.encode(entities, show_progress_bar=False,
                              convert_to_numpy=True, normalize_embeddings=True)

    # Pairwise cosine (embeddings are L2-normalized, so dot product = cosine)
    sim_matrix = embeddings @ embeddings.T

    # Find pairs above threshold (upper triangle only)
    pairs = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            cos = float(sim_matrix[i, j])
            if cos >= threshold:
                pairs.append((entities[i], entities[j], cos))

    pairs.sort(key=lambda x: -x[2])

    print(f"\n  Pairs with cosine >= {threshold}: {len(pairs)}")
    if pairs:
        print(f"    {'Entity A':35s}  {'Entity B':35s}  {'Cosine':>6s}  {'Types':>12s}")
        print(f"    {'-'*35}  {'-'*35}  {'------':>6s}  {'-'*12}")
        for a, b, cos in pairs:
            ta = entity_info.get(a, {}).get("type", "")
            tb = entity_info.get(b, {}).get("type", "")
            types = f"{ta}/{tb}" if ta != tb else ta
            print(f"    {a:35s}  {b:35s}  {cos:.4f}  {types}")

    # Export
    dup_csv = outdir / "duplicate_candidates.csv"
    with open(dup_csv, "w") as f:
        f.write("entity_a,entity_b,cosine,type_a,type_b\n")
        for a, b, cos in pairs:
            ta = entity_info.get(a, {}).get("type", "")
            tb = entity_info.get(b, {}).get("type", "")
            f.write(f'"{a}","{b}",{cos:.4f},"{ta}","{tb}"\n')
    print(f"  -> {dup_csv}")

    # Also export full similarity matrix for entities with any high-sim pair
    flagged = set()
    for a, b, _ in pairs:
        flagged.add(a)
        flagged.add(b)

    dup_summary = {
        "model": model_name,
        "threshold": threshold,
        "total_entities": len(entities),
        "pairs_above_threshold": len(pairs),
        "flagged_entities": sorted(flagged),
        "pairs": [
            {"entity_a": a, "entity_b": b, "cosine": round(cos, 4)}
            for a, b, cos in pairs
        ],
    }
    (outdir / "duplicate_scan.json").write_text(
        json.dumps(dup_summary, indent=2, ensure_ascii=False))
    print(f"  -> {outdir}/duplicate_scan.json")

    return pairs


def _string_heuristic_scan(entities, entity_info, outdir, threshold=0.8):
    """Fallback: flag entities sharing long common substrings or edit distance."""
    print(f"\n  Running string-overlap heuristic (no GPU needed)...")

    pairs = []
    for i, a in enumerate(entities):
        for j in range(i + 1, len(entities)):
            b = entities[j]
            # Simple: Jaccard on word tokens
            wa = set(a.split())
            wb = set(b.split())
            if not wa or not wb:
                continue
            jaccard = len(wa & wb) / len(wa | wb)
            # Also check containment
            containment = max(len(wa & wb) / len(wa), len(wa & wb) / len(wb))
            score = max(jaccard, containment)
            if score >= threshold:
                pairs.append((a, b, score))

    pairs.sort(key=lambda x: -x[2])

    print(f"  Pairs with word-overlap >= {threshold}: {len(pairs)}")
    for a, b, sc in pairs:
        ta = entity_info.get(a, {}).get("type", "")
        tb = entity_info.get(b, {}).get("type", "")
        print(f"    {a:35s}  {b:35s}  {sc:.3f}  {ta}/{tb}")

    dup_csv = outdir / "duplicate_candidates_heuristic.csv"
    with open(dup_csv, "w") as f:
        f.write("entity_a,entity_b,word_overlap,type_a,type_b\n")
        for a, b, sc in pairs:
            ta = entity_info.get(a, {}).get("type", "")
            tb = entity_info.get(b, {}).get("type", "")
            f.write(f'"{a}","{b}",{sc:.4f},"{ta}","{tb}"\n')
    print(f"  -> {dup_csv}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="KG Quantitative Analysis — OntoGeoRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--kg", required=True,
                        help="Path to final KG JSON (e.g. tiered_kg_m4.json)")
    parser.add_argument("--outdir", default="./kg_analysis",
                        help="Output directory (default: ./kg_analysis)")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-transformer model for duplicate scan")
    parser.add_argument("--dup-threshold", type=float, default=0.85,
                        help="Cosine threshold for duplicate flagging (default: 0.85)")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip embedding-based duplicate scan (use string heuristic)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("KG QUANTITATIVE ANALYSIS — OntoGeoRAG")
    print("=" * 65)
    print(f"  KG file:    {args.kg}")
    print(f"  Output dir: {outdir}")

    triples = load_kg(args.kg)
    print(f"  Loaded: {len(triples)} triples")

    # 1. Portrait
    active, nodes, total_deg = kg_portrait(triples, outdir)

    # 2. MTD ego-subgraph
    mtd_ego_subgraph(triples, outdir)

    # 3. Duplicate scan
    duplicate_scan(triples, outdir,
                   model_name=args.model,
                   threshold=args.dup_threshold,
                   use_embeddings=not args.no_embeddings)

    print("\n" + "=" * 65)
    print("DONE — all outputs in:", outdir)
    print("=" * 65)


if __name__ == "__main__":
    main()