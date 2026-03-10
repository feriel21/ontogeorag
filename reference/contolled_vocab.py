import os
import json
import re
import pandas as pd
import networkx as nx

# =========================
# CONFIG
# =========================
XLSX = "reference/Table_Supplementary_1_V2.xlsx"
OUT = "schema_seed_output"
os.makedirs(OUT, exist_ok=True)


# =========================
# HELPERS
# =========================
def canon(s: str) -> str:
    """Canonicalize concept strings (stable IDs)."""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[“”\"']", "", s)
    s = re.sub(r"[^a-z0-9\s\-\(\)\.:/]", "", s)
    return s.strip()

def suggest_node_type(term: str) -> str:
    """
    Heuristic suggestion ONLY.
    You will later refine manually or with LLM+RAG.
    """
    t = term.lower()

    # Objects
    if "mtd" in t or "mass transport deposit" in t or "mtc" in t:
        return "Geological_Object"

    # Processes (trigger/transport/deposition) - keywords
    if any(k in t for k in [
        "trigger", "seismicity", "earthquake", "overpressure", "fluid migration",
        "remobilization", "erosion", "deposition", "compaction", "slump", "slide",
        "debris flow", "flow", "transport", "hydroplaning"
    ]):
        return "Process"

    # Environmental controls
    if any(k in t for k in [
        "slope", "sea level", "sedimentation rate", "topography", "confinement",
        "basin", "margin", "depth", "bathymetry"
    ]):
        return "Environmental_Control"

    # Descriptors / observables
    if any(k in t for k in [
        "facies", "chaotic", "transparent", "amplitude", "thickness", "runout",
        "headscarp", "basal surface", "thrust", "fold", "fault", "blocks",
        "geometry", "roughness"
    ]):
        return "Descriptor"

    # Default
    return "Descriptor"

def map_relation_type(edge_type: str) -> str:
    """
    Since the Excel doesn't encode verb semantics:
    - Directed -> AFFECTS (weak placeholder)
    - Undirected -> RELATED_TO
    """
    et = str(edge_type).strip().lower()
    if et == "directed":
        return "AFFECTS"
    return "RELATED_TO"

# =========================
# LOAD DATA
# =========================
df = pd.read_excel(XLSX)  # columns: Label, Type, Reference #, Comment

G = nx.DiGraph()
edges = []
excluded = []
concept_rows = []

# =========================
# BUILD GRAPH + VOCAB SEED
# =========================
for _, row in df.iterrows():

    label = str(row["Label"]).strip()
    edge_type = "" if pd.isna(row["Type"]) else str(row["Type"]).strip()
    comment = "" if pd.isna(row["Comment"]) else str(row["Comment"]).strip()
    raw_ref = row["Reference #"]

    if not label:
        continue

    if "undetermined edge" in comment.lower():
        excluded.append(label)
        continue

    if " - " not in label:
        excluded.append(label)
        continue

    src_raw, tgt_raw = [x.strip() for x in label.split(" - ", 1)]
    src = canon(src_raw)
    tgt = canon(tgt_raw)

    # references
    if pd.isna(raw_ref):
        ref_ids = []
    else:
        ref_ids = [int(x) for x in re.findall(r"\d+", str(raw_ref))]

    # nodes + suggested node types
    src_type = suggest_node_type(src_raw)
    tgt_type = suggest_node_type(tgt_raw)

    G.add_node(src, label=src_raw, node_type=src_type, source="LeBouteiller_TableS1")
    G.add_node(tgt, label=tgt_raw, node_type=tgt_type, source="LeBouteiller_TableS1")

    rel = map_relation_type(edge_type)

    # add edge
    G.add_edge(
        src,
        tgt,
        edge_type=edge_type,
        relation_type=rel,
        reference_ids=",".join(map(str, ref_ids)),
        comment=comment
    )

    edges.append({
        "source": src,
        "target": tgt,
        "source_label": src_raw,
        "target_label": tgt_raw,
        "edge_type": edge_type,
        "relation_type": rel,
        "reference_ids": ref_ids,
        "comment": comment
    })

# =========================
# EXPORT VOCAB (CONCEPT LIST)
# =========================
# compute degrees to prioritize validation
deg = dict(G.degree())

for n, data in G.nodes(data=True):
    concept_rows.append({
        "canonical_term": n,
        "original_label": data.get("label", ""),
        "suggested_node_type": data.get("node_type", "Descriptor"),
        "degree": deg.get(n, 0),
        "source": data.get("source", "LeBouteiller_TableS1"),
    })

concepts_df = pd.DataFrame(concept_rows).sort_values(["degree", "canonical_term"], ascending=[False, True])
concepts_df.to_csv(os.path.join(OUT, "concepts_seed.csv"), index=False)

# edges
pd.DataFrame(edges).to_csv(os.path.join(OUT, "edges_seed.csv"), index=False)

# graphml
nx.write_graphml(G, os.path.join(OUT, "graph_seed.graphml"))

# excluded
with open(os.path.join(OUT, "excluded_edges.txt"), "w", encoding="utf-8") as f:
    for e in excluded:
        f.write(e + "\n")

print("✅ Seed vocab + seed graph exported")
print(f"Nodes          : {G.number_of_nodes()}")
print(f"Edges          : {G.number_of_edges()}")
print(f"Excluded edges : {len(excluded)}")
print(f"Output dir     : {OUT}")
