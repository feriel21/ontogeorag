#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BUILD REFERENCE KNOWLEDGE GRAPH (STRICT – Le Bouteiller compliant, phase-aware)
==============================================================================

Goal
----
Create a machine-readable formalization of Le Bouteiller's relation map where:

Environmental Controls (Green)
    -> influence Mass-Transport Event (MTE) Properties / Mass Movement Properties (Orange; phase-aware)
        -> generate observable MTD Descriptors (Blue; seismic signatures)

STRICT RULES
------------
- Nodes come ONLY from EDGES_TABLE (source of truth)
- Descriptor categories come ONLY from Edges_OrganizedByDescriptor (when available)
- Process/Control categories come from EXPLICIT controlled mappings (no inference)
- Edge direction is read EXACTLY from Excel column 'Type'
- No "transversal warning" logic: transversal processes are valid high-degree nodes
- Lexical normalization only (controlled renaming to match thesis terminology)
"""

import pandas as pd
import json
import os
from collections import Counter

# ==================================================
# CONFIG
# ==================================================
EXCEL_FILE = "reference/Table_Supplementary_1_V2.xlsx"
SHEET_EDGES = "EDGES_TABLE"
SHEET_REFS  = "References"
SHEET_DESC  = "Edges_OrganizedByDescriptor"
OUTPUT_FILENAME = "reference/reference_kg.json"

# ==================================================
# HIERARCHY (CONCEPTUAL HEADER)
# ==================================================
HIERARCHY = {
    "MTD Descriptor (Blue)": [
        "Morphology",
        "Basal surface",
        "Upper surface",
        "Internal facies distributions",
        "Headscarp",
        "Position",
        "Global environment"
    ],
    "Mass Movement Property (Orange)": [
        "Trigger phase",
        "Transport phase",
        "Post-deposition phase"
    ],
    "Environmental Control (Green)": [
        "Environmental controls"
    ]
}

SUBCAT_TO_MAIN = {
    sub: main for main, subs in HIERARCHY.items() for sub in subs
}

# ==================================================
# TERMINOLOGY NORMALIZATION (CONTROLLED, LEXICAL ONLY)
# ==================================================
# Renommages contrôlés : permettent de coller aux noms canoniques de la thèse
CONTROLLED_RENAMES = {
    "MTD volume": "volume",
    "MTD principal direction": "principal direction",
    "max. horizontal length": "maximum horizontal length",
    "max. horizontal length, if attached": "maximum horizontal length",
    # Ajouts pour concepts manquants (évite les fautes de syntaxe)
    "BS flat sub-horizontal zone": "BS flat sub-horizontal zone",
    "deformed facies distribution": "deformed facies distribution",
    "faulty facies distribution": "faulty facies distribution",
    "multiple terracing downslope": "multiple terracing downslope",
    "presence of 'tongues' at toe": "presence of 'tongues' at toe",
    "ridged facies distribution": "ridged facies distribution",
    "transparent facies distribution": "transparent facies distribution",
    "terminal dispersion": "terminal dispersion",
}

# Dictionnaire de synonymes : associe les variantes courtes à leur forme canonique
SYNONYM_MAP = {
    "transparent facies": "transparent facies distribution",
    "ridged facies": "ridged facies distribution",
    "faulty facies": "faulty facies distribution",
    "deformed facies": "deformed facies distribution",
    "terracing downslope": "multiple terracing downslope",
    "tongues at toe": "presence of 'tongues' at toe",
}

def normalize_concept_name(name: str) -> str:
    """
    Lexical normalization only (no semantic inference):
    - trim
    - remove conditional suffixes after comma
    - remove trailing dot
    - collapse spaces
    - apply controlled renames to match thesis terminology
    - map short synonyms to canonical forms (SYNONYM_MAP)
    """
    if not isinstance(name, str):
        return name

    # Retrait des espaces superflus
    name = name.strip()
    # Retirer les suffixes conditionnels « , if attached »
    name = name.split(",")[0].strip()
    # Retirer la ponctuation en fin de chaîne
    name = name.rstrip(".")
    # Compactage des espaces multiples en un seul
    name = " ".join(name.split())

    # D’abord appliquer les renommages contrôlés
    name = CONTROLLED_RENAMES.get(name, name)
    # Puis appliquer une éventuelle correspondance de synonymes (en minuscules)
    candidate = SYNONYM_MAP.get(name.lower(), None)
    if candidate:
        name = candidate

    return name

# ==================================================
# PHASE-SPECIFIC METADATA ALIGNMENT (EXPLICIT MAPPING)
# ==================================================
PROCESS_TO_PHASE = {
    # Trigger phase
    "seismicity": "Trigger phase",
    "seismicity or waves": "Trigger phase",
    "inducing seismicity": "Trigger phase",
    "gravity": "Trigger phase",
    "fluid overpressure": "Trigger phase",
    "overpressure": "Trigger phase",
    "pore pressure increase": "Trigger phase",
    "pore pressure increase by compression": "Trigger phase",
    "pore pressure increase by fluid migration": "Trigger phase",
    # Transport phase
    "flow behavior": "Transport phase",
    "fluidization": "Transport phase",
    "erosion": "Transport phase",
    "plowing effect on underlying material": "Transport phase",
    "loss of mass": "Transport phase",
    "remobilization": "Transport phase",
    # Post-deposition phase
    "compaction during burial": "Post-deposition phase",
    "posterior fluid migrations": "Post-deposition phase",
    "terminal dispersion": "Post-deposition phase",
    "post deposition regional deformation": "Post-deposition phase",
}

# Descriptions facultatives pour les processus clés
PROCESS_DESCRIPTIONS = {
    "flow behavior": "Transversal causal control impacting multiple MTD surfaces and facies.",
    "fluidization": "Process influencing internal facies organization and seismic transparency/chaos.",
    "compaction during burial": "Post-depositional process affecting seismic expression and thickness evolution.",
    "posterior fluid migrations": "Post-depositional process potentially modifying seismic attributes and facies expression.",
    "terminal dispersion": "Late-stage process related to final spreading / distal evolution.",
}

# ==================================================
# ENVIRONMENTAL CONTROLS (EXPLICIT LIST / CONTROLLED)
# ==================================================
FORCE_ENV_CONTROL = {
    "sea level evolution",
    "sedimentation rate and type",
    "basin depocenter position",
    "existing geomorphology",
    "topography confinement downwards",
    "subsidence/uplift, extension/compression",
    "frontal compression",
    "chemical effects",
}

# ==================================================
# LOAD EXCEL
# ==================================================
print(f"[REF] Loading Excel: {EXCEL_FILE}")
xls = pd.ExcelFile(EXCEL_FILE)
df_edges = pd.read_excel(xls, sheet_name=SHEET_EDGES)
df_refs  = pd.read_excel(xls, sheet_name=SHEET_REFS)
df_desc  = pd.read_excel(xls, sheet_name=SHEET_DESC, usecols=[0, 1])

# ==================================================
# REFERENCES LOOKUP
# ==================================================
refs_lookup = {
    str(r.iloc[0]).split(".")[0].strip(): str(r.iloc[1])
    for _, r in df_refs.iterrows()
    if pd.notna(r.iloc[0])
}

# ==================================================
# DESCRIPTOR → CATEGORY (ONLY FROM DESCRIPTOR SHEET)
# ==================================================
df_desc.columns = ["Category", "Concept"]
df_desc["Category"] = df_desc["Category"].ffill()

# On normalise la colonne Concept avec notre nouvelle fonction
concept_to_subcat = {
    normalize_concept_name(str(r["Concept"])): str(r["Category"]).strip()
    for _, r in df_desc.iterrows()
    if pd.notna(r["Concept"])
}

# ==================================================
# HELPERS: process/control classification (controlled, no inference)
# ==================================================
def phase_for_process(concept: str) -> str | None:
    """Returns a phase if the process is explicitly mapped."""
    cl = concept.lower()
    for k, phase in PROCESS_TO_PHASE.items():
        if k in cl:
            return phase
    return None

def is_forced_env_control(concept: str) -> bool:
    cl = concept.lower()
    return any(k in cl for k in FORCE_ENV_CONTROL)

def base_process_key(concept: str) -> str:
    """Take prefix before ':' for process descriptions (no identity change)."""
    if ":" in concept:
        return concept.split(":", 1)[0].strip().lower()
    return concept.lower()

# ==================================================
# BUILD GRAPH
# ==================================================
nodes = {}
edges = []

print("[REF] Building reference graph...")

for _, row in df_edges.iterrows():
    label = str(row.get("Label", ""))
    if " - " not in label:
        continue

    raw_source, raw_target = [x.strip() for x in label.split(" - ", 1)]

    source = normalize_concept_name(raw_source)
    target = normalize_concept_name(raw_target)

    # Enregistrement des nœuds
    for n in [source, target]:
        if n in nodes:
            continue

        # Descripteurs (Blue)
        if n in concept_to_subcat and concept_to_subcat[n] in SUBCAT_TO_MAIN:
            subcat = concept_to_subcat[n]
            nodes[n] = {
                "label": n,
                "sub_category": subcat,
                "main_category": SUBCAT_TO_MAIN[subcat]
            }
            continue

        # Processus (Orange)
        phase = phase_for_process(n)
        if phase is not None:
            desc_key = base_process_key(n)
            node = {
                "label": n,
                "sub_category": phase,
                "main_category": "Mass Movement Property (Orange)"
            }
            if desc_key in PROCESS_DESCRIPTIONS:
                node["description"] = PROCESS_DESCRIPTIONS[desc_key]
            nodes[n] = node
            continue

        # Contrôles environnementaux (Green)
        nodes[n] = {
            "label": n,
            "sub_category": "Environmental controls",
            "main_category": "Environmental Control (Green)"
        }

    # Déterminer si l'arête est dirigée
    edge_type = str(row.get("Type", "")).strip().lower()
    directed = edge_type == "directed"

    # Gérer les références bibliographiques
    citations = []
    if pd.notna(row.get("Reference #", None)):
        ids = [x.strip().split(".")[0] for x in str(row["Reference #"]).split(",")]
        citations = [refs_lookup.get(i, "Unknown") for i in ids]

    # Ajout de l’arête
    edges.append({
        "source": source,
        "target": target,
        "directed": directed,
        "relation": "impacts" if directed else "related_to",
        "citations": citations,
        "raw_label": f"{raw_source} - {raw_target}"
    })

# ==================================================
# REPORT
# ==================================================
subcat_stats = Counter(v["sub_category"] for v in nodes.values())
main_stats = Counter(v["main_category"] for v in nodes.values())

print("[REF] Node main-category distribution:")
for k, v in main_stats.items():
    print(f"  - {k}: {v}")

print("[REF] Node sub-category distribution:")
for k, v in subcat_stats.items():
    print(f"  - {k}: {v}")

# ==================================================
# SAVE
# ==================================================
final_data = {
    "project": "MTD Ontology – Expert Reference (Le Bouteiller compliant, phase-aware)",
    "hierarchy": HIERARCHY,
    "nodes": nodes,
    "edges": edges
}

os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)
with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=4, ensure_ascii=False)

print(f"[REF] DONE — {len(nodes)} nodes | {len(edges)} edges")
print(f"[REF] Saved → {OUTPUT_FILENAME}")
