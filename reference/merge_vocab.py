import os, re, json, pandas as pd
from pathlib import Path

XLSX = "reference/Table_Supplementary_1_V2.xlsx"
OUT = Path("schema_seed_output")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(XLSX)

def canon(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("’","'")
    s = re.sub(r"[“”\"']", "", s)
    s = re.sub(r"[^a-z0-9\s\-\(\)\.:/]", "", s)
    return s.strip()

def norm_for_merge(s: str) -> str:
    """More aggressive normalization for duplicate detection."""
    s = canon(s)
    # remove punctuation that often differs
    s2 = re.sub(r"[\-\(\)\.:/]", " ", s)
    s2 = re.sub(r"\s+", " ", s2).strip()
    # simple plural normalization
    if s2.endswith("s") and len(s2) > 4:
        s2 = s2[:-1]
    return s2

# Build concept list from labels (src - tgt)
concepts = []
excluded = []
for _, row in df.iterrows():
    label = str(row.get("Label","")).strip()
    comment = "" if pd.isna(row.get("Comment")) else str(row.get("Comment")).strip()
    if not label:
        continue
    if "undetermined edge" in comment.lower():
        excluded.append(label); 
        continue
    if " - " not in label:
        excluded.append(label); 
        continue
    src_raw, tgt_raw = [x.strip() for x in label.split(" - ", 1)]
    concepts.extend([src_raw, tgt_raw])

concepts_df = pd.DataFrame({"original_label": concepts}).drop_duplicates().reset_index(drop=True)
concepts_df["canonical_term"] = concepts_df["original_label"].map(canon)
concepts_df["merge_key"] = concepts_df["original_label"].map(norm_for_merge)

# Group possible duplicates (same merge_key but different canonical_term/original_label)
dup_groups = (concepts_df.groupby("merge_key")
              .agg(count=("original_label","size"),
                   examples=("original_label", lambda x: list(x)[:10]))
              .reset_index()
              .query("count > 1")
              .sort_values(["count","merge_key"], ascending=[False, True]))

# Create a suggested mapping: pick shortest label as preferred name per merge_key
preferred = (concepts_df.assign(label_len=concepts_df["original_label"].str.len())
             .sort_values(["merge_key","label_len","original_label"])
             .groupby("merge_key").first().reset_index()[["merge_key","original_label","canonical_term"]]
             .rename(columns={"original_label":"preferred_label","canonical_term":"preferred_canonical"}))

concepts_df = concepts_df.merge(preferred, on="merge_key", how="left")

# Create alias list per preferred_canonical
aliases = (concepts_df.groupby("preferred_canonical")["original_label"]
           .apply(lambda x: sorted(set(x)))
           .reset_index()
           .rename(columns={"original_label":"aliases"}))

# Build cleaned concepts table
clean_concepts = (preferred.merge(aliases, left_on="preferred_canonical", right_on="preferred_canonical", how="left")
                  .drop(columns=["preferred_canonical"])
                  .rename(columns={"preferred_label":"concept_label"}))

# Placeholder node type column for user to edit
clean_concepts["node_type"] = ""  # user will fill: Geological_Object/Descriptor/Process/Environmental_Control/Evidence
clean_concepts["source"] = "LeBouteiller_TableS1"

# Save outputs
concepts_seed_path = OUT / "concepts_seed.csv"
concepts_df.to_csv(concepts_seed_path, index=False)

dup_report_path = OUT / "duplicate_candidates.csv"
dup_groups.to_csv(dup_report_path, index=False)

clean_concepts_path = OUT / "concepts_cleaned_template.csv"
clean_concepts.to_csv(clean_concepts_path, index=False)

# Build Step-1 schema files from fixed node types + relation types (minimal) + lexicon
schema = {
    "node_types": ["Geological_Object","Descriptor","Process","Environmental_Control","Evidence"],
    "relation_types": ["AFFECTS","CAUSES","TRIGGERS","CONTROLS","CONDITIONS","INDICATES","SUGGESTS","EVIDENCES","PART_OF","HAS_DESCRIPTOR","HAS_PROPERTY","HAS_PHASE","LOCATED_IN","RELATED_TO"],
    "allowed_node_attributes": {
        "Geological_Object": ["name","canonical_id","object_class","description","dataset_id","geometry_ref","aliases"],
        "Descriptor": ["name","canonical_id","descriptor_subtype","value_type","units","description","aliases"],
        "Process": ["name","canonical_id","process_subtype","phase","description","aliases"],
        "Environmental_Control": ["name","canonical_id","control_type","value_type","units","description","aliases"],
        "Evidence": ["source_id","source_type","citation","span","page","figure","confidence"]
    },
    "allowed_edge_attributes": {
        "AFFECTS": ["polarity","strength","confidence","source_id","span"],
        "CAUSES": ["confidence","source_id","span"],
        "TRIGGERS": ["confidence","source_id","span"],
        "CONTROLS": ["confidence","source_id","span"],
        "CONDITIONS": ["confidence","source_id","span"],
        "INDICATES": ["confidence","source_id","span"],
        "SUGGESTS": ["confidence","source_id","span"],
        "EVIDENCES": ["confidence","source_id","span"],
        "PART_OF": ["confidence","source_id","span"],
        "HAS_DESCRIPTOR": ["confidence","source_id","span"],
        "HAS_PROPERTY": ["confidence","source_id","span"],
        "HAS_PHASE": ["confidence","source_id","span"],
        "LOCATED_IN": ["confidence","source_id","span"],
        "RELATED_TO": ["confidence","source_id","span"],
    },
    "lexicon": {
        "concepts_file": str(clean_concepts_path.name),
        "notes": "Fill node_type in concepts_cleaned_template.csv, then export lexicon.json from it."
    }
}

schema_json_path = OUT / "schema_step1.json"
with open(schema_json_path, "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2, ensure_ascii=False)

# Show a small preview
dup_groups.head(10), clean_concepts.head(10), str(OUT)
