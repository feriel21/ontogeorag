#!/usr/bin/env python3
"""
03b_canonicalize.py — Entity Canonicalization (EDC-inspired).

Groups semantically similar entities via embedding clustering,
then selects a canonical form for each group.

FIXED:
- Threshold lowered to 0.06 (was 0.15 — caused wrong merges)
- Added same-type constraint (never merge Process with Descriptor)
- Protected LB2019 descriptor terms from being merged

Input:  cleaned_triples_v4.jsonl (from step 03)
Output: canonical_triples_v4.jsonl (with merged entities)
        canonical_map_v4.json (entity -> canonical mapping)
"""

import argparse
import json
import os
from collections import Counter, defaultdict

# ── LEXICON (must match 03_validate_and_clean_v4.py) ──────────────────────

LEXICON = {
    "mass transport deposit": "SeismicObject",
    "mass transport complex": "SeismicObject",
    "debris flow deposit": "SeismicObject",
    "turbidite": "SeismicObject",
    "submarine landslide": "SeismicObject",
    "slide": "SeismicObject",
    "slump": "SeismicObject",
    "channel levee": "SeismicObject",
    "levee": "SeismicObject",
    "submarine fan": "SeismicObject",
    "canyon": "SeismicObject",
    "scarp": "SeismicObject",
    "headscarp": "SeismicObject",
    "basal surface": "SeismicObject",
    "upper surface": "SeismicObject",
    "basal shear surface": "SeismicObject",
    "lateral erosive wall": "SeismicObject",
    "preserved block": "SeismicObject",
    "rafted block": "SeismicObject",
    "megaclast": "SeismicObject",
    "toe": "SeismicObject",
    "deformed strata": "SeismicObject",
    "disrupted strata": "SeismicObject",
    "undeformed strata": "SeismicObject",
    "chaotic": "Descriptor",
    "transparent": "Descriptor",
    "blocky": "Descriptor",
    "massive": "Descriptor",
    "hummocky": "Descriptor",
    "discontinuous": "Descriptor",
    "high-amplitude": "Descriptor",
    "low-amplitude": "Descriptor",
    "undeformed": "Descriptor",
    "layered": "Descriptor",
    "stratified": "Descriptor",
    "continuous": "Descriptor",
    "parallel": "Descriptor",
    "erosional": "Descriptor",
    "irregular": "Descriptor",
    "lobate": "Descriptor",
    "elongated": "Descriptor",
    "tabular": "Descriptor",
    "ponded": "Descriptor",
    "arcuate": "Descriptor",
    "stepped": "Descriptor",
    "deformed": "Descriptor",
    "disrupted": "Descriptor",
    "thickness": "Descriptor",
    "morphology": "Descriptor",
    "internal facies": "Descriptor",
    "debris flow": "Process",
    "turbidity current": "Process",
    "slope failure": "Process",
    "retrogressive failure": "Process",
    "erosion": "Process",
    "remobilization": "Process",
    "deposition": "Process",
    "sedimentation": "Process",
    "compaction": "Process",
    "burial compaction": "Process",
    "fluid migration": "Process",
    "fluid overpressure": "Process",
    "mass wasting": "Process",
    "frontal compression": "Process",
    "basal erosion": "Process",
    "flow behavior": "Process",
    "grain heterogeneity": "Process",
    "sedimentation rate": "Process",
    "tectonic activity": "Process",
    "seismic loading": "Process",
    "gas hydrate dissociation": "Process",
    "wave loading": "Process",
    "rapid sedimentation": "Process",
    "slope instability": "Process",
    "continental slope": "Setting",
    "continental margin": "Setting",
    "passive margin": "Setting",
    "slope": "Setting",
    "delta": "Setting",
    "shelf": "Setting",
    "deep-sea fan": "Setting",
    "submarine slope": "Setting",
    "abyssal plain": "Setting",
}

# LB2019 descriptor terms that must NEVER be merged with each other
LB_DESCRIPTORS = {
    "chaotic", "transparent", "blocky", "massive", "hummocky",
    "discontinuous", "high-amplitude", "low-amplitude",
    "undeformed", "layered", "stratified", "continuous", "parallel",
}


# ── Embedding model ───────────────────────────────────────────────────────

_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        print("  Loading SciBERT for entity embeddings...")
        _embed_model = SentenceTransformer("allenai/scibert_scivocab_uncased")
    return _embed_model


# ── Clustering ────────────────────────────────────────────────────────────

def build_canonical_map(entities, embeddings, distance_threshold=0.06):
    """
    Cluster entities by cosine distance and pick canonical forms.

    distance_threshold=0.06 is very conservative — only truly redundant
    strings like "mtd" / "mass transport deposit" will merge.
    """
    from sklearn.cluster import AgglomerativeClustering

    if len(entities) < 2:
        return {}

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    groups = defaultdict(list)
    for entity, label in zip(entities, labels):
        groups[label].append(entity)

    canonical_map = {}
    for label, members in groups.items():
        if len(members) == 1:
            continue

        # Priority: in LEXICON > longer > alphabetical first
        best = sorted(members, key=lambda e: (
            e not in LEXICON,
            -len(e),
            e,
        ))[0]

        for m in members:
            if m != best:
                canonical_map[m] = best

    return canonical_map


def validate_canonical_map(canonical_map):
    """Remove merges that would corrupt the KG."""
    bad_merges = []

    for old, new in list(canonical_map.items()):
        old_type = LEXICON.get(old, "Unknown")
        new_type = LEXICON.get(new, "Unknown")

        # Rule 1: Never merge entities of different ontology types
        if (old_type != "Unknown" and new_type != "Unknown"
                and old_type != new_type):
            bad_merges.append((old, new, f"cross-type: {old_type}->{new_type}"))
            del canonical_map[old]
            continue

        # Rule 2: Never merge distinct LB2019 descriptor terms
        if old in LB_DESCRIPTORS and new in LB_DESCRIPTORS and old != new:
            bad_merges.append((old, new, "both LB descriptors"))
            del canonical_map[old]
            continue

        # Rule 3: Never merge an LB descriptor with a non-descriptor
        if (old in LB_DESCRIPTORS) != (new in LB_DESCRIPTORS):
            bad_merges.append((old, new, "LB descriptor + non-descriptor"))
            del canonical_map[old]
            continue

    return canonical_map, bad_merges


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="03b — Entity canonicalization (EDC-inspired)"
    )
    parser.add_argument(
        "--input",
        default="output/step4/cleaned_triples_v4.jsonl",
        help="Input JSONL (cleaned triples from step 03)",
    )
    parser.add_argument(
        "--output",
        default="output/step4/canonical_triples_v4.jsonl",
        help="Output JSONL with canonicalized entities",
    )
    parser.add_argument(
        "--map",
        default="output/step4/canonical_map_v4.json",
        help="Output JSON mapping old entity -> canonical entity",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.06,
        help="Cosine distance threshold for clustering (default: 0.06)",
    )
    args = parser.parse_args()

    # ── Load triples ──────────────────────────────────────────────────────
    print(f"Loading triples from {args.input}...")
    triples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    triples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"  Loaded {len(triples)} triples")

    if not triples:
        print("  No triples to canonicalize. Writing empty output.")
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        open(args.output, "w").close()
        with open(args.map, "w") as f:
            json.dump({}, f)
        return

    # ── Collect all unique entities ───────────────────────────────────────
    all_entities = set()
    for t in triples:
        s = t.get("source_norm", "")
        tgt = t.get("target_norm", "")
        if s:
            all_entities.add(s)
        if tgt:
            all_entities.add(tgt)

    entity_list = sorted(all_entities)
    print(f"  Unique entities: {len(entity_list)}")

    if len(entity_list) < 2:
        print("  Too few entities to cluster. Copying input to output.")
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for t in triples:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        with open(args.map, "w") as f:
            json.dump({}, f)
        return

    # ── Compute embeddings ────────────────────────────────────────────────
    print(f"  Computing SciBERT embeddings for {len(entity_list)} entities...")
    model = get_embed_model()
    embeddings = model.encode(entity_list, show_progress_bar=False)
    print(f"  Embeddings shape: {embeddings.shape}")

    # ── Build & validate canonical map ────────────────────────────────────
    print(f"  Clustering (distance_threshold={args.threshold})...")
    canonical_map = build_canonical_map(
        entity_list, embeddings, distance_threshold=args.threshold
    )

    # Validate: remove cross-type and descriptor merges
    canonical_map, bad_merges = validate_canonical_map(canonical_map)

    print(f"  Canonical map: {len(canonical_map)} merge rules")

    if bad_merges:
        print(f"  Blocked {len(bad_merges)} bad merges:")
        for old, new, reason in bad_merges:
            print(f"    BLOCKED: '{old}' -> '{new}' ({reason})")

    if canonical_map:
        print(f"\n  Applied merges:")
        for old, new in sorted(canonical_map.items()):
            print(f"    '{old}' -> '{new}'")

    # ── Apply canonical map to triples ────────────────────────────────────
    print(f"\n  Applying canonical map to {len(triples)} triples...")
    merged_count = 0
    for t in triples:
        s = t.get("source_norm", "")
        tgt = t.get("target_norm", "")

        if s in canonical_map:
            t["_source_before_canon"] = s
            t["source_norm"] = canonical_map[s]
            merged_count += 1

        if tgt in canonical_map:
            t["_target_before_canon"] = tgt
            t["target_norm"] = canonical_map[tgt]
            merged_count += 1

    print(f"  Entity occurrences merged: {merged_count}")

    # ── Re-deduplicate ────────────────────────────────────────────────────
    print(f"\n  Re-deduplicating after canonicalization...")
    seen = set()
    unique = []
    for t in triples:
        key = (
            t.get("source_norm", ""),
            t.get("relation_norm", ""),
            t.get("target_norm", ""),
        )
        if key not in seen:
            seen.add(key)
            unique.append(t)

    dupes_removed = len(triples) - len(unique)
    print(f"  Before: {len(triples)}")
    print(f"  After:  {len(unique)}")
    print(f"  Duplicates removed: {dupes_removed}")

    # ── Filter self-loops created by merging ──────────────────────────────
    before_loop_filter = len(unique)
    unique = [
        t for t in unique
        if t.get("source_norm", "") != t.get("target_norm", "")
    ]
    loops_removed = before_loop_filter - len(unique)
    if loops_removed:
        print(f"  Self-loops removed: {loops_removed}")

    # ── Write outputs ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for t in unique:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"\n  Wrote {len(unique)} canonical triples to {args.output}")

    os.makedirs(os.path.dirname(args.map) or ".", exist_ok=True)
    with open(args.map, "w", encoding="utf-8") as f:
        json.dump(canonical_map, f, indent=2, ensure_ascii=False)
    print(f"  Wrote canonical map to {args.map}")

    # ── Summary ───────────────────────────────────────────────────────────
    final_entities = set()
    for t in unique:
        if t.get("source_norm"):
            final_entities.add(t["source_norm"])
        if t.get("target_norm"):
            final_entities.add(t["target_norm"])

    print(f"\n{'='*60}")
    print("  CANONICALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Input triples:      {len(triples)}")
    print(f"  Output triples:     {len(unique)}")
    print(f"  Entities before:    {len(entity_list)}")
    print(f"  Entities after:     {len(final_entities)}")
    print(f"  Merge rules:        {len(canonical_map)}")
    print(f"  Bad merges blocked: {len(bad_merges)}")
    print(f"  Duplicates removed: {dupes_removed}")
    print(f"  Self-loops removed: {loops_removed}")


if __name__ == "__main__":
    main()