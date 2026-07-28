#!/usr/bin/env python3
"""
pipeline/generate_augmented_queries.py — Type-Aware Query Augmentation
===========================================================================
WHY
    Generating every query strategy (descriptor/causal/context/spatial) for
    every entity produces geologically nonsensical queries (e.g. "what
    triggers this Descriptor?") that waste LLM calls and dilute retrieval
    quality. Gating by entity type keeps augmentation queries coherent.

WHAT
    Generates augmented queries from a KG, respecting each entity's TYPE.
    A strategy is only generated if it is geologically coherent with the
    entity's type (TYPE_STRATEGIES), and only for strategies not already
    covered by an existing relation on that entity. Avoids query noise.
"""

import argparse
import json
import re
from collections import Counter, defaultdict

TEMPLATES = {
    "descriptor": "What are the seismic facies and descriptors of {e}?",
    "causal": "What processes cause or trigger {e}? What does {e} cause?",
    "context": "In what geological or depositional setting does {e} occur?",
    "spatial": "What is the spatial or stratigraphic position of {e} relative to other deposits?",
}
REL_TO_STRATEGY = {
    "hasdescriptor": "descriptor",
    "causes": "causal",
    "triggers": "causal",
    "formedby": "causal",
    "controls": "causal",
    "occursin": "context",
    "partof": "context",
    "overlies": "spatial",
    "underlies": "spatial",
}
# strategies ALLOWED per entity type (geological coherence)
TYPE_STRATEGIES = {
    "SeismicObject": {"descriptor", "context", "spatial", "causal"},
    "GeologicalObject": {"descriptor", "context", "spatial", "causal"},
    "GeologicalFeature": {"descriptor", "context", "spatial"},
    "StructuralComponent": {"context", "spatial"},
    "Process": {"causal", "context"},
    "Condition": {"causal"},
    "Factor": {"causal"},
    "Overpressure": {"causal"},
    "Domain": {"context", "spatial"},
    "GeologicalSetting": {"context", "spatial"},
    "Substance": {"causal", "context"},
    # types NOT expanded (relation objects or properties) -> empty set
    "Descriptor": set(),
    "Facies": set(),
    "Shape": set(),
    "Attribute": set(),
    "Property": set(),
    "Rate": set(),
    "Measurement": set(),
    "Criterion": set(),
    "Effect": set(),
    "Event": set(),
    "Collapse": set(),
    "Location": set(),
    "GeologicalSurface": set(),
    "StructuralSurface": set(),
    "SurfaceFeature": set(),
}


def norm(s):
    """Lowercase and collapse whitespace in `s`; no side effects."""
    return re.sub(r"\s+", " ", (s or "").lower().strip())


def main():
    """CLI entry point: scans --kg for entities and their relation-covered strategies, generates missing type-coherent augmented queries, and writes them (one JSON object per line) to --output."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-entity-count", type=int, default=2)
    args = ap.parse_args()

    d = json.load(open(args.kg))
    triples = (
        d.get("triples_final", d.get("triples", d))
        if isinstance(d, dict)
        else d
    )

    covered = defaultdict(set)
    entity_count = defaultdict(int)
    entity_type = {}
    for t in triples:
        rel = (
            norm(t.get("relation", ""))
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )
        strat = REL_TO_STRATEGY.get(rel)
        for role, typ in (
            ("subject", "subject_type"),
            ("object", "object_type"),
        ):
            e = norm(t.get(role, ""))
            if not e:
                continue
            entity_count[e] += 1
            # keep the most informative type seen so far for this entity
            et = t.get(typ, "")
            if et and e not in entity_type:
                entity_type[e] = et
            if strat:
                covered[e].add(strat)

    queries = []
    seen = set()
    kept = 0
    skipped_type = 0
    skipped_count = 0
    for e, cnt in entity_count.items():
        if cnt < args.min_entity_count:
            skipped_count += 1
            continue
        etype = entity_type.get(e, "")
        allowed = TYPE_STRATEGIES.get(
            etype, {"causal", "context"}
        )  # conservative default
        if not allowed:
            skipped_type += 1
            continue
        kept += 1
        for strat, tmpl in TEMPLATES.items():
            if strat in allowed and strat not in covered[e]:
                q = tmpl.format(e=e)
                key = (q, strat)
                if key in seen:
                    continue
                seen.add(key)
                queries.append(
                    {
                        "query": q,
                        "strategy": strat,
                        "focus": e,
                        "focus_type": etype,
                        "source": "kg_augmented",
                    }
                )

    with open(args.output, "w") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(
        f"Entités expansées: {kept} | filtrées (type non-expansable): {skipped_type} | (rares <{args.min_entity_count}): {skipped_count}"
    )
    print(f"Requêtes augmentées: {len(queries)} -> {args.output}")
    print(f"  par stratégie: {dict(Counter(q['strategy'] for q in queries))}")
    print(
        f"  par type focus: {dict(Counter(q['focus_type'] for q in queries))}"
    )


if __name__ == "__main__":
    main()
