# rag/validate.py
# -*- coding: utf-8 -*-
"""
Triple validation against schema and lexicon.
"""
from typing import Dict, Any, List, Tuple
from rag.schema import KGSchema

def canonicalize_entity(schema: KGSchema, name: str) -> Tuple[str, bool]:
    """
    Map alias to canonical term using lexicon.
    
    Args:
        schema: KGSchema object
        name: Entity name to canonicalize
    
    Returns:
        (canonical_name, in_lexicon)
    """
    key = name.strip().lower()
    if key in schema.alias_to_canonical:
        return schema.alias_to_canonical[key], True
    return name.strip(), False

def validate_triple(schema: KGSchema, t: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate a single triple against schema.
    
    RELAXED VALIDATION:
    - Accepts entities not in lexicon (just marks status)
    - Only rejects invalid types/relations
    
    Args:
        schema: KGSchema object
        t: Triple dict to validate
    
    Returns:
        (is_valid, reason, enriched_triple)
    """
    required = ["source", "source_type", "relation", "target", "target_type", "evidence"]
    
    # Check required fields
    for k in required:
        if k not in t:
            return False, f"missing_field:{k}", t
    
    # Check relation type (strict)
    if t["relation"] not in schema.allowed_relation_types_set:
        return False, "relation_not_allowed", t
    
    # Check node types (strict)
    if t["source_type"] not in schema.allowed_node_types_set:
        return False, "source_type_not_allowed", t
    if t["target_type"] not in schema.allowed_node_types_set:
        return False, "target_type_not_allowed", t
    
    # Canonicalize entities (but don't reject if not in lexicon)
    src_can, src_ok = canonicalize_entity(schema, t["source"])
    tgt_can, tgt_ok = canonicalize_entity(schema, t["target"])
    
    # Enrich triple with canonicalization info
    t2 = dict(t)
    t2["source_canonical"] = src_can
    t2["target_canonical"] = tgt_can
    t2["source_in_lexicon"] = src_ok
    t2["target_in_lexicon"] = tgt_ok
    
    # Check evidence format
    ev = t2.get("evidence", {})
    if not isinstance(ev, dict) or "quote" not in ev or "confidence" not in ev:
        return False, "bad_evidence", t2
    
    # ACCEPT (even if not in lexicon)
    return True, "ok", t2

def validate_triples(schema: KGSchema, triples: List[Dict[str, Any]]) -> Tuple[List[dict], List[dict]]:
    """
    Validate a list of triples.
    
    Args:
        schema: KGSchema object
        triples: List of triple dicts
    
    Returns:
        (accepted_triples, rejected_triples)
    """
    accepted = []
    rejected = []
    
    for t in triples:
        ok, reason, t2 = validate_triple(schema, t)
        if ok:
            accepted.append(t2)
        else:
            rejected.append({"reason": reason, "triple": t2})
    
    return accepted, rejected