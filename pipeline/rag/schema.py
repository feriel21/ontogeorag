# rag/schema.py
# -*- coding: utf-8 -*-
"""
Knowledge Graph schema and lexicon management.
"""
import json
from pathlib import Path
from typing import List, Dict, Set
from dataclasses import dataclass

@dataclass
class KGSchema:
    """Knowledge Graph schema with validation rules."""
    node_types: List[str]
    relation_types: List[str]
    lexicon: List[Dict]  # [{"concept": "...", "node_type": "...", "aliases": [...]}]
    
    # Computed sets for fast lookup
    allowed_node_types_set: Set[str]
    allowed_relation_types_set: Set[str]
    alias_to_canonical: Dict[str, str]  # {"alias_lower": "canonical_term"}

def load_schema(schema_path: str, lexicon_path: str) -> KGSchema:
    """
    Load schema and lexicon from JSON files.
    
    Args:
        schema_path: Path to schema_step1.json
        lexicon_path: Path to lexicon.json
    
    Returns:
        KGSchema object with validation rules
    """
    schema_data = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    lexicon_data = json.loads(Path(lexicon_path).read_text(encoding="utf-8"))
    
    node_types = schema_data["node_types"]
    relation_types = schema_data["relation_types"]
    
    # Build alias-to-canonical mapping
    alias_to_canonical = {}
    for entry in lexicon_data:
        canonical = entry["concept"]
        aliases = entry.get("aliases", [])
        
        # Map canonical term itself
        alias_to_canonical[canonical.lower()] = canonical
        
        # Map all aliases to canonical
        if isinstance(aliases, list):
            for alias in aliases:
                alias_to_canonical[alias.lower()] = canonical
    
    return KGSchema(
        node_types=node_types,
        relation_types=relation_types,
        lexicon=lexicon_data,
        allowed_node_types_set=set(node_types),
        allowed_relation_types_set=set(relation_types),
        alias_to_canonical=alias_to_canonical
    )