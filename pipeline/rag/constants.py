"""
pipeline/rag/constants.py — Shared ontology constants.

Single source of truth — import from here in every pipeline script.
"""

# ── Relation ontology ─────────────────────────────────────────────────

ALLOWED_RELATIONS: set[str] = {
    "hasDescriptor", "occursIn", "formedBy", "partOf",
    "triggers", "causes", "controls", "affects",
    "overlies", "underlies", "indicates", "evidences", "relatedTo",
}

RELATION_MAP: dict[str, str] = {
    "hasdescriptor":         "hasDescriptor",
    "occursin":              "occursIn",
    "formedby":              "formedBy",
    "partof":                "partOf",
    "triggeredby":           "triggers",
    "triggered_by":          "triggers",
    "overlays":              "overlies",
    "relatedto":             "relatedTo",
    "related_to":            "relatedTo",
    "related to":            "relatedTo",
    "hasfeature":            "hasDescriptor",
    "ischaracterizedby":     "hasDescriptor",
    "characterizedby":       "hasDescriptor",
    "locatedin":             "occursIn",
    "foundin":               "occursIn",
}

RELATION_GLOSSES: dict[str, str] = {
    "hasDescriptor": "is characterised by / exhibits",
    "occursIn":      "is found in / located in",
    "formedBy":      "is formed by / produced by",
    "partOf":        "is a part / component of",
    "triggers":      "initiates / triggers",
    "causes":        "directly produces",
    "controls":      "governs / regulates",
    "affects":       "influences / modifies",
    "overlies":      "is stratigraphically above",
    "underlies":     "is stratigraphically below",
    "indicates":     "indicates",
    "evidences":     "provides evidence for",
    "relatedTo":     "is related to",
}


# ── LB2019 ground truth ───────────────────────────────────────────────

LB2019_DESCRIPTORS: set[str] = {
    "blocky", "chaotic", "continuous", "discontinuous",
    "high-amplitude", "hummocky", "layered", "low-amplitude",
    "massive", "parallel", "stratified", "transparent", "undeformed",
}

DESCRIPTOR_SYNONYMS: dict[str, str] = {
    "stratified":                "layered",
    "sub-parallel":              "parallel",
    "sub parallel":              "parallel",
    "essentially undeformed":    "undeformed",
    "low amplitude":             "low-amplitude",
    "high amplitude":            "high-amplitude",
    "low-amplitude reflection":  "low-amplitude",
    "high-amplitude reflection": "high-amplitude",
}

# All 26 LB2019 reference edges (ground truth)
LB2019_REFERENCE_EDGES: list[tuple[str, str, str]] = [
    ("mass transport deposit", "hasDescriptor", "chaotic"),
    ("mass transport deposit", "hasDescriptor", "transparent"),
    ("mass transport deposit", "hasDescriptor", "hummocky"),
    ("mass transport deposit", "hasDescriptor", "blocky"),
    ("mass transport deposit", "hasDescriptor", "discontinuous"),
    ("mass transport deposit", "hasDescriptor", "massive"),
    ("turbidite", "hasDescriptor", "parallel"),
    ("turbidite", "hasDescriptor", "continuous"),
    ("turbidite", "hasDescriptor", "layered"),
    ("turbidite", "hasDescriptor", "high-amplitude"),
    ("debris flow", "hasDescriptor", "chaotic"),
    ("debris flow", "hasDescriptor", "hummocky"),
    ("slide", "hasDescriptor", "blocky"),
    ("slide", "hasDescriptor", "undeformed"),
    ("hemipelagite", "hasDescriptor", "parallel"),
    ("hemipelagite", "hasDescriptor", "continuous"),
    ("hemipelagite", "hasDescriptor", "low-amplitude"),
    ("slope failure", "causes", "mass transport deposit"),
    ("earthquake", "triggers", "slope failure"),
    ("pore pressure", "controls", "slope failure"),
    ("turbidity current", "formedBy", "debris flow"),
    ("mass transport deposit", "occursIn", "continental slope"),
    ("mass transport deposit", "occursIn", "abyssal plain"),
    ("debris flow", "occursIn", "continental slope"),
    ("turbidite", "occursIn", "basin floor"),
    ("slide", "overlies", "hemipelagite"),
]


# ── Known geological terms ─────────────────────────────────────────────

KNOWN_DESCRIPTORS: set[str] = LB2019_DESCRIPTORS | {
    "mounded", "divergent", "convergent", "wavy", "contorted",
    "folded", "faulted", "deformed", "disrupted", "draping",
    "onlapping", "erosional", "aggradational", "progradational",
    "retrogradational", "tabular", "lenticular", "wedge-shaped",
    "sheet-like", "channelised", "irregular", "smooth", "rough",
    "thick", "thin", "variable-amplitude", "moderate-amplitude",
}

KNOWN_SETTINGS: set[str] = {
    "continental slope", "continental shelf", "continental margin",
    "abyssal plain", "basin floor", "submarine canyon", "channel",
    "deep-water environment", "deep-water environments", "deep water",
    "passive margin", "active margin", "accretionary prism",
    "trench", "mid-ocean ridge", "seamount", "delta", "fan",
    "submarine fan", "levee", "overbank",
}

# Entity normalization map (MTD variants → canonical)
MTD_VARIANTS: set[str] = {
    "mass-transport deposit", "mass-transport deposits",
    "mass-transport deposits (mtd)", "mtd", "mass transport deposits",
    "mass transport deposit 1", "mass transport deposit 2",
    "mtd 1", "mtd 2", "mass transport complex", "mass-transport complex",
}

ENTITY_NORMS: dict[str, str] = {v: "mass transport deposit" for v in MTD_VARIANTS}
ENTITY_NORMS.update({
    "debris flows":                  "debris flow",
    "debrites":                      "debris flow",
    "turbidity currents":            "turbidity current",
    "turbidites":                    "turbidite",
    "slides":                        "slide",
    "slumps":                        "slump",
    "low amplitude":                 "low-amplitude",
    "high amplitude":                "high-amplitude",
    "low amplitude reflections":     "low-amplitude",
    "high amplitude reflections":    "high-amplitude",
})


def normalize_relation(rel: str) -> str:
    """Normalize a relation string to canonical form."""
    if not rel:
        return ""
    key = rel.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    return RELATION_MAP.get(key, rel.strip())


def normalize_entity(text: str) -> str:
    """Lowercase, strip, collapse whitespace, apply entity norms."""
    import re
    t = re.sub(r"\s+", " ", (text or "").lower().strip()).rstrip(".,;:")
    return ENTITY_NORMS.get(t, t)


def normalize_descriptor(text: str) -> str:
    """Apply descriptor synonyms after entity normalization."""
    t = normalize_entity(text)
    return DESCRIPTOR_SYNONYMS.get(t, t)