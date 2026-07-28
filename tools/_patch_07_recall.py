"""
In-place patch of pipeline/07_final_metrics.py:
- Replace the recall() function with a version that:
    * loads reference edges from configs/lb_reference_edges.json (34-edge),
    * uses normalize_relation() from pipeline.rag.constants,
    * uses unidirectional substring matching (ref_subj in kg_subj, ref_obj in kg_obj),
    * also reports an exact-match recall as a 'lower_bound' field.
- Does not touch any other function.
- Refuses to run if the file has already been patched (idempotent).
"""
from pathlib import Path
import re, sys

src_path = Path("pipeline/07_final_metrics.py")
src = src_path.read_text()

MARKER = "# === unified recall block (patched) ==="
if MARKER in src:
    print("Already patched. Aborting to avoid double-patching.")
    sys.exit(0)

# Locate the existing recall() function and replace it.
pat = re.compile(
    r"def recall\(triples\):.*?(?=\n(?:def |class |\Z))",
    re.DOTALL,
)
m = pat.search(src)
if not m:
    print("Could not locate existing recall() function. Aborting.")
    sys.exit(1)

new_block = MARKER + """
def _load_reference_edges_from_json(ref_path="configs/lb_reference_edges.json"):
    import json
    from pathlib import Path as _P
    obj = json.load(open(_P(ref_path)))
    edges = obj.get("edges", obj) if isinstance(obj, dict) else obj
    out = []
    for e in edges:
        if isinstance(e, dict):
            out.append((e.get("subject",""), e.get("relation",""), e.get("object","")))
        else:
            out.append(tuple(e))
    return out

def recall(triples, ref_path="configs/lb_reference_edges.json"):
    \"\"\"Recall vs LB2019 benchmark, unified with the SLURM inline / paper headline matcher.

    Matching rule (per paper §3): a benchmark edge (rs, rr, ro) is recovered iff
    there exists an extracted triple (ts, tr, to) such that
        rs is a substring of ts (after lowercasing/whitespace/entity normalization),
        ro is a substring of to,
        normalize_relation(rr) == normalize_relation(tr).

    Returns the headline (substring) recall plus an exact-match 'lower_bound'.
    \"\"\"
    try:
        from pipeline.rag.constants import normalize_relation
    except Exception:
        def normalize_relation(r):
            return (r or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "")

    ref_edges = _load_reference_edges_from_json(ref_path)
    n = len(ref_edges)

    # Build extracted set once
    extracted = [(norm(t["subject"]), normalize_relation(t.get("relation","")), norm(t["object"]))
                 for t in triples]

    def match_substring(rs, rr, ro):
        rrn = normalize_relation(rr)
        rs_n, ro_n = norm(rs), norm(ro)
        for ts, tr, to in extracted:
            if tr == rrn and rs_n in ts and ro_n in to:
                return (rs, rr, ro)
        return None

    def match_exact(rs, rr, ro):
        rrn = normalize_relation(rr)
        rs_n, ro_n = norm(rs), norm(ro)
        for ts, tr, to in extracted:
            if tr == rrn and ts == rs_n and to == ro_n:
                return (rs, rr, ro)
        return None

    sub_hits   = [match_substring(s, r, o) for s, r, o in ref_edges]
    sub_hits   = [h for h in sub_hits if h is not None]
    exact_hits = [match_exact(s, r, o)     for s, r, o in ref_edges]
    exact_hits = [h for h in exact_hits if h is not None]

    return {
        "recall":          len(sub_hits) / n if n else 0.0,
        "hits":            len(sub_hits),
        "total_reference": n,
        "matched_edges":   sub_hits,
        "matcher":         "substring_normRel (paper headline)",
        "lower_bound": {
            "recall":          len(exact_hits) / n if n else 0.0,
            "hits":            len(exact_hits),
            "total_reference": n,
            "matched_edges":   exact_hits,
            "matcher":         "exact_normRel (strict tuple equality + relation normalization)",
        },
        "ref_path":        ref_path,
    }
"""

patched = src[:m.start()] + new_block + src[m.end():]
src_path.write_text(patched)
print("Patched recall() in pipeline/07_final_metrics.py.")
print(f"  - Old recall: lines {src[:m.start()].count(chr(10))+1}-{src[:m.end()].count(chr(10))+1}")
print(f"  - New recall: {new_block.count(chr(10))} lines")
