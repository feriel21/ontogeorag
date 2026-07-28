import argparse, json
from pathlib import Path
from audit_lb2019_recoverability import load_chunks, patterns_for, matches_in_chunk
ap = argparse.ArgumentParser()
ap.add_argument("--schema", default="./schema_generated/lb2019_schema_generated.json")
ap.add_argument("--chunks", required=True)
a = ap.parse_args()
schema = json.loads(Path(a.schema).read_text(encoding="utf-8"))
chunks = load_chunks(Path(a.chunks))
zero, found = [], []
for label, v in schema["nodes"].items():
    pats = patterns_for(label)
    ndocs = len({c["doc"] for c in chunks if matches_in_chunk(pats, c["text"])})
    (found if ndocs else zero).append((ndocs, label, v["role"]))
print(f"\nnoeuds: {len(schema['nodes'])} | trouves >=1 doc: {len(found)} | JAMAIS trouves: {len(zero)}")
print("\n--- NOEUDS A 0 OCCURRENCE (suspects = matcher rate) ---")
for _, label, role in sorted(zero, key=lambda x: x[2]):
    print(f"  [{role[:4]}] {label}")
