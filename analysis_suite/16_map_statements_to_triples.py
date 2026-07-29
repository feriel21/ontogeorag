#!/usr/bin/env python3
"""
16_map_statements_to_triples.py — Rebuild the statement -> triple mapping.
==========================================================================
WHY
    The Section 4.4 expert file (expert_labels_statements.csv) stores 29
    hand-written geological statements with Y/P/N verdicts, but NOT the
    triples they were derived from. `output/expert_annotation_protocol.json`
    turned out to hold a DIFFERENT, machine-phrased statement set (0/28 text
    matches), and `reference/expert_labels.csv` (the format m4_metrics.py
    documents) does not exist. Without the mapping, the expert verdicts —
    the most expensive data in the project — cannot be joined to anything.

    A silent automatic mapping is not acceptable: an earlier id-based join
    produced a plausible-looking but entirely scrambled result. This script
    therefore PROPOSES candidates and requires human confirmation.

WHAT
    For every statement, scores every KG triple on whether its subject and
    object surface forms occur in the statement text, with morphological
    tolerance (plural, hyphen/space, MTD/MTC abbreviations, OCR ligatures)
    and a small relation-verb bonus ("causes", "triggers", "occurs in",
    "characterized by/displays/exhibits" -> hasDescriptor).

    Outputs (in --outdir):
      statement_triple_candidates.csv   one row per statement: top-3
          candidates with scores + an EMPTY `confirmed_*` column set to fill
      statement_triple_candidates.md    same, readable for eyeball review
      mapping_review_stats.json         how many statements got a confident
          top candidate, how many are ambiguous

    Workflow:
      1. run this script
      2. open the CSV, fill `confirmed_subject/relation/object` (copy from
         the proposed columns when correct, correct them when not, leave
         blank when no triple applies)
      3. feed the completed CSV to 15_confidence_validation.py --experts
         (it has subject/relation/object columns, so no --protocol needed)

USAGE
    python analysis_suite/16_map_statements_to_triples.py \
        --statements m4/expert_labels_statements.csv \
        --kg output/run11_kg/tiered_kg_run11.json \
        --outdir output/run11_kg/e2_mapping
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kg_io import get_object, get_relation, get_subject, load_kg

ABBREV = {
    "mtd": "mass transport deposit",
    "mtds": "mass transport deposit",
    "mtc": "mass transport complex",
    "mtcs": "mass transport complex",
}
LIGATURES = {"\ufb01": "fi", "\ufb02": "fl", "\u2019": "'", "\u2013": "-"}

# Reverse map: a statement may use the abbreviation while the KG entity is
# spelled out (observed: "MTDs can exhibit..." vs entity "mass transport
# deposit"). Both directions must be searched.
ABBREV_REV = {}
for _abbr, _full in ABBREV.items():
    ABBREV_REV.setdefault(_full, set()).add(_abbr)

# relation -> verbs/phrases that signal it in natural language
REL_HINTS = {
    "hasdescriptor": ["characteri", "display", "exhibit", "show", "appear",
                      "is described", "facies", "seismic"],
    "causes": ["cause", "produce", "result in", "lead to", "generate",
               "form"],
    "triggers": ["trigger", "initiate", "induce", "set off"],
    "controls": ["control", "govern", "modulate", "depend"],
    "occursin": ["occur", "found in", "located", "situated", "setting"],
    "overlies": ["overli", "above", "on top of"],
    "underlies": ["underli", "below", "beneath"],
    "partof": ["part of", "component", "belongs"],
    "affects": ["affect", "influence", "impact"],
}


def clean(s):
    s = str(s or "")
    for bad, good in LIGATURES.items():
        s = s.replace(bad, good)
    s = s.lower().replace("-", " ").replace("_", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def variants(entity):
    """Surface variants of an entity to look for in a statement."""
    e = clean(entity)
    out = {e}
    if e in ABBREV:                     # entity is an abbreviation
        out.add(clean(ABBREV[e]))
    for ab in ABBREV_REV.get(e, ()):    # entity is the expanded form
        out.add(ab)
    # singular/plural of the head word and of the whole phrase
    toks = e.split()
    if toks:
        if toks[-1].endswith("s") and len(toks[-1]) > 3:
            out.add(" ".join(toks[:-1] + [toks[-1][:-1]]))
        else:
            out.add(" ".join(toks[:-1] + [toks[-1] + "s"]))
    # for multiword entities, the head noun alone is a weak variant
    if len(toks) > 1:
        out.add(toks[-1])
    return {v for v in out if len(v) >= 3}


def match_score(entity, statement_clean):
    """0 = absent, 1 = full phrase present, 0.6 = plural/abbrev variant,
    0.3 = head noun only."""
    e = clean(entity)
    if not e:
        return 0.0
    if e in statement_clean:
        return 1.0
    best = 0.0
    toks = e.split()
    for v in variants(entity):
        if v in statement_clean:
            if v == e:
                best = max(best, 1.0)
            elif len(toks) > 1 and v == toks[-1]:
                best = max(best, 0.3)
            else:
                best = max(best, 0.6)
    return best


def rel_bonus(relation, statement_clean):
    hints = REL_HINTS.get(clean(relation).replace(" ", ""), [])
    return 0.15 if any(h in statement_clean for h in hints) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--statements", required=True)
    ap.add_argument("--kg", required=True)
    ap.add_argument("--outdir", default="output/run11_kg/e2_mapping")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--confident-threshold", type=float, default=1.6,
                    help="score above which the top candidate is called "
                         "confident (subject+object both fully present)")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.statements, newline="", encoding="utf-8") as f:
        stmts = list(csv.DictReader(f))
    cols = list(stmts[0].keys())
    scol = next((c for c in cols if "statement" in c.lower()
                 and "id" not in c.lower()), None)
    idcol = next((c for c in cols if c.lower() in
                  ("statement_id", "id")), None)
    if not scol:
        raise SystemExit(f"No statement-text column found in {cols}")

    kg = load_kg(args.kg)
    triples = [(get_subject(t), get_relation(t), get_object(t),
                t.get("_tier"))
               for t in kg["triples"] if get_subject(t) and get_object(t)]
    print(f"[map] {len(stmts)} statements x {len(triples)} triples")

    rows, n_conf, n_amb, n_none = [], 0, 0, 0
    for i, st in enumerate(stmts, 1):
        text = clean(st[scol])
        scored = []
        for s, r, o, tier in triples:
            ss, os_ = match_score(s, text), match_score(o, text)
            if ss == 0 or os_ == 0:
                continue          # both ends must be present at all
            scored.append((ss + os_ + rel_bonus(r, text), s, r, o, tier,
                           ss, os_))
        partial = False
        if not scored:
            # No triple has BOTH ends in the statement. Rather than
            # returning nothing (which hides usable information from the
            # reviewer), propose single-end matches, clearly marked.
            for s, r, o, tier in triples:
                ss, os_ = match_score(s, text), match_score(o, text)
                if ss == 0 and os_ == 0:
                    continue
                scored.append((ss + os_ + rel_bonus(r, text), s, r, o, tier,
                               ss, os_))
            partial = bool(scored)
        scored.sort(key=lambda x: -x[0])
        top = scored[:args.topk]
        row = {"statement_id": st.get(idcol, i) if idcol else i,
               "statement": st[scol]}
        for k, v in st.items():
            if "verdict" in k.lower() or "comment" in k.lower():
                row[k] = v
        for j in range(args.topk):
            p = f"cand{j+1}_"
            if j < len(top):
                sc, s, r, o, tier, ss, os_ = top[j]
                row[p + "subject"] = s
                row[p + "relation"] = r
                row[p + "object"] = o
                row[p + "tier"] = tier
                row[p + "score"] = round(sc, 2)
                row[p + "subj_match"] = ss
                row[p + "obj_match"] = os_
            else:
                for suffix in ("subject", "relation", "object", "tier",
                               "score", "subj_match", "obj_match"):
                    row[p + suffix] = ""
        # verdict on the proposal itself
        if not top:
            row["status"] = "NO_CANDIDATE"
            n_none += 1
        elif partial:
            row["status"] = "PARTIAL_ONLY"
            n_amb += 1
        elif top[0][0] >= args.confident_threshold and (
                len(top) == 1 or top[0][0] - top[1][0] >= 0.3):
            row["status"] = "CONFIDENT"
            n_conf += 1
        else:
            row["status"] = "AMBIGUOUS"
            n_amb += 1
        # to be filled by the human
        row["confirmed_subject"] = ""
        row["confirmed_relation"] = ""
        row["confirmed_object"] = ""
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(outdir / "statement_triple_candidates.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(rows)

    with open(outdir / "statement_triple_candidates.md", "w",
              encoding="utf-8") as f:
        f.write("# Statement -> triple candidates (to be confirmed)\n\n")
        f.write("Fill `confirmed_subject/relation/object` in the CSV. "
                "Copy from a candidate when correct; correct it when not; "
                "leave blank when no triple applies.\n\n")
        for r in rows:
            f.write(f"## [{r['status']}] #{r['statement_id']}\n")
            f.write(f"*{r['statement']}*\n\n")
            for j in range(args.topk):
                p = f"cand{j+1}_"
                if r.get(p + "subject"):
                    f.write(f"- {j+1}. `{r[p+'subject']}` "
                            f"--[{r[p+'relation']}]--> `{r[p+'object']}` "
                            f"(T{r[p+'tier']}, score {r[p+'score']}, "
                            f"subj {r[p+'subj_match']}, "
                            f"obj {r[p+'obj_match']})\n")
            if r["status"] == "NO_CANDIDATE":
                f.write("- *(no triple shares any entity with this "
                        "statement)*\n")
            elif r["status"] == "PARTIAL_ONLY":
                f.write("- *(no triple has BOTH ends present; the above "
                        "are single-end matches — check whether the "
                        "statement paraphrases one of them)*\n")
            f.write("\n")

    stats = {"n_statements": len(rows), "confident": n_conf,
             "ambiguous": n_amb, "no_candidate": n_none,
             "n_triples_searched": len(triples)}
    with open(outdir / "mapping_review_stats.json", "w",
              encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("=" * 60)
    print("STATEMENT -> TRIPLE MAPPING PROPOSALS")
    print("=" * 60)
    print(f"CONFIDENT   : {n_conf}")
    print(f"AMBIGUOUS   : {n_amb}   (need a human decision, incl. "
          "PARTIAL_ONLY)")
    print(f"NO_CANDIDATE: {n_none}  (statement may not come from the KG)")
    print(f"\nReview: {outdir}/statement_triple_candidates.md")
    print(f"Fill in : {outdir}/statement_triple_candidates.csv "
          "(confirmed_* columns)")
    print("Then run 15_confidence_validation.py --experts <that csv>")


if __name__ == "__main__":
    main()