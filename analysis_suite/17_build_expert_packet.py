#!/usr/bin/env python3
"""
17_build_expert_packet.py — Stratified, blinded expert validation packet.
=========================================================================
WHY
    The existing Section 4.4 material cannot validate the pipeline's own
    reliability signals, for two structural reasons:
      1. RANGE RESTRICTION — the annotated items were all Tier-1 /
         STRONG_SUPPORT, so w_tier and w_m4 were constant and only the
         consensus channel varied. A score cannot be tested on a sample
         drawn from one end of its own scale.
      2. NO TRIPLE ATTACHED — the verdicts were recorded against free-text
         statements with no (subject, relation, object), so they cannot be
         joined to any per-triple quantity. Reconstructing that link after
         the fact proved impossible without arbitrary assumptions.
    This script produces a packet that fixes both by construction.

WHAT
    Stratified random sample across tier x confidence bins, with every item
    carrying its triple, and split into two files:

      expert_packet_blind.csv / .md   -> given to the expert. Contains the
          verbalized statement, the supporting quote and its source paper,
          and empty verdict/comment fields. It deliberately does NOT show
          tier, confidence, M4 decision or support counts: if the reviewer
          sees the pipeline's own confidence, their judgement anchors on it
          and any later correlation becomes circular.

      expert_packet_key.csv           -> kept by the researcher. Maps
          item_id back to (subject, relation, object), tier, confidence,
          its components, support counts and M4 decision, plus the stratum
          the item was drawn from.

      expert_packet_design.json       -> sampling design: seed, bin edges,
          population and sample size per stratum, so the design is
          reproducible and reportable in the manuscript.

    A PRIORITY group can be added on top of the stratified sample
    (--priority quarantine,flagged): items that need review regardless of
    sampling. They are tagged as such in the key and EXCLUDED from the
    stratified analysis, so they cannot bias the score validation.

    Verbalization is deliberately literal: "X is described in seismic data
    as Y", never "X may commonly be described as Y". Adding hedges would
    change what the expert is judging.

USAGE
    python analysis_suite/17_build_expert_packet.py \
        --kg output/run13/kg/tiered_kg_run13_enforced.json \
        --decisions output/run13/m4_panel/m4_panel_decisions.jsonl \
        --outdir output/run13/expert_packet \
        --n-per-stratum 5 --seed 42
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kg_io import get_object, get_relation, get_subject, load_kg

# Literal verbalization templates — no hedging, no added modality.
TEMPLATES = {
    "hasdescriptor": "{s} is described in seismic data as {o}.",
    "causes": "{s} causes {o}.",
    "triggers": "{s} triggers {o}.",
    "controls": "{s} controls {o}.",
    "affects": "{s} affects {o}.",
    "occursin": "{s} occurs in {o}.",
    "overlies": "{s} overlies {o}.",
    "underlies": "{s} underlies {o}.",
    "partof": "{s} is part of {o}.",
    "indicates": "{s} indicates {o}.",
    "evidences": "{s} is evidence of {o}.",
    "relatedto": "{s} is related to {o}.",
}

INSTRUCTIONS = """# Expert validation — instructions

You are asked to judge a set of statements extracted from the literature on
mass-transport deposits. Each statement is shown with the passage it was
extracted from and the source article.

For each statement, please give one verdict:

  Y  the statement is geologically correct as written
  P  partially correct — correct only under conditions, or imprecise as
     phrased (please say which condition or which imprecision)
  N  the statement is incorrect

Please add a short comment whenever you answer P or N — the comment is more
valuable to us than the verdict itself.

Notes:
  * Judge the statement as written, not what it could mean if reworded.
  * Statements are literal renderings of extracted relations; awkward
    phrasing is expected and is itself useful information.
  * The statements are presented in random order and carry no indication of
    how confident the system was — this is deliberate.
  * There is no expectation that all statements are correct.

Estimated time: about {minutes} minutes for {n} statements.
"""


def verbalize(s, r, o):
    tpl = TEMPLATES.get(r.strip().lower().replace(" ", "").replace("_", ""))
    if tpl:
        return tpl.format(s=s, o=o)
    return f"{s} — {r} — {o}."


def conf_bin(c, edges):
    for i, e in enumerate(edges):
        if c <= e:
            return f"conf{i + 1}"
    return f"conf{len(edges) + 1}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True,
                    help="KG with provenance/confidence (kg_with_provenance "
                         "or an enforced KG that carries confidence)")
    ap.add_argument("--decisions", default=None,
                    help="m4_panel_decisions.jsonl, for M4 decision + flags")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-per-stratum", type=int, default=5)
    ap.add_argument("--conf-edges", type=float, nargs="*", default=None,
                    help="confidence bin upper edges; default = terciles of "
                         "the observed distribution")
    ap.add_argument("--priority", default="quarantine,flagged",
                    help="comma list among {quarantine,flagged,none}")
    ap.add_argument("--max-priority", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--minutes-per-item", type=float, default=1.5)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    kg = load_kg(args.kg)
    triples = kg["triples"]
    if not any("confidence" in t for t in triples):
        raise SystemExit(
            "No `confidence` field found in the KG. Run "
            "08_rebuild_provenance.py first and point --kg at its "
            "kg_with_provenance.json output.")

    # M4 decisions / flags
    dec = {}
    if args.decisions:
        for line in open(args.decisions, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            k = (str(d.get("subject", "")).lower().strip(),
                 str(d.get("relation", "")).strip(),
                 str(d.get("object", "")).lower().strip())
            dec[k] = d

    def dkey(t):
        return (get_subject(t).lower().strip(), get_relation(t).strip(),
                get_object(t).lower().strip())

    # ── strata ────────────────────────────────────────────────────────
    confs = sorted(t.get("confidence", 0.0) for t in triples)
    if args.conf_edges:
        edges = list(args.conf_edges)
    else:
        n = len(confs)
        edges = [confs[n // 3], confs[2 * n // 3]]
    edges = sorted(set(edges))

    pop = defaultdict(list)
    priority = []
    prio_modes = {m.strip() for m in args.priority.split(",")}
    for t in triples:
        d = dec.get(dkey(t), {})
        flags = str(d.get("flags_any_judge") or d.get("flags") or "")
        is_quar = t.get("_status") == "quarantine"
        is_flag = "parametric_risk" in flags
        if (("quarantine" in prio_modes and is_quar)
                or ("flagged" in prio_modes and is_flag)):
            priority.append((t, d, "quarantine" if is_quar
                             else "parametric_risk"))
            continue
        tier = t.get("_tier", 0)
        stratum = f"tier{tier}_{conf_bin(t.get('confidence', 0.0), edges)}"
        pop[stratum].append((t, d))

    rng.shuffle(priority)
    priority = priority[:args.max_priority]

    sample, design = [], {}
    for stratum in sorted(pop):
        items = pop[stratum][:]
        rng.shuffle(items)
        take = items[:args.n_per_stratum]
        design[stratum] = {"population": len(items), "sampled": len(take)}
        for t, d in take:
            sample.append((t, d, stratum))

    for t, d, why in priority:
        sample.append((t, d, f"priority:{why}"))
    design["priority"] = {"population": len(priority), "sampled":
                          len(priority), "note": "reviewed regardless of "
                          "sampling; excluded from score-validation "
                          "analyses"}

    rng.shuffle(sample)  # random presentation order (blinding)

    # ── outputs ───────────────────────────────────────────────────────
    blind_rows, key_rows = [], []
    for i, (t, d, stratum) in enumerate(sample, 1):
        s, r, o = get_subject(t), get_relation(t), get_object(t)
        item_id = f"S{i:03d}"
        quote = str(t.get("evidence") or "").strip().strip('"')
        if len(quote) > 700:
            quote = quote[:700] + " […]"
        blind_rows.append({
            "item_id": item_id,
            "statement": verbalize(s, r, o),
            "supporting_passage": quote,
            "source_paper": t.get("evidence_paper", ""),
            "verdict_YPN": "",
            "comment": "",
        })
        cc = t.get("conf_components", {}) or {}
        key_rows.append({
            "item_id": item_id, "stratum": stratum,
            "subject": s, "relation": r, "object": o,
            "tier": t.get("_tier"), "status": t.get("_status"),
            "confidence": t.get("confidence"),
            "w_tier": cc.get("w_tier"), "w_m4": cc.get("w_m4"),
            "w_consensus": cc.get("w_consensus"),
            "support_papers": t.get("support_papers"),
            "support_chunks": t.get("support_chunks"),
            "m4_decision": d.get("m4_decision", ""),
            "m4_confidence": d.get("m4_confidence", ""),
            "flags": d.get("flags_any_judge", d.get("flags", "")),
            "evidence_paper": t.get("evidence_paper", ""),
        })

    with open(outdir / "expert_packet_blind.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(blind_rows[0].keys()))
        w.writeheader()
        w.writerows(blind_rows)

    with open(outdir / "expert_packet_key.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()),
                           restval="")
        w.writeheader()
        w.writerows(key_rows)

    minutes = int(round(len(blind_rows) * args.minutes_per_item))
    with open(outdir / "expert_packet_blind.md", "w",
              encoding="utf-8") as f:
        f.write(INSTRUCTIONS.format(minutes=minutes, n=len(blind_rows)))
        f.write("\n---\n\n")
        for r_ in blind_rows:
            f.write(f"### {r_['item_id']}\n\n**{r_['statement']}**\n\n")
            if r_["supporting_passage"]:
                f.write(f"> {r_['supporting_passage']}\n\n")
            if r_["source_paper"]:
                f.write(f"*Source: {r_['source_paper']}*\n\n")
            f.write("Verdict (Y / P / N): ______   Comment: "
                    "_________________________________\n\n---\n\n")

    design_out = {
        "seed": args.seed, "n_per_stratum": args.n_per_stratum,
        "confidence_bin_edges": edges,
        "n_items": len(blind_rows),
        "estimated_minutes": minutes,
        "strata": design,
        "blinding": "The expert sheet shows statement, passage and source "
                    "only. Tier, confidence, support counts and M4 verdicts "
                    "are withheld to keep the later score-validation "
                    "non-circular; the mapping is in expert_packet_key.csv.",
        "verbalization": "Literal templates, no hedging added.",
    }
    with open(outdir / "expert_packet_design.json", "w",
              encoding="utf-8") as f:
        json.dump(design_out, f, indent=2, ensure_ascii=False)

    print("=" * 62)
    print("EXPERT VALIDATION PACKET")
    print("=" * 62)
    print(f"confidence bin edges : {edges}")
    for stratum in sorted(design):
        d_ = design[stratum]
        print(f"  {stratum:22s} population {d_['population']:4d} -> "
              f"sampled {d_['sampled']}")
    print(f"\nitems total          : {len(blind_rows)} "
          f"(~{minutes} min for the reviewer)")
    tiers = Counter(k["tier"] for k in key_rows)
    print(f"tier spread in sample: {dict(tiers)}")
    cvals = [k["confidence"] for k in key_rows
             if isinstance(k["confidence"], (int, float))]
    if cvals:
        print(f"confidence spread    : {min(cvals):.3f} – {max(cvals):.3f}"
              "   (must span the scale, otherwise the score stays "
              "untestable)")
    print(f"\nto send   : {outdir}/expert_packet_blind.md (or .csv)")
    print(f"keep back : {outdir}/expert_packet_key.csv")


if __name__ == "__main__":
    main()