#!/usr/bin/env python3
"""
m4_inspection_list.py — Priority inspection list for expert validation
========================================================================

Builds the stratified inspection list for the domain expert (Antoine):

  Group A — parametric_risk : blind PLAUSIBLE + evidence NOT_SUPPORTED
            (the triple "sounds right" but the passage does not state it —
             highest-priority error candidates)
  Group B — hasDescriptor UNCERTAIN : the relation that feeds Part II,
            demoted by the independent verifier

For each triple, the sheet shows side by side:
  * the triple, its tier (original + new), the Qwen verdict
  * the BLIND judge's reasoning (what makes it sound plausible)
  * the EVIDENCE judge's quote + reasoning (why the text does not support it)
  * the full source passage

Outputs:
  m4_inspection_list.md    human-readable review sheet
  m4_inspection_list.csv   same content, one row per triple, with an empty
                           `expert_verdict` column (Y/P/N) and `expert_comment`
                           column — fill and feed back to m4_metrics.py
                           (after joining to triple keys).

Usage:
    python m4_inspection_list.py \
        --decisions ~/ontogeorag/output/m4/m4_decisions.jsonl \
        --verdicts  ~/ontogeorag/output/m4/m4_verdicts.jsonl \
        --kg        ~/ontogeorag/output/run11_kg/tiered_kg_run11.json \
        --output    ~/ontogeorag/output/m4
"""

import argparse
import csv
import json
from pathlib import Path

from m4_verify import load_triples, triple_fields, evidence_from_provenance


def load_jsonl(path):
    out = []
    with open(Path(path).expanduser(), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def norm_key(s, r, o):
    return (s.strip().lower(), r.strip(), o.strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--kg", required=True,
                    help="Original tiered KG (for source passages)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions = load_jsonl(args.decisions)
    verdicts = {v["m4_index"]: v for v in load_jsonl(args.verdicts)}

    # passages from the original KG (provenance-embedded)
    passages = {}
    for t in load_triples(Path(args.kg).expanduser()):
        s, r, o = triple_fields(t)
        passages[norm_key(s, r, o)] = evidence_from_provenance(t)

    group_a = [d for d in decisions if "parametric_risk" in d.get("flags", [])]
    group_b = [d for d in decisions
               if d["relation"] == "hasDescriptor"
               and d["m4_decision"] == "UNCERTAIN"
               and "parametric_risk" not in d.get("flags", [])]

    md_path = out_dir / "m4_inspection_list.md"
    csv_path = out_dir / "m4_inspection_list.csv"

    def write_group(fmd, writer, title, note, items, prefix):
        fmd.write(f"\n## {title}\n\n{note}\n")
        for j, d in enumerate(items, 1):
            v = verdicts.get(d["m4_index"], {})
            blind = v.get("blind", {})
            evid = v.get("evidence", {})
            passage = passages.get(
                norm_key(d["subject"], d["relation"], d["object"]), "")

            fmd.write(f"\n---\n\n### {prefix}{j}  "
                      f"({d['subject']}, {d['relation']}, {d['object']})\n\n")
            fmd.write(f"- **Tier**: {d.get('tier')} | "
                      f"**Qwen**: {d.get('qwen_verdict')} | "
                      f"**M4**: blind={d['blind_verdict']}, "
                      f"evidence={d['evidence_verdict']}, "
                      f"decision={d['m4_decision']} "
                      f"(conf {d['m4_confidence']})\n\n")
            fmd.write(f"**Blind judge (plausibility, no text)**\n\n"
                      f"> {blind.get('reasoning', '(none)')}\n\n")
            fmd.write(f"**Evidence judge (textual support)**\n\n"
                      f"> Quote: {evid.get('quote', '(none)')}\n>\n"
                      f"> Reasoning: {evid.get('reasoning', '(none)')}\n\n")
            fmd.write(f"**Source passage**\n\n"
                      f"```\n{(passage or '(passage not embedded)')[:1500]}\n```\n\n")
            fmd.write("**Expert verdict (Y/P/N)**: ______   "
                      "**Comment**: ______________________\n")

            writer.writerow({
                "group": prefix.rstrip("-"),
                "id": f"{prefix}{j}",
                "subject": d["subject"],
                "relation": d["relation"],
                "object": d["object"],
                "tier": d.get("tier"),
                "qwen_verdict": d.get("qwen_verdict"),
                "blind_verdict": d["blind_verdict"],
                "evidence_verdict": d["evidence_verdict"],
                "m4_decision": d["m4_decision"],
                "m4_confidence": d["m4_confidence"],
                "blind_reasoning": blind.get("reasoning", ""),
                "evidence_quote": evid.get("quote", ""),
                "evidence_reasoning": evid.get("reasoning", ""),
                "source_passage": (passage or "")[:2000],
                "expert_verdict": "",
                "expert_comment": "",
            })

    fields = ["group", "id", "subject", "relation", "object", "tier",
              "qwen_verdict", "blind_verdict", "evidence_verdict",
              "m4_decision", "m4_confidence", "blind_reasoning",
              "evidence_quote", "evidence_reasoning", "source_passage",
              "expert_verdict", "expert_comment"]

    with open(md_path, "w", encoding="utf-8") as fmd, \
         open(csv_path, "w", encoding="utf-8", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        writer.writeheader()

        fmd.write("# M4 Priority Inspection List — Expert Validation\n\n"
                  "For each triple: assign Y (geologically correct as "
                  "stated), P (partially correct / context-dependent), or "
                  "N (incorrect), and comment. The two machine judgments "
                  "are shown for context; please judge independently of "
                  "them, based on the source passage and your expertise.\n")

        write_group(
            fmd, writer,
            f"Group A — Parametric risk ({len(group_a)} triples)",
            "Blind judge finds these geologically plausible, but the "
            "independent evidence judge finds them NOT supported by their "
            "source passage. Question for the expert: is the relation (a) "
            "true and in the passage (machine error), (b) true but not in "
            "this passage (grounding failure), or (c) not established?",
            group_a, "A-")

        write_group(
            fmd, writer,
            f"Group B — hasDescriptor demoted ({len(group_b)} triples)",
            "Descriptor relations judged UNCERTAIN by the independent "
            "verifier. These feed the Part II candidate-mask, so their "
            "status matters most. Question for the expert: does the "
            "passage establish the descriptor link, and is the descriptor "
            "assignment geologically standard?",
            group_b, "B-")

    print(f"Group A (parametric_risk): {len(group_a)} triples")
    print(f"Group B (hasDescriptor UNCERTAIN): {len(group_b)} triples")
    print(f"Markdown sheet: {md_path}")
    print(f"CSV sheet:      {csv_path}")


if __name__ == "__main__":
    main()