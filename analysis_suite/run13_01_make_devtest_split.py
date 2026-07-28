#!/usr/bin/env python3
"""
run13_01_make_devtest_split.py — Freeze the evaluation protocol BEFORE run13.
=============================================================================
WHY (D1, review objection #1)
    The 60 rescue queries were designed while observing benchmark misses,
    exposing the headline recall to a test-set-tuning objection. Full
    historical remediation is impossible; the strongest feasible protocol is:
      (1) the query set is FROZEN as of run11 (no further edits),
      (2) the 34-edge benchmark is split ONCE, randomly and stratified,
          into dev (17) and test (17), BEFORE run13 is launched,
      (3) run13 recall is reported separately on dev and test, plus on
          original-26 vs extended-8,
      (4) this file writes a signed protocol declaration (timestamp + SHA256
          of the split files + explicit statement) that goes into the repo.
    Because the split is random and stratified by (source, relation family),
    any residual rescue-query advantage distributes evenly across dev/test;
    a dev-test recall gap therefore measures that advantage directly —
    which is itself a reportable number.

WHAT
    configs/lb_dev.json, configs/lb_test.json  — same schema as
        lb_extended_benchmark.json, directly usable by 07_final_metrics.py
        via --ref (NO reimplementation of the matching: numbers stay
        comparable with run11).
    configs/run13_protocol_declaration.json    — the frozen statement.

ASSUMPTION (verify once): in lb_extended_benchmark.json the first
    `original_count` edges are the 26 original LB2019 edges and the
    remaining 8 are the corpus-grounded extensions. If a per-edge marker
    exists instead, pass --marker-key.

USAGE
    python run13_01_make_devtest_split.py \
        --benchmark configs/lb_extended_benchmark.json \
        --outdir configs [--seed 20260727]
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REL_FAMILY = {
    "hasdescriptor": "descriptor",
    "causes": "causal",
    "triggers": "causal",
    "controls": "causal",
    "affects": "causal",
    "formedby": "causal",
    "occursin": "context",
    "overlies": "context",
    "underlies": "context",
    "partof": "context",
    "indicates": "context",
    "evidences": "context",
    "relatedto": "context",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--outdir", default="configs")
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument(
        "--marker-key",
        default=None,
        help="per-edge key marking original vs extended, if any",
    )
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bench = json.load(open(args.benchmark, encoding="utf-8"))
    edges = bench["edges"]
    n_orig = int(bench.get("original_count", 26))

    # tag source
    for i, e in enumerate(edges):
        if args.marker_key and args.marker_key in e:
            e["_source"] = str(e[args.marker_key])
        else:
            e["_source"] = "original" if i < n_orig else "extended"
        fam = REL_FAMILY.get(e["relation"].lower(), "context")
        e["_stratum"] = f"{e['_source']}|{fam}"

    # stratified 50/50 split
    rng = random.Random(args.seed)
    strata = defaultdict(list)
    for e in edges:
        strata[e["_stratum"]].append(e)
    dev, test = [], []
    for key in sorted(strata):
        group = strata[key][:]
        rng.shuffle(group)
        half = len(group) // 2
        extra = len(group) % 2
        # alternate the odd element deterministically by stratum order
        if extra and (len(dev) <= len(test)):
            dev.extend(group[: half + 1])
            test.extend(group[half + 1 :])
        else:
            dev.extend(group[:half])
            test.extend(group[half:])

    def dump(split, name):
        payload = {
            "description": (
                f"run13 frozen {name} split of the 34-edge "
                f"benchmark (seed {args.seed}, stratified by "
                "source x relation-family). Query set frozen "
                "as of run11; split drawn before run13 launch."
            ),
            "seed": args.seed,
            "n_edges": len(split),
            "edges": [
                {k: v for k, v in e.items() if not k.startswith("_")}
                for e in split
            ],
            "strata": sorted({e["_stratum"] for e in split}),
        }
        p = outdir / f"lb_{name}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return p

    p_dev = dump(dev, "dev")
    p_test = dump(test, "test")

    decl = {
        "protocol": "run13 evaluation protocol — FROZEN",
        "declared_at_utc": datetime.now(timezone.utc).isoformat(),
        "statement": (
            "The 249-query set is frozen as of run11 and will not be "
            "modified for run13. The 34-edge benchmark is split into dev "
            "and test before run13 execution. Headline recall will be "
            "reported on the test split; dev recall and the dev-test gap "
            "will be reported alongside as a measure of residual "
            "query-design adaptation. Original-26 vs extended-8 recall "
            "will also be reported separately."
        ),
        "seed": args.seed,
        "benchmark_sha256": sha256(Path(args.benchmark)),
        "dev_file": str(p_dev),
        "dev_sha256": sha256(p_dev),
        "dev_n": len(dev),
        "test_file": str(p_test),
        "test_sha256": sha256(p_test),
        "test_n": len(test),
        "dev_strata": sorted({e["_stratum"] for e in dev}),
        "test_strata": sorted({e["_stratum"] for e in test}),
    }
    p_decl = outdir / "run13_protocol_declaration.json"
    with open(p_decl, "w", encoding="utf-8") as f:
        json.dump(decl, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("PROTOCOL FROZEN")
    print("=" * 60)
    print(f"dev : {len(dev)} edges -> {p_dev}")
    print(f"test: {len(test)} edges -> {p_test}")
    print(f"declaration: {p_decl}")
    print("Commit these three files BEFORE launching run13:")
    print(
        f"  git add {p_dev} {p_test} {p_decl} && "
        f"git commit -m 'run13: freeze eval protocol (dev/test split)'"
    )


if __name__ == "__main__":
    main()
