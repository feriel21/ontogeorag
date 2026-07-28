#!/usr/bin/env python3
"""
validate_pipeline.py — Stage-by-stage validation harness for OntoGeoRAG.
========================================================================
Runs every pipeline stage in smoke mode (small inputs / --limit / --max-queries)
plus integrity checks on existing artifacts, captures logs, and writes
`validation_report.md` with per-stage status (OK / FAIL / SKIP / KNOWN_ISSUE),
log tails, detected causes, and recommendations.

Modes:
  --quick : CPU-only — artifact integrity + CPU stages (04c, 05a, 06, 06b,
            07, analysis suite loaders). No model loading. ~2 min.
  --gpu   : additionally smoke-tests 02 (--max-queries 3), 03 (5-triple
            sample) and m4_verify (--limit 3). Needs a GPU allocation and
            resolved local model paths. ~10 min.

Usage (from ~/ontogeorag):
    python validate_pipeline.py --quick
    python validate_pipeline.py --gpu
Report: validation_report.md (+ logs under validation_logs/)
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path.cwd()
LOGDIR = REPO / "validation_logs"
RESULTS = []
RUN13 = REPO / "output/run13"
SMOKE = REPO / "output/validation_smoke"


def record(stage, status, detail="", log="", reco=""):
    RESULTS.append({"stage": stage, "status": status, "detail": detail,
                    "log": log, "reco": reco})
    print(f"[{status:<11}] {stage} — {detail}")


def run(stage, cmd, timeout=900, env_extra=None, reco_on_fail=""):
    LOGDIR.mkdir(exist_ok=True)
    logfile = LOGDIR / (stage.replace(" ", "_").replace("/", "_") + ".log")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["HF_HUB_OFFLINE"] = "1"
    for k in ("SLURM_PROCID", "SLURM_NTASKS", "RANK", "WORLD_SIZE",
              "MASTER_ADDR", "MASTER_PORT"):
        env.pop(k, None)
    if env_extra:
        env.update(env_extra)
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, env=env, cwd=REPO)
        logfile.write_text(p.stdout + "\n--- STDERR ---\n" + p.stderr)
        tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-6:])
        if p.returncode == 0:
            record(stage, "OK", f"exit 0", str(logfile))
            return True
        record(stage, "FAIL", f"exit {p.returncode}", str(logfile),
               reco_on_fail or f"see {logfile}; tail:\n{tail}")
    except subprocess.TimeoutExpired:
        record(stage, "FAIL", f"timeout {timeout}s", str(logfile),
               reco_on_fail)
    return False


def local_model(hub_id):
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(hub_id, local_files_only=True)
    except Exception:
        return None


# ── artifact integrity checks (always run) ────────────────────────────

def check_index():
    p = RUN13 / "step1/chunks.jsonl"
    if not p.exists():
        record("01 index artifact", "FAIL", f"{p} missing", "",
               "run run13_build_index_from_run11.py")
        return
    papers, hashes, dup = set(), set(), 0
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        papers.add(r["doc_id"])
        h = hashlib.md5((r["doc_id"] + r["text"]).encode()).hexdigest()
        dup += h in hashes
        hashes.add(h)
    ok = dup == 0 and len(papers) == 37
    record("01 index artifact", "OK" if ok else "FAIL",
           f"papers={len(papers)}, chunks={len(hashes)}, dup={dup}",
           reco="" if ok else "rebuild index (expect 37 papers, 0 dup)")


def check_pdf_parser():
    record("01 PDF parser", "KNOWN_ISSUE",
           "memory leak (~1.7GB/3s) in current env; run13 bypassed via "
           "chunk-level dedup of run11 index",
           reco="pin/repair the PDF parsing dependency before any corpus "
                "extension; document versions with pip freeze")


def check_kg_artifacts():
    for name, path in [
        ("06 fused KG", RUN13 / "kg/tiered_kg_run13.json"),
        ("06b provenance KG", RUN13 / "kg/tiered_kg_run13_prov.json"),
        ("M4 panel KG", RUN13 / "m4_panel/tiered_kg_m4.json"),
        ("final enforced KG", RUN13 / "kg/tiered_kg_run13_enforced.json"),
    ]:
        if not path.exists():
            record(name, "FAIL", f"{path} missing")
            continue
        try:
            sys.path.insert(0, str(REPO / "analysis_suite"))
            from kg_io import load_kg
            kg = load_kg(path)
            n = len(kg["triples"])
            record(name, "OK", f"loads via kg_io, {n} triples")
        except Exception as e:
            record(name, "FAIL", f"load error: {e}")


def check_schema_types():
    path = RUN13 / "kg/tiered_kg_run13_enforced.json"
    if not path.exists():
        record("04b schema enforcement", "FAIL", "enforced KG missing")
        return
    d = json.load(open(path))
    types = set()
    for k in ("tier1", "tier2", "quarantine"):
        for t in d.get(k, []):
            for f in ("subject_type", "object_type"):
                if t.get(f):
                    types.add(t[f])
    if len(types) <= 7:
        record("04b schema enforcement", "OK", f"{len(types)} types")
    else:
        record("04b schema enforcement", "KNOWN_ISSUE",
               f"{len(types)} entity types present (schema declares 5); "
               "04b script not yet deployed",
               reco="recreate 04b (TYPE_MAP 25->5, length filter, relation "
                    "signatures, self-loops) and re-run FROM_STAGE=8")


def check_protocol():
    ok = all((REPO / f"configs/{f}").exists() for f in
             ("lb_dev.json", "lb_test.json",
              "run13_protocol_declaration.json"))
    record("frozen eval protocol", "OK" if ok else "FAIL",
           "dev/test/declaration committed" if ok else "protocol files "
           "missing")


# ── CPU stage smoke tests ─────────────────────────────────────────────

def smoke_cpu():
    SMOKE.mkdir(parents=True, exist_ok=True)
    pa = RUN13 / "pass_a/canonical_triples_v5.jsonl"
    if pa.exists():
        run("05a rule-canonicalize (smoke)",
            f"python analysis_suite/05a_rule_canonicalize.py --input {pa}")
        run("04c lexicon guard (smoke)",
            f"python analysis_suite/04c_lexicon_enforce.py --input {pa} "
            "--report")
    else:
        record("05a/04c", "SKIP", "pass_a canonical file missing")
    kgp = RUN13 / "kg/tiered_kg_run13_prov.json"
    if kgp.exists():
        run("07 metrics (full34)",
            f"python pipeline/07_final_metrics.py --kg {kgp} "
            f"--output {SMOKE}/metrics_check.json")
        run("08 provenance (analysis)",
            f"cd analysis_suite && python 08_rebuild_provenance.py "
            f"--kg {kgp} --chunks {RUN13}/step1/chunks.jsonl "
            f"--outdir {SMOKE}/analysis")
    else:
        record("07/08", "SKIP", "prov KG missing")


# ── GPU stage smoke tests ─────────────────────────────────────────────

def smoke_gpu():
    qwen = local_model("Qwen/Qwen2.5-7B-Instruct")
    llama = local_model("meta-llama/Llama-3.1-8B-Instruct")
    if not qwen:
        record("02/03 smoke", "SKIP", "Qwen snapshot not in local cache")
    else:
        SMOKE.mkdir(parents=True, exist_ok=True)
        ok = run("02 extraction (smoke, 3 queries)",
                 f"python -u pipeline/02_extract_triples.py "
                 f"--index-dir {RUN13}/step1 "
                 f"--schema configs/ontology_schema.json "
                 f"--queries configs/descriptor_queries.jsonl "
                 f"--output {SMOKE}/raw_smoke.jsonl --model {qwen} "
                 f"--backend hf --top-k 5 --bm25-topn 20 --min-bm25 2.0 "
                 f"--temperature 0.0 "
                 f"--reranker cross-encoder/ms-marco-MiniLM-L-6-v2 "
                 f"--max-queries 3", timeout=1200,
                 reco_on_fail="check GPU allocation and local model path")
        if ok:
            run("03 verification (smoke)",
                f"python -u pipeline/03_verify_triples.py "
                f"--input {SMOKE}/raw_smoke.jsonl "
                f"--output {SMOKE}/verified_smoke.jsonl "
                f"--model {qwen} --backend hf", timeout=1200)
    kgp = RUN13 / "kg/tiered_kg_run13_prov.json"
    if llama and kgp.exists():
        run("M4 verify (smoke, 3 triples)",
            f"cd ~/m4_verifier && python m4_verify.py --kg {kgp} "
            f"--index {RUN13}/step1 --output {SMOKE}/m4 "
            f"--model {llama} --limit 3", timeout=1200)
    else:
        record("M4 smoke", "SKIP", "Llama snapshot or prov KG missing")


def write_report():
    lines = ["# Pipeline Validation Report",
             f"*Generated {datetime.datetime.now().isoformat(timespec='seconds')} on {os.uname().nodename}*",
             "", "| Stage | Status | Detail |", "|---|---|---|"]
    for r in RESULTS:
        lines.append(f"| {r['stage']} | **{r['status']}** | {r['detail']} |")
    lines.append("")
    fails = [r for r in RESULTS if r["status"] in ("FAIL", "KNOWN_ISSUE")]
    if fails:
        lines.append("## Failures / known issues & recommendations")
        for r in fails:
            lines.append(f"### {r['stage']} — {r['status']}")
            lines.append(f"- Detail: {r['detail']}")
            if r["log"]:
                lines.append(f"- Log: `{r['log']}`")
            if r["reco"]:
                lines.append(f"- Recommendation: {r['reco']}")
            lines.append("")
    Path("validation_report.md").write_text("\n".join(lines),
                                            encoding="utf-8")
    n_ok = sum(1 for r in RESULTS if r["status"] == "OK")
    print(f"\nReport: validation_report.md — {n_ok}/{len(RESULTS)} OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    if not (args.quick or args.gpu):
        args.quick = True

    check_protocol()
    check_index()
    check_pdf_parser()
    check_kg_artifacts()
    check_schema_types()
    smoke_cpu()
    if args.gpu:
        smoke_gpu()
    write_report()


if __name__ == "__main__":
    main()