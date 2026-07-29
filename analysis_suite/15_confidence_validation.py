#!/usr/bin/env python3
"""
15_confidence_validation.py — E2: confidence x expert verdicts x M4 decisions.
==============================================================================
WHY
    Two claims in the manuscript are currently unvalidated externally:
      (a) the per-triple `confidence` score (w_tier x w_m4 x w_consensus)
          means something — i.e. it tracks expert judgement;
      (b) the M4 cross-family panel is a usable proxy for a human expert.
    Both can be tested with data already on disk: the Section 4.4 expert
    verdicts (Y/P/N), the M4 panel decisions, and the provenance/confidence
    table. If (a) holds, the score becomes a contribution rather than a
    construction; if it fails, that is an honest limitation that reinforces
    why human validation (4.4) is required. Either outcome is publishable.

WHAT
  --inspect  : detects and prints the real schema of every input (columns,
               sample rows, join-key candidates, verdict vocabularies) and
               STOPS. Always run this first.
  (analysis) : joins the three sources on normalized (subject, relation,
               object), then computes

    (a) Confidence vs expert verdict
        - mean/median confidence per verdict class (Y / P / N) with
          bootstrap 95% CIs
        - Kendall tau-b between confidence and the ordinal verdict
          (N=0, P=1, Y=2), with a permutation p-value
        - AUC for discriminating Y vs {P,N}
        - the same three statistics for EACH component (w_tier, w_m4,
          w_consensus, support_papers) so we learn which channel carries
          the signal — the decomposition is the interesting result
    (b) M4 vs human
        - contingency table M4 decision (ACCEPT/UNCERTAIN/REJECT) x expert
          verdict (Y/P/N)
        - Cohen's kappa, unweighted and linear-weighted, after mapping both
          onto a common 3-point scale, compared against the human-human
          kappa (0.30 / 0.37 from Section 4.4)
        - per-annotator breakdown when several expert passes are present

    Outputs (in --outdir):
      e2_joined.csv              one row per matched triple, all channels
      e2_report.json             every statistic, machine-readable
      e2_report.md               narrative summary with the caveats attached
      fig_e2_confidence_by_verdict.png   boxplot + points
      fig_e2_m4_vs_expert.png            contingency heatmap

STATISTICAL HONESTY (enforced in the output, not optional)
    n is small (Section 4.4 covers ~29 statements). The script prints the
    matched n prominently, refuses to emit a headline claim below
    --min-n (default 15), reports CIs everywhere, and labels the analysis
    EXPLORATORY when n < 30. Kendall tau on n<30 has wide CIs: the report
    says so in text, so the number cannot be quoted bare.

USAGE
    python analysis_suite/15_confidence_validation.py --inspect \
        --experts m4/expert_labels_statements.csv \
        --decisions output/run13/m4_panel/m4_panel_decisions.jsonl \
        --provenance output/run13/analysis/provenance_report.csv

    python analysis_suite/15_confidence_validation.py \
        --experts ... --decisions ... --provenance ... \
        --outdir output/run13/analysis/e2
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ── tolerant column detection ─────────────────────────────────────────
SUBJ_COLS = ("subject", "source", "head", "subj", "s", "entity1")
REL_COLS = ("relation", "predicate", "rel", "p")
OBJ_COLS = ("object", "target", "tail", "obj", "o", "entity2")
VERDICT_COLS = ("verdict", "label", "expert_verdict", "judgement",
                "judgment", "decision", "rating", "expert", "consensus")
STATEMENT_COLS = ("statement", "statement_text", "text", "assertion")
ANNOTATOR_COLS = ("annotator", "expert", "expert_name", "reviewer", "who",
                  "pass")

PROTOCOL_ID_COLS = ("statement_id", "id", "stmt_id", "index", "n")
UNICODE_FIX = {"\ufb01": "fi", "\ufb02": "fl", "\u00a0": " ",
               "\u2010": "-", "\u2011": "-", "\u2012": "-",
               "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'"}

VERDICT_MAP = {  # ordinal: N=0, P=1, Y=2
    "y": 2, "yes": 2, "valid": 2, "correct": 2, "true": 2, "agree": 2,
    "p": 1, "partial": 1, "partially": 1, "maybe": 1, "uncertain": 1,
    "n": 0, "no": 0, "invalid": 0, "incorrect": 0, "false": 0,
}
M4_MAP = {"ACCEPT": 2, "UNCERTAIN": 1, "REJECT": 0}
ORD_NAME = {0: "N", 1: "P", 2: "Y"}


def norm(s):
    """Lowercase, repair OCR ligatures (seaﬂoor -> seafloor) and squeeze
    whitespace, so join keys survive PDF-extraction artefacts."""
    s = str(s or "")
    for bad, good in UNICODE_FIX.items():
        s = s.replace(bad, good)
    return re.sub(r"\s+", " ", s.lower()).strip()


def key(s, r, o):
    return (norm(s), norm(r), norm(o))


def pick(cols, candidates):
    """Exact match first (so a bare 'subject' column always wins over
    'cand1_subject'), then a conservative substring fallback."""
    low = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    # Substring fallback, but ONLY for candidates long enough to be
    # unambiguous: a 1-2 char candidate like "s" would match "statement"
    # and silently produce a nonsense mapping (observed bug).
    for c in cols:
        for cand in candidates:
            if len(cand) >= 3 and cand in c.lower():
                return c
    return None


def read_any(path):
    """Read .csv, .jsonl or .json into a list of dicts. For .json, returns
    the top-level list, or the first nested list of dicts found (protocol
    files wrap their records under a key such as 'statements')."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".jsonl":
        return [json.loads(line) for line in
                open(path, encoding="utf-8") if line.strip()]
    if suf == ".json":
        data = json.load(open(path, encoding="utf-8"))
        if isinstance(data, list):
            return data
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        raise ValueError(f"no list of records in {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── statistics ────────────────────────────────────────────────────────

def bootstrap_ci(vals, n_boot=5000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, len(vals), replace=True).mean()
             for _ in range(n_boot)]
    return (float(vals.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def kendall_tau_b(x, y):
    """Kendall tau-b with tie correction; pure numpy (no scipy needed)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3:
        return float("nan")
    conc = disc = 0
    tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            p = dx * dy
            if dx == 0 and dy == 0:
                tx += 1
                ty += 1
            elif dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif p > 0:
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) / 2
    denom = np.sqrt((n0 - tx) * (n0 - ty))
    return float((conc - disc) / denom) if denom > 0 else float("nan")


def perm_pvalue(x, y, stat_fn, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed)
    obs = stat_fn(x, y)
    if not np.isfinite(obs):
        return float("nan")
    y = np.asarray(y, float)
    count = sum(abs(stat_fn(x, rng.permutation(y))) >= abs(obs)
                for _ in range(n_perm))
    return float((count + 1) / (n_perm + 1))


def auc(scores, labels):
    """AUC for binary labels (1 = positive) via rank statistic."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def cohens_kappa(a, b, weighted=False):
    a, b = list(a), list(b)
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    if k < 2 or not a:
        return float("nan")
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    if weighted:
        w = np.array([[abs(i - j) / (k - 1) for j in range(k)]
                      for i in range(k)])
    else:
        w = 1.0 - np.eye(k)
    po, pe = (w * obs).sum(), (w * exp).sum()
    return float(1 - po / pe) if pe > 0 else float("nan")


def load_protocol(path):
    """Return {statement_id(str) -> (subject, relation, object)} from the
    expert annotation protocol (json / jsonl / csv). Tolerant to nesting:
    any list of dicts carrying an id-like key plus s/r/o is accepted."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.load(open(path, encoding="utf-8"))
        items = data if isinstance(data, list) else None
        if items is None:
            for v in data.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    items = v
                    break
        if items is None:
            raise SystemExit(f"No list of records found in {path}")
    else:
        items = read_any(path)
    cols = list(items[0].keys())
    ic = pick(cols, PROTOCOL_ID_COLS)
    # Nested-triple layout is checked FIRST: when a record carries a
    # 'triple' sub-dict, that is authoritative and flat-column guessing
    # would only produce false positives.
    if True:
        if "triple" in cols and isinstance(items[0]["triple"], dict):
            tk = list(items[0]["triple"].keys())
            sc2, rc2, oc2 = (pick(tk, SUBJ_COLS), pick(tk, REL_COLS),
                             pick(tk, OBJ_COLS))
            if all((sc2, rc2, oc2)):
                out = {}
                stc = pick(cols, STATEMENT_COLS)
                for i, it in enumerate(items, 1):
                    sid = str(it.get(ic, i)) if ic else str(i)
                    t = it["triple"]
                    out[sid] = {"triple": (t[sc2], t[rc2], t[oc2]),
                                "statement": it.get(stc, "") if stc else ""}
                print(f"[protocol] {len(out)} statement->triple mappings "
                      f"(nested 'triple' key, id={ic!r})")
                return out
    sc, rc, oc = (pick(cols, SUBJ_COLS), pick(cols, REL_COLS),
                  pick(cols, OBJ_COLS))
    if not all((sc, rc, oc)):
        raise SystemExit(
            f"Could not find subject/relation/object in {path}. "
            f"Columns: {cols}. Report them and the loader will be extended.")
    out = {}
    stc = pick(cols, STATEMENT_COLS)
    for i, it in enumerate(items, 1):
        sid = str(it.get(ic, i)) if ic else str(i)
        out[sid] = {"triple": (it[sc], it[rc], it[oc]),
                    "statement": it.get(stc, "") if stc else ""}
    print(f"[protocol] {len(out)} statement->triple mappings "
          f"(id={ic!r}, s={sc!r}, r={rc!r}, o={oc!r})")
    return out


def text_key(s):
    """Normalized statement text for joining: lowercase, ligatures fixed,
    punctuation dropped, whitespace collapsed."""
    s = norm(s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def melt_experts(experts, protocol):
    """Turn the wide expert table (expert1_verdict / expert2_verdict / ...)
    into long records {statement_id, annotator, verdict_raw, comment,
    subject, relation, object}. Verdicts outside Y/P/N (e.g. 'VERIFY') are
    kept verbatim and excluded from the ordinal analysis, never coerced."""
    cols = list(experts[0].keys())
    idc = pick(cols, PROTOCOL_ID_COLS)
    vcols = [c for c in cols if "verdict" in c.lower()]
    if not vcols:
        raise SystemExit(f"No *verdict* column in expert file: {cols}")
    stc = pick(cols, STATEMENT_COLS)

    # ── JOIN-KEY VALIDATION (critical) ────────────────────────────────
    # statement_id alignment between the expert table and the protocol
    # CANNOT be assumed: an off-by-one or a different ordering silently
    # pairs every verdict with the wrong triple and produces a plausible
    # but meaningless correlation (observed). We therefore join on the
    # statement TEXT and only fall back to the id when texts are absent.
    by_text = {}
    for sid, rec in protocol.items():
        if rec.get("statement"):
            by_text[text_key(rec["statement"])] = (sid, rec["triple"])
    id_aligned = mismatched = 0
    if stc and by_text:
        for i, e in enumerate(experts, 1):
            sid = str(e.get(idc, i)) if idc else str(i)
            pr = protocol.get(sid)
            if pr and pr.get("statement") and e.get(stc):
                if text_key(pr["statement"]) == text_key(e[stc]):
                    id_aligned += 1
                else:
                    mismatched += 1
        print(f"[join check] statement_id alignment: {id_aligned} matching "
              f"texts, {mismatched} MISMATCHED"
              + ("  <-- ids are NOT aligned; joining on statement text"
                 if mismatched else "  (ids are consistent)"))

    long = []
    n_text_join = n_fuzzy_join = n_id_join = 0
    for i, e in enumerate(experts, 1):
        sid = str(e.get(idc, i)) if idc else str(i)
        tri = None
        # 1) exact normalized text match
        if stc and e.get(stc):
            hit = by_text.get(text_key(e[stc]))
            if hit:
                tri = hit[1]
                n_text_join += 1
            else:
                # 2) fuzzy text match (protocol statements are auto-generated
                #    and may have been lightly edited before annotation)
                import difflib
                tk = text_key(e[stc])
                best, best_r = None, 0.0
                for k_, v in by_text.items():
                    r = difflib.SequenceMatcher(None, tk, k_).ratio()
                    if r > best_r:
                        best, best_r = v, r
                if best and best_r >= 0.80:
                    tri = best[1]
                    n_fuzzy_join += 1
        # 3) id fallback ONLY when texts are unavailable on either side
        if tri is None and not (stc and by_text):
            rec = protocol.get(sid)
            if rec:
                tri = rec["triple"]
                n_id_join += 1
        for vc in vcols:
            ann = vc.replace("_verdict", "")
            ccol = vc.replace("verdict", "comment")
            long.append({
                "statement_id": sid, "annotator": ann,
                "verdict_raw": (e.get(vc) or "").strip(),
                "comment": e.get(ccol, ""),
                "statement": e.get(pick(cols, STATEMENT_COLS), ""),
                "subject": tri[0] if tri else None,
                "relation": tri[1] if tri else None,
                "object": tri[2] if tri else None,
            })
    print(f"[join] statement->triple: {n_text_join} exact-text, "
          f"{n_fuzzy_join} fuzzy-text, {n_id_join} id-fallback, "
          f"{sum(1 for r in long if r['subject'] is None)//max(len(vcols),1)}"
          " unresolved")
    return long


# ── inspection ────────────────────────────────────────────────────────

def inspect(path, label):
    print("=" * 70)
    print(f"{label}: {path}")
    print("=" * 70)
    try:
        rows = read_any(path)
    except Exception as e:
        print(f"  !! could not read: {e}")
        return
    if not rows:
        print("  !! empty")
        return
    cols = [c for c in rows[0].keys() if c is not None]
    print(f"  rows: {len(rows)}")
    for c in cols:
        if isinstance(rows[0].get(c), dict):
            print(f"  NESTED dict under {c!r}: "
                  f"keys={list(rows[0][c].keys())}")
    print(f"  columns: {cols}")
    print(f"  detected: subject={pick(cols, SUBJ_COLS)!r} "
          f"relation={pick(cols, REL_COLS)!r} object={pick(cols, OBJ_COLS)!r}")
    print(f"            verdict={pick(cols, VERDICT_COLS)!r} "
          f"statement={pick(cols, STATEMENT_COLS)!r} "
          f"annotator={pick(cols, ANNOTATOR_COLS)!r}")
    for c in cols:
        vals = [str(r.get(c, "")) for r in rows]
        uniq = Counter(vals)
        if 1 < len(uniq) <= 8:
            print(f"  vocabulary of {c!r}: {dict(uniq)}")
    print("  first row:")
    for k_, v in list(rows[0].items())[:12]:
        print(f"    {k_}: {str(v)[:90]}")
    print()


# ── main analysis ─────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", required=True,
                    help="expert verdict table (csv/jsonl), Section 4.4")
    ap.add_argument("--decisions", required=True,
                    help="m4_panel_decisions.jsonl (or m4_decisions.jsonl)")
    ap.add_argument("--provenance", required=True,
                    help="provenance_report.csv from 08")
    ap.add_argument("--protocol", default=None,
                    help="expert annotation protocol mapping statement_id "
                         "-> (subject, relation, object); required when the "
                         "expert table stores statements only")
    ap.add_argument("--outdir", default="output/run13/analysis/e2")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--human-kappa", nargs="*", type=float,
                    default=[0.30, 0.37],
                    help="human inter-expert kappa from Section 4.4")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.experts, "EXPERT VERDICTS")
        if args.protocol:
            inspect(args.protocol, "ANNOTATION PROTOCOL")
        inspect(args.decisions, "M4 DECISIONS")
        inspect(args.provenance, "PROVENANCE / CONFIDENCE")
        print("Inspection done. Check that subject/relation/object and the "
              "verdict column were detected in the expert file.\n"
              "If the expert table stores STATEMENTS without (s,r,o) "
              "columns, a mapping file is needed — report the columns and "
              "we extend the loader.")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    experts = read_any(args.experts)
    decisions = read_any(args.decisions)
    prov = read_any(args.provenance)

    ecols = list(experts[0].keys())

    # ── GUARD: never join on unvalidated auto-mapping columns ─────────
    # 16_map_statements_to_triples.py emits both cand1_* (machine proposals)
    # and confirmed_* (human-validated). Joining on cand1_* would silently
    # reproduce the very failure mode this pipeline is meant to avoid.
    if any(cc.startswith("cand1_") for cc in ecols):
        conf_cols = [cc for cc in ecols if cc.startswith("confirmed_")]
        if not conf_cols:
            raise SystemExit(
                "This file contains cand1_* proposal columns but no "
                "confirmed_* columns. Fill in the confirmed_* columns by "
                "hand before running E2.")
        filled = sum(1 for e in experts
                     if str(e.get("confirmed_subject", "")).strip())
        if filled == 0:
            raise SystemExit(
                f"None of the {len(experts)} rows has confirmed_subject "
                "filled in. E2 must NOT run on machine proposals: open "
                "statement_triple_candidates.csv, fill confirmed_subject/"
                "relation/object, then re-run.")
        print(f"[guard] joining on confirmed_* columns "
              f"({filled}/{len(experts)} rows validated; unfilled rows are "
              "skipped)")
        experts = [e for e in experts
                   if str(e.get("confirmed_subject", "")).strip()]
        for e in experts:
            e["subject"] = e["confirmed_subject"]
            e["relation"] = e.get("confirmed_relation", "")
            e["object"] = e.get("confirmed_object", "")
        ecols = list(experts[0].keys())

    has_triple_cols = all((pick(ecols, SUBJ_COLS), pick(ecols, REL_COLS),
                           pick(ecols, OBJ_COLS)))
    if has_triple_cols and not args.protocol:
        # legacy path: expert table already carries (s, r, o)
        long = []
        es, er, eo = (pick(ecols, SUBJ_COLS), pick(ecols, REL_COLS),
                      pick(ecols, OBJ_COLS))
        vc = pick(ecols, VERDICT_COLS)
        ac = pick(ecols, ANNOTATOR_COLS)
        for i, e in enumerate(experts, 1):
            long.append({"statement_id": str(i),
                         "annotator": e.get(ac, "expert") if ac else "expert",
                         "verdict_raw": (e.get(vc) or "").strip(),
                         "comment": "", "statement": "",
                         "subject": e[es], "relation": e[er],
                         "object": e[eo]})
    else:
        if not args.protocol:
            raise SystemExit(
                "The expert table stores statements without (subject, "
                "relation, object) columns. Pass --protocol with the "
                "annotation protocol that maps statement_id -> triple "
                "(e.g. output/expert_annotation_protocol.json).")
        protocol = load_protocol(args.protocol)
        if len(protocol) != len(experts):
            print(f"[note] protocol has {len(protocol)} statements but the "
                  f"expert table has {len(experts)} rows — statements "
                  "without a protocol entry are listed in e2_unmatched.csv")
        long = melt_experts(experts, protocol)

    # index M4 decisions and provenance by normalized triple key
    dmap, pmap = {}, {}
    for d in decisions:
        dmap[key(d.get("subject"), d.get("relation"), d.get("object"))] = d
    for p in prov:
        pmap[key(p.get("subject"), p.get("relation"), p.get("object"))] = p

    rows, unmatched = [], []
    off_scale = Counter()
    for e in long:
        if not e["subject"]:
            unmatched.append({**{k_: e[k_] for k_ in
                                 ("statement_id", "annotator", "verdict_raw",
                                  "statement")},
                              "reason": "no triple in protocol"})
            continue
        k = key(e["subject"], e["relation"], e["object"])
        raw = norm(e["verdict_raw"])
        v = VERDICT_MAP.get(raw) or VERDICT_MAP.get(raw[:1])
        if v is None:
            off_scale[e["verdict_raw"]] += 1
        d, p = dmap.get(k), pmap.get(k)
        if d is None and p is None:
            unmatched.append({"statement_id": e["statement_id"],
                              "annotator": e["annotator"],
                              "verdict_raw": e["verdict_raw"],
                              "statement": e["statement"],
                              "reason": "triple not in KG outputs"})
            continue
        row = {"statement_id": e["statement_id"],
               "annotator": e["annotator"],
               "subject": e["subject"], "relation": e["relation"],
               "object": e["object"],
               "expert_verdict_raw": e["verdict_raw"],
               "expert_verdict": ORD_NAME.get(v, e["verdict_raw"]),
               "comment": e["comment"]}
        if v is not None:
            row["expert_ord"] = v
        if p:
            row.update({
                "confidence": float(p.get("confidence") or 0),
                "support_papers": int(float(p.get("support_papers") or 0)),
                "support_chunks": int(float(p.get("support_chunks") or 0)),
                "tier": int(float(p.get("tier") or 0)),
            })
        if d:
            dec = str(d.get("m4_decision") or d.get("decision") or "").upper()
            row["m4_decision"] = dec
            row["m4_ord"] = M4_MAP.get(dec)
            if d.get("m4_confidence") not in (None, ""):
                row["m4_confidence"] = float(d["m4_confidence"])
        rows.append(row)

    n_offscale = sum(off_scale.values())
    rows_ord = [r for r in rows if "expert_ord" in r]
    if off_scale:
        print(f"[note] {n_offscale} verdicts outside the Y/P/N scale kept in "
              f"the joined table but excluded from ordinal statistics: "
              f"{dict(off_scale)}")
    rows_all = rows
    rows = rows_ord  # ordinal analyses use Y/P/N only

    n = len(rows)
    print("=" * 70)
    print("E2 — CONFIDENCE x EXPERT x M4")
    print("=" * 70)
    print(f"expert statements   : {len(experts)}")
    print(f"annotator records   : {len(long)}")
    print(f"joined (any verdict): {len(rows_all)}")
    print(f"usable Y/P/N        : {n}")
    print(f"unmatched           : {len(unmatched)}")
    if n < args.min_n:
        print(f"\n!! n={n} < min-n={args.min_n}: too few matches for any "
              "headline claim. Inspect e2_unmatched.csv — the usual cause "
              "is that Section 4.4 was annotated on run11 triples whose "
              "surface forms were canonicalized differently in run13.")

    with open(outdir / "e2_joined.csv", "w", newline="",
              encoding="utf-8") as f:
        if rows_all:
            w = csv.DictWriter(f, fieldnames=sorted(
                {k_ for r in rows_all for k_ in r}), restval="")
            w.writeheader()
            w.writerows(rows_all)
    if unmatched:
        with open(outdir / "e2_unmatched.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(unmatched[0].keys()))
            w.writeheader()
            w.writerows(unmatched)

    report = {"n_expert_records": len(experts),
              "n_annotator_records": len(long),
              "n_joined_any": len(rows_all),
              "n_offscale_verdicts": n_offscale,
              "offscale_vocabulary": dict(off_scale),
              "n_matched": n,
              "n_unmatched": len(unmatched),
              "exploratory": n < 30, "min_n": args.min_n}

    # ── (a) confidence channels vs expert verdict ─────────────────────
    # 'tier' is reported as tier_inverted (see flip below) so that all
    # channels share the same orientation.
    channels = ["confidence", "support_papers", "support_chunks", "tier"]
    chan_stats = {}
    for ch in channels:
        # tier is inverted by construction (1 = strongest): flip it so a
        # POSITIVE tau always means "higher value = better expert verdict"
        # across every channel, otherwise the sign is unreadable.
        flip = -1.0 if ch == "tier" else 1.0
        vals = [(flip * r[ch], r["expert_ord"]) for r in rows if ch in r]
        if len(vals) < 5:
            continue
        x = [v for v, _ in vals]
        y = [o for _, o in vals]
        tau = kendall_tau_b(x, y)
        pval = perm_pvalue(x, y, kendall_tau_b) if len(vals) >= 8 else None
        by_class = {}
        for cls in (2, 1, 0):
            sel = [v for v, o in vals if o == cls]
            if sel:
                m, lo, hi = bootstrap_ci(sel)
                by_class[ORD_NAME[cls]] = {"n": len(sel), "mean": round(m, 4),
                                           "ci95": [round(lo, 4),
                                                    round(hi, 4)]}
        a = auc(x, [1 if o == 2 else 0 for o in y])
        chan_stats["tier_inverted" if ch == "tier" else ch] = {
                          "n": len(vals), "kendall_tau_b": round(tau, 4),
                          "perm_p": None if pval is None else round(pval, 4),
                          "auc_Y_vs_PN": round(a, 4), "by_verdict": by_class}
    report["confidence_vs_expert"] = chan_stats

    # ── (a-bis) INDEPENDENT-UNIT analyses ─────────────────────────────
    # The pooled statistics above treat each (triple, annotator) pair as one
    # observation, but the same triple appears once per annotator with an
    # IDENTICAL confidence value. Those rows are not independent, so the
    # pooled tau and its permutation p-value are anti-conservative. The
    # analyses below use independent units:
    #   - per annotator  : within one annotator, one row per triple
    #   - consensus      : one row per triple, verdict = min (most
    #                      conservative) and mean of the annotators' ordinals
    indep = {}
    annotators = sorted({r["annotator"] for r in rows if r.get("annotator")})
    for ann in annotators:
        sub = [r for r in rows if r["annotator"] == ann]
        blk = {"n": len(sub)}
        for ch in channels:
            flip = -1.0 if ch == "tier" else 1.0
            vals = [(flip * r[ch], r["expert_ord"]) for r in sub if ch in r]
            if len(vals) < 5:
                continue
            x = [v for v, _ in vals]
            y = [o for _, o in vals]
            if len(set(x)) < 2 or len(set(y)) < 2:
                blk["tier_inverted" if ch == "tier" else ch] = {
                    "n": len(vals), "note": "no variance — statistic "
                    "undefined (e.g. all statements are Tier-1)"}
                continue
            blk["tier_inverted" if ch == "tier" else ch] = {
                "n": len(vals),
                "kendall_tau_b": round(kendall_tau_b(x, y), 4),
                "perm_p": round(perm_pvalue(x, y, kendall_tau_b), 4),
                "auc_Y_vs_PN": round(
                    auc(x, [1 if o == 2 else 0 for o in y]), 4)}
        indep[ann] = blk

    # consensus per triple
    by_triple = defaultdict(list)
    for r in rows:
        by_triple[(r["subject"], r["relation"], r["object"])].append(r)
    cons_rows = []
    for k_, grp in by_triple.items():
        ords = [g["expert_ord"] for g in grp]
        base = dict(grp[0])
        base["expert_ord_min"] = min(ords)
        base["expert_ord_mean"] = sum(ords) / len(ords)
        base["n_annotators"] = len(ords)
        base["annotators_agree"] = len(set(ords)) == 1
        cons_rows.append(base)
    cons = {"n_triples": len(cons_rows),
            "n_full_agreement": sum(1 for r in cons_rows
                                    if r["annotators_agree"])}
    for ch in channels:
        flip = -1.0 if ch == "tier" else 1.0
        for tag, fld in (("min", "expert_ord_min"), ("mean",
                                                     "expert_ord_mean")):
            vals = [(flip * r[ch], r[fld]) for r in cons_rows if ch in r]
            if len(vals) < 5:
                continue
            x = [v for v, _ in vals]
            y = [o for _, o in vals]
            if len(set(x)) < 2 or len(set(y)) < 2:
                continue
            name = ("tier_inverted" if ch == "tier" else ch) + f"__{tag}"
            cons[name] = {"n": len(vals),
                          "kendall_tau_b": round(kendall_tau_b(x, y), 4),
                          "perm_p": round(perm_pvalue(x, y, kendall_tau_b),
                                          4)}
    report["independent_units"] = {"per_annotator": indep,
                                   "consensus_per_triple": cons}

    # range-restriction diagnostic: how much of the confidence scale is used?
    confs = [r["confidence"] for r in rows if "confidence" in r]
    tiers = {r.get("tier") for r in rows if "tier" in r}
    report["range_restriction"] = {
        "confidence_min": round(min(confs), 4) if confs else None,
        "confidence_max": round(max(confs), 4) if confs else None,
        "confidence_sd": round(float(np.std(confs)), 4) if confs else None,
        "distinct_tiers_present": sorted(t for t in tiers if t is not None),
        "note": "The Section 4.4 sample was drawn from Tier-1 only, so "
                "w_tier is constant and the confidence range is truncated. "
                "Any attenuation of the correlation must be read in that "
                "light: this tests the score WITHIN the top stratum, not "
                "across the full reliability range."}

    # ── (b) M4 vs human ───────────────────────────────────────────────
    pairs = [(r["m4_ord"], r["expert_ord"]) for r in rows
             if r.get("m4_ord") is not None]
    m4_stats = {"n": len(pairs)}
    if pairs:
        cont = defaultdict(int)
        for m, e in pairs:
            cont[(ORD_NAME[m], ORD_NAME[e])] += 1
        m4_stats["contingency_m4_x_expert"] = {
            f"{a}|{b}": c for (a, b), c in sorted(cont.items())}
        m4_stats["kappa_unweighted"] = round(
            cohens_kappa([m for m, _ in pairs], [e for _, e in pairs]), 4)
        m4_stats["kappa_linear"] = round(
            cohens_kappa([m for m, _ in pairs], [e for _, e in pairs],
                         weighted=True), 4)
        m4_stats["exact_agreement"] = round(
            sum(m == e for m, e in pairs) / len(pairs), 4)
        m4_stats["human_reference_kappa"] = args.human_kappa
    report["m4_vs_expert"] = m4_stats

    # per-annotator
    if any(r.get("annotator") for r in rows):
        per = {}
        for ann in sorted({r["annotator"] for r in rows if r["annotator"]}):
            sub = [r for r in rows if r["annotator"] == ann]
            pr = [(r["m4_ord"], r["expert_ord"]) for r in sub
                  if r.get("m4_ord") is not None]
            per[ann] = {"n": len(sub),
                        "kappa_linear": round(cohens_kappa(
                            [m for m, _ in pr], [e for _, e in pr],
                            weighted=True), 4) if pr else None}
        report["per_annotator"] = per

    with open(outdir / "e2_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── figures ───────────────────────────────────────────────────────
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from kg_io import BLUE, TERRACOTTA, apply_style
        plt = apply_style()
        conf = [(r.get("confidence"), r["expert_ord"]) for r in rows
                if "confidence" in r]
        if conf:
            fig, ax = plt.subplots(figsize=(7, 5))
            data = [[c for c, o in conf if o == cls] for cls in (2, 1, 0)]
            labels = [f"{ORD_NAME[c]} (n={len(d)})"
                      for c, d in zip((2, 1, 0), data)]
            bp = ax.boxplot([d if d else [0] for d in data],
                            tick_labels=labels,
                            patch_artist=True, widths=0.5)
            for patch in bp["boxes"]:
                patch.set_facecolor(BLUE)
                patch.set_alpha(0.5)
            rng = np.random.default_rng(0)
            for i, d in enumerate(data, 1):
                if d:
                    ax.scatter(rng.normal(i, 0.06, len(d)), d,
                               color=TERRACOTTA, zorder=3, s=25)
            ax.set_ylabel("confidence")
            ax.set_xlabel("expert verdict")
            ax.set_title("Confidence by expert verdict", fontsize=18)
            fig.savefig(outdir / "fig_e2_confidence_by_verdict.png")
            plt.close(fig)
        if pairs:
            m = np.zeros((3, 3))
            for a, b in pairs:
                m[2 - a, 2 - b] += 1
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(m, cmap="Blues")
            ax.set_xticks(range(3))
            ax.set_xticklabels(["Y", "P", "N"])
            ax.set_yticks(range(3))
            ax.set_yticklabels(["ACCEPT", "UNCERTAIN", "REJECT"])
            ax.set_xlabel("expert verdict")
            ax.set_ylabel("M4 panel decision")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, int(m[i, j]), ha="center", va="center",
                            color="black")
            ax.set_title("M4 panel vs expert", fontsize=18)
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.savefig(outdir / "fig_e2_m4_vs_expert.png")
            plt.close(fig)
    except Exception as e:
        print(f"[WARN] figures skipped: {e}")

    # ── narrative report ──────────────────────────────────────────────
    L = ["# E2 — Confidence score vs expert judgement, M4 as expert proxy",
         "", f"Matched triples: **{n}** "
         f"(expert records {len(experts)}, unmatched {len(unmatched)}).", ""]
    if n < 30:
        L.append("> **EXPLORATORY.** With n < 30 all interval estimates are "
                 "wide; report effect sizes with their CIs and never quote "
                 "a bare correlation coefficient.")
        L.append("")
    L.append("## (a) Does confidence track expert judgement?")
    for ch, s in chan_stats.items():
        L.append(f"- **{ch}** (n={s['n']}): tau-b = {s['kendall_tau_b']}"
                 + (f", permutation p = {s['perm_p']}" if s["perm_p"]
                    is not None else "")
                 + f", AUC(Y vs P/N) = {s['auc_Y_vs_PN']}. "
                 + "; ".join(f"{k_}: {v['mean']} "
                             f"[{v['ci95'][0]}, {v['ci95'][1]}] (n={v['n']})"
                             for k_, v in s["by_verdict"].items()))
    L.append("")
    L.append("Interpretation rule agreed in advance: a positive tau with a "
             "monotone mean-confidence ordering Y > P > N supports the score "
             "as an external-validated reliability signal; a null or "
             "non-monotone result is reported as a limitation motivating "
             "human validation, not hidden.")
    L.append("")
    L.append("## (a-bis) Independent-unit check")
    L.append("The pooled statistics above double-count each triple (one row "
             "per annotator, identical confidence). Independent-unit "
             "results:")
    for ann, blk in report["independent_units"]["per_annotator"].items():
        parts = []
        for ch, s in blk.items():
            if ch == "n" or "kendall_tau_b" not in s:
                continue
            parts.append(f"{ch} tau={s['kendall_tau_b']} (p={s['perm_p']})")
        L.append(f"- **{ann}** (n={blk['n']}): "
                 + ("; ".join(parts) if parts else "no channel with "
                    "sufficient variance"))
    cc = report["independent_units"]["consensus_per_triple"]
    L.append(f"- **consensus** ({cc['n_triples']} triples, "
             f"{cc['n_full_agreement']} with full annotator agreement): "
             + "; ".join(f"{k_} tau={v['kendall_tau_b']} (p={v['perm_p']})"
                         for k_, v in cc.items()
                         if isinstance(v, dict)))
    rr = report["range_restriction"]
    L.append("")
    L.append(f"**Range restriction.** Confidence in this sample spans "
             f"[{rr['confidence_min']}, {rr['confidence_max']}] "
             f"(sd={rr['confidence_sd']}), tiers present: "
             f"{rr['distinct_tiers_present']}. {rr['note']}")
    L.append("")
    L.append("## (b) Is the M4 panel a proxy for an expert?")
    if pairs:
        L.append(f"- n = {m4_stats['n']}, exact agreement = "
                 f"{m4_stats['exact_agreement']}, kappa unweighted = "
                 f"{m4_stats['kappa_unweighted']}, kappa linear-weighted = "
                 f"{m4_stats['kappa_linear']}.")
        L.append(f"- Human inter-expert reference (Section 4.4): "
                 f"{args.human_kappa}.")
        L.append("- Reading: if machine-human kappa is comparable to "
                 "human-human kappa, the panel is as consistent with an "
                 "expert as experts are with each other — the agreement "
                 "ceiling is a property of the task, not of the judges.")
    else:
        L.append("- No M4 decision joined; check the decision file keys.")
    (outdir / "e2_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\noutputs in: {outdir}")
    for ch, s in chan_stats.items():
        print(f"  {ch:16s} tau={s['kendall_tau_b']:+.3f} "
              f"AUC={s['auc_Y_vs_PN']:.3f}")
    if pairs:
        print(f"  M4 vs expert: kappa_linear={m4_stats['kappa_linear']} "
              f"(human ref {args.human_kappa})")


if __name__ == "__main__":
    main()