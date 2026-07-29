#!/usr/bin/env python3
"""
18_paper_figures.py — Manuscript figures for the run13 results.
================================================================
WHY
    The four headline results established during the run13 campaign have no
    figure yet: the corpus-contamination ablation, the frozen dev/test
    evaluation protocol, the anatomy of the confidence score, and the
    provenance/consensus profile. Everything else (KG portrait, descriptors,
    relations, growth, M4 verdicts) already has a generator.

WHAT — every number is read from disk, nothing is hard-coded:
    Fig A  fig_paper_contamination.{pdf,png}
           run11 vs run13: index composition, effective top-5 context
           diversity, raw triples per pass, per-pass recall.
    Fig B  fig_paper_frozen_protocol.{pdf,png}
           recall on the frozen dev/test splits + combined benchmarks, with
           Wilson 95% intervals (n=17 per split — the interval is the point).
    Fig C  fig_paper_confidence_anatomy.{pdf,png}
           the three orthogonal channels (w_tier, w_m4, w_consensus) and the
           resulting composite distribution, by tier.
    Fig D  fig_paper_provenance.{pdf,png}
           evidence anchoring outcome and inter-article consensus
           distribution (how many papers support each triple).
    figure_manifest.md — every manuscript figure, its generator, its status.

INPUTS (paths are defaults; override if your layout differs)
    run11: output/step1/chunks.jsonl, output/run11_a/{raw_triples,
           canonical_triples_v5}.jsonl, output/run11_kg/metrics_*.json
    run13: output/run13/step1/chunks.jsonl, output/run13/pass_a/*.jsonl,
           output/run13/kg/metrics_{dev,test,full34}.json,
           output/run13/analysis/provenance_report.csv

USAGE
    python analysis_suite/18_paper_figures.py --outdir figures/paper
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from kg_io import BLUE, TERRACOTTA, apply_style

GREEN = "#6B8F71"
GREY = "#8A8A8A"


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def mannwhitney_p(a, b):
    """Two-sided Mann-Whitney U p-value (normal approximation with tie
    correction). Pure numpy; returns (U, p) or (nan, nan) if degenerate."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3:
        return float("nan"), float("nan")
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):          # average ranks over ties
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    _, counts = np.unique(allv, return_counts=True)
    tie = sum(t ** 3 - t for t in counts)
    n = n1 + n2
    sd = math.sqrt(n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1))))
    if sd == 0:
        return float(u1), float("nan")
    z = (u1 - mu) / sd
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return float(u1), float(p)


def count_lines(p):
    p = Path(p)
    if not p.exists():
        return None
    return sum(1 for line in open(p, encoding="utf-8") if line.strip())


def chunk_stats(path):
    """(n_chunks, n_papers_normalized, n_checkpoint_records)."""
    path = Path(path)
    if not path.exists():
        return None
    n, papers, ck = 0, set(), 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        doc = str(r.get("doc_id", ""))
        if doc.endswith("-checkpoint"):
            ck += 1
        papers.add(re.sub(r"-checkpoint$", "", doc))
    return n, len(papers), ck


def context_diversity(path):
    """Mean unique chunks among the retrieved top-k, and % duplicate slots,
    computed from `_provenance.selected_chunks` stored at extraction time."""
    path = Path(path)
    if not path.exists():
        return None
    divs, tot, ck = [], 0, 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        t = json.loads(line)
        sc = (t.get("_provenance") or {}).get("selected_chunks") or []
        if not sc:
            continue
        norm = {re.sub(r"-checkpoint$", "", c.split("::")[0]) + "::"
                + c.split("::")[-1] for c in sc}
        divs.append(len(norm))
        tot += len(sc)
        ck += sum(1 for c in sc if "checkpoint" in c.split("::")[0])
    if not divs:
        return None
    return float(np.mean(divs)), 100.0 * ck / max(tot, 1), len(divs)


def load_metrics(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.load(open(path, encoding="utf-8"))


def recall_pair(metrics):
    """(hits, total) for Tier1+2 from a 07_final_metrics.py output."""
    if not metrics:
        return None
    r = metrics.get("recall_vs_lb2019", {}).get("tier12", {})
    hits = r.get("hits")
    tot = r.get("total_reference") or r.get("n_reference") or r.get("total")
    if hits is None or not tot:
        return None
    return int(hits), int(tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures/paper")
    ap.add_argument("--run11-chunks", default="output/step1/chunks.jsonl")
    ap.add_argument("--run13-chunks",
                    default="output/run13/step1/chunks.jsonl")
    ap.add_argument("--run11-raw",
                    default="output/run11_a/raw_triples.jsonl")
    ap.add_argument("--run13-raw",
                    default="output/run13/pass_a/raw_triples.jsonl")
    ap.add_argument("--run11-canon",
                    default="output/run11_a/canonical_triples_v5.jsonl")
    ap.add_argument("--run13-canon",
                    default="output/run13/pass_a/canonical_triples_v5.jsonl")
    ap.add_argument("--run11-stats",
                    default="output/run11_a/cleaning_stats_v5.json")
    ap.add_argument("--run13-stats",
                    default="output/run13/pass_a/cleaning_stats_v5.json")
    ap.add_argument("--metrics-dev",
                    default="output/run13/kg/metrics_dev.json")
    ap.add_argument("--metrics-test",
                    default="output/run13/kg/metrics_test.json")
    ap.add_argument("--metrics-full",
                    default="output/run13/kg/metrics_full34.json")
    ap.add_argument("--provenance",
                    default="output/run13/analysis/provenance_report.csv")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt = apply_style()
    made, skipped = [], []

    def save(fig, name):
        fig.savefig(outdir / f"{name}.pdf")
        fig.savefig(outdir / f"{name}.png", dpi=300)
        plt.close(fig)
        made.append(name)
        print(f"  [ok] {name}")

    # ── Fig A — contamination ablation ────────────────────────────────
    c11, c13 = chunk_stats(args.run11_chunks), chunk_stats(args.run13_chunks)
    d11 = context_diversity(args.run11_canon)
    d13 = context_diversity(args.run13_canon)
    r11, r13 = count_lines(args.run11_raw), count_lines(args.run13_raw)
    s11, s13 = load_metrics(args.run11_stats), load_metrics(args.run13_stats)

    if c11 and c13 and r11 and r13:
        fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))

        ax = axes[0]
        dup11 = c11[2]
        ax.bar(["run11", "run13"], [c11[0] - dup11, c13[0]], color=BLUE,
               label="unique")
        ax.bar(["run11", "run13"], [dup11, 0], bottom=[c11[0] - dup11, c13[0]],
               color=TERRACOTTA, label="duplicate")
        ax.set_ylabel("chunks in index")
        ax.set_title("Index composition", fontsize=15)
        ax.legend(frameon=False, fontsize=9)
        if dup11:
            ax.text(0, c11[0] * 1.02, f"{100*dup11/c11[0]:.1f}% dup",
                    ha="center", fontsize=9)

        ax = axes[1]
        if d11 and d13:
            ax.bar(["run11", "run13"], [d11[0], d13[0]],
                   color=[TERRACOTTA, BLUE])
            ax.axhline(5, ls="--", c=GREY, lw=1)
            ax.text(1.35, 5.02, "top-k = 5", fontsize=8, color=GREY)
            ax.set_ylim(0, 5.6)
            ax.set_ylabel("unique passages in top-5")
            for i, v in enumerate([d11[0], d13[0]]):
                ax.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=10)
        ax.set_title("Context diversity", fontsize=15)

        ax = axes[2]
        ax.bar(["run11", "run13"], [r11, r13], color=[TERRACOTTA, BLUE])
        for i, v in enumerate([r11, r13]):
            ax.text(i, v * 1.01, str(v), ha="center", fontsize=10)
        ax.set_ylabel("raw triples (pass A)")
        ax.set_title(f"Extraction yield (+{100*(r13-r11)/r11:.0f}%)",
                     fontsize=15)

        ax = axes[3]
        if s11 and s13 and s11.get("lb_recall") and s13.get("lb_recall"):
            def parse(s):
                a, b = s.split("/")
                return int(a), int(b)
            h11, n11 = parse(s11["lb_recall"])
            h13, n13 = parse(s13["lb_recall"])
            for i, (h, n_, col) in enumerate(((h11, n11, TERRACOTTA),
                                              (h13, n13, BLUE))):
                p, lo, hi = wilson(h, n_)
                ax.bar(i, 100 * p, color=col)
                ax.errorbar(i, 100 * p, yerr=[[100 * (p - lo)],
                                              [100 * (hi - p)]],
                            fmt="none", ecolor="black", capsize=4, lw=1)
                ax.text(i, 100 * hi + 2, f"{h}/{n_}", ha="center",
                        fontsize=10)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["run11", "run13"])
            ax.set_ylim(0, 105)
            ax.set_ylabel("per-pass recall (%)")
        ax.set_title("Recall, pass A", fontsize=15)

        fig.suptitle("Corpus contamination ablation (pipeline held constant)",
                     fontsize=19, fontweight="bold", y=1.02)
        fig.tight_layout()
        save(fig, "fig_paper_contamination")
    else:
        skipped.append("fig_paper_contamination (missing run11/run13 inputs)")

    # ── Fig B — frozen evaluation protocol ────────────────────────────
    mdev, mtest = load_metrics(args.metrics_dev), load_metrics(
        args.metrics_test)
    mfull = load_metrics(args.metrics_full)
    bars = []
    for label, m, col in (("dev\n(frozen)", mdev, GREY),
                          ("test\n(frozen)", mtest, BLUE),
                          ("34-edge\ncombined", mfull, TERRACOTTA)):
        rp = recall_pair(m)
        if rp:
            bars.append((label, rp[0], rp[1], col))
    if mfull:
        o26 = mfull.get("recall_vs_lb2019_orig26", {}).get("tier12", {})
        if o26.get("hits") is not None:
            tot = o26.get("total_reference") or 26
            bars.append(("26-edge\noriginal", int(o26["hits"]), int(tot),
                         GREEN))
    if bars:
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (lab, h, n_, col) in enumerate(bars):
            p, lo, hi = wilson(h, n_)
            ax.bar(i, 100 * p, color=col, width=0.6)
            ax.errorbar(i, 100 * p, yerr=[[100 * (p - lo)],
                                          [100 * (hi - p)]],
                        fmt="none", ecolor="black", capsize=5, lw=1.2)
            ax.text(i, 100 * hi + 2.5, f"{h}/{n_}\n{100*p:.1f}%",
                    ha="center", fontsize=10)
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels([b[0] for b in bars])
        ax.set_ylim(0, 118)
        ax.set_ylabel("recall vs LB2019 (%)")
        ax.set_title("Recall under the pre-registered split", fontsize=18)
        ax.text(0.5, -0.22, "Split drawn, hashed and committed before run13. "
                "Error bars: Wilson 95%; n=17 per split, so one edge ≈ 5.9 pp.",
                transform=ax.transAxes, ha="center", fontsize=9, color="#444")
        fig.tight_layout()
        save(fig, "fig_paper_frozen_protocol")
    else:
        skipped.append("fig_paper_frozen_protocol (metrics json missing)")

    # ── Figs C & D — confidence anatomy and provenance ────────────────
    prov = Path(args.provenance)
    if prov.exists():
        rows = list(csv.DictReader(open(prov, newline="", encoding="utf-8")))
        conf = [float(r["confidence"]) for r in rows if r.get("confidence")]
        tiers = [int(float(r["tier"])) for r in rows if r.get("tier")]
        papers = [int(float(r["support_papers"] or 0)) for r in rows]
        match = Counter(r.get("evidence_match", "none").split("(")[0]
                        for r in rows)

        # Three panels. The composite score CONTAINS w_tier, so a split by
        # tier on the composite is partly tautological; the middle panel
        # therefore shows the tier-INDEPENDENT part of the score
        # (w_m4 x w_consensus). If that still separates the tiers, the
        # separation is a genuine result rather than an artefact of the
        # formula.
        w_m4 = [float(r.get("w_m4") or 0) for r in rows]
        w_cons = [float(r.get("w_consensus") or 0) for r in rows]
        if not any(w_m4):  # components not in the CSV -> read them from JSON
            kgp = prov.parent / "kg_with_provenance.json"
            if kgp.exists():
                import sys as _s
                _s.path.insert(0, str(Path(__file__).parent))
                from kg_io import load_kg as _lk
                _kg = _lk(kgp)
                w_m4, w_cons, conf, tiers = [], [], [], []
                for t in _kg["triples"]:
                    cc = t.get("conf_components") or {}
                    w_m4.append(float(cc.get("w_m4") or 0))
                    w_cons.append(float(cc.get("w_consensus") or 0))
                    conf.append(float(t.get("confidence") or 0))
                    tiers.append(int(t.get("_tier") or 0))

        # Two panels only: the former third panel (paper support, all
        # triples pooled) duplicated the by-tier panel; the pooled headline
        # figure now lives in the panel title and the caption.
        indep = [a * b for a, b in zip(w_m4, w_cons)]
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
        bins = np.linspace(0, 1, 21)

        ax = axes[0]
        c_t1 = [c for c, t in zip(conf, tiers) if t == 1]
        c_t2 = [c for c, t in zip(conf, tiers) if t == 2]
        ax.hist([c_t1, c_t2], bins=bins, stacked=True,
                color=[BLUE, TERRACOTTA],
                label=[f"Tier-1 (n={len(c_t1)})", f"Tier-2 (n={len(c_t2)})"])
        ax.set_xlabel("composite confidence")
        ax.set_ylabel("triples")
        ax.legend(frameon=False, fontsize=9)
        ax.set_title("A. Composite score", fontsize=15, loc="left")

        # The ONLY channel that is genuinely independent of the tier is
        # w_consensus: after M4 integration, Tier-1 requires a panel ACCEPT,
        # so w_m4 is (partly) determined by the same decision that sets the
        # tier — plotting it "by tier" would be circular. Paper support is
        # counted from corpus co-occurrence and knows nothing about tiers.
        ax = axes[1]
        p_t1 = [p_ for p_, t in zip(papers, tiers) if t == 1]
        p_t2 = [p_ for p_, t in zip(papers, tiers) if t == 2]
        if p_t1 and p_t2:
            hi = max(max(p_t1), max(p_t2))
            pbins = np.arange(-0.5, min(hi, 10) + 1.5, 1)
            ax.hist([np.clip(p_t1, 0, 10), np.clip(p_t2, 0, 10)],
                    bins=pbins, stacked=True, color=[BLUE, TERRACOTTA],
                    label=[f"Tier-1 (n={len(p_t1)})",
                           f"Tier-2 (n={len(p_t2)})"])
            m1, m2 = float(np.mean(p_t1)), float(np.mean(p_t2))
            md1 = float(np.median(p_t1))
            md2 = float(np.median(p_t2))
            _, pv = mannwhitney_p(p_t1, p_t2)
            ptxt = ("p < 0.001" if pv == pv and pv < 0.001
                    else (f"p = {pv:.3f}" if pv == pv else "p n/a"))
            sig = "no detectable difference" if (pv != pv or pv >= 0.05) \
                else "difference detected"
            ax.set_title("B. Independent channel: paper support by tier\n"
                         f"means {m1:.2f} / {m2:.2f}, medians "
                         f"{md1:.0f} / {md2:.0f}, {ptxt} — {sig}",
                         fontsize=12, loc="left")
            ax.set_xlabel("independent papers (clipped at 10)")
            ax.legend(frameon=False, fontsize=9)
        else:
            ax.text(0.5, 0.5, "tier split unavailable", ha="center",
                    transform=ax.transAxes)

        multi = sum(1 for p_ in papers if p_ >= 2)
        pooled = (f"Pooled over all triples, {100*multi/max(len(papers),1):.0f}"
                  f"% are supported by at least two independent papers "
                  f"(max {max(papers) if papers else 0}).")
        fig.text(0.5, -0.06, pooled, ha="center", fontsize=10, color="#444")
        fig.suptitle("Confidence and consensus profile of the knowledge "
                     "graph", fontsize=19, fontweight="bold", y=1.04)
        fig.tight_layout()
        fig.subplots_adjust(top=0.78, wspace=0.28)
        save(fig, "fig_paper_confidence_anatomy")

        fig, ax = plt.subplots(figsize=(7, 4.4))
        order = [k for k in ("exact", "fuzzy", "none") if k in match]
        cols = {"exact": BLUE, "fuzzy": GREEN, "none": TERRACOTTA}
        vals = [match[k] for k in order]
        ax.barh(range(len(order))[::-1], vals,
                color=[cols[k] for k in order])
        ax.set_yticks(range(len(order))[::-1])
        ax.set_yticklabels({"exact": "exact match", "fuzzy": "fuzzy match",
                            "none": "not anchored"}[k] for k in order)
        tot = sum(vals)
        for i, v in enumerate(vals):
            ax.text(v + tot * 0.01, len(order) - 1 - i,
                    f"{v} ({100*v/tot:.1f}%)", va="center", fontsize=10)
        ax.set_xlim(0, tot * 1.18)
        ax.set_xlabel("triples")
        ax.set_title("Evidence anchoring to the source corpus", fontsize=18)
        fig.tight_layout()
        save(fig, "fig_paper_provenance")
    else:
        skipped.append("fig_paper_confidence_anatomy / fig_paper_provenance "
                       f"({prov} missing — run 08_rebuild_provenance.py)")

    # ── manifest ──────────────────────────────────────────────────────
    manifest = [
        "# Manuscript figure manifest", "",
        "| Figure | Generator | Status |", "|---|---|---|",
        "| Pipeline overview | `pipeline/plot_pipeline_overview.py` | "
        "existing |",
        "| Retrieval comparison | `pipeline/plot_retrieval_comparison.py` | "
        "existing (hard-coded values — check against run13) |",
        "| Corpus diagnostic / failure modes | "
        "`pipeline/plot_corpus_diagnostic.py` | existing |",
        "| KG subgraph, vignette | `pipeline/plot_kg_subgraph.py`, "
        "`plot_vignette_subgraph.py` | existing (hand-curated triples — "
        "verify they still exist in run13) |",
        "| KG portrait, descriptors, relations, graph, growth | "
        "`analysis_suite/10–13` via `run_full_analysis.sh` | regenerated "
        "per run |",
        "| M4 verdicts, blind-vs-evidence, panel agreement, tier flow | "
        "`m4/m4_figures.py`, `m4/m4_figures_v2.py` | **regenerate for "
        "run13** |",
        "| Contamination ablation | `analysis_suite/18_paper_figures.py` | "
        "new |",
        "| Frozen protocol recall | `analysis_suite/18_paper_figures.py` | "
        "new |",
        "| Confidence & consensus | `analysis_suite/18_paper_figures.py` | "
        "new |",
        "| Evidence anchoring | `analysis_suite/18_paper_figures.py` | "
        "new |", "",
        "## Regenerate the M4 figures for run13", "",
        "```bash",
        "python m4/m4_figures.py \\",
        "    --decisions output/run13/m4_panel/m4_panel_decisions.jsonl \\",
        "    --output    figures/paper/m4_run13",
        "python m4/m4_figures.py \\",
        "    --decisions output/run13/m4/m4_decisions.jsonl \\",
        "    --output    figures/paper/m4_run13_llama",
        "python m4/m4_figures_v2.py \\",
        "    --panel  output/run13/m4_panel/m4_panel_report.json \\",
        "    --output figures/paper/m4_run13",
        "```", "",
        f"Generated: {', '.join(made) if made else 'none'}",
    ]
    if skipped:
        manifest += ["", "Skipped:"] + [f"- {s}" for s in skipped]
    (outdir / "figure_manifest.md").write_text("\n".join(manifest),
                                               encoding="utf-8")

    print("=" * 62)
    print(f"{len(made)} figures written to {outdir} (pdf + png)")
    for s in skipped:
        print(f"  [skip] {s}")
    print(f"manifest: {outdir}/figure_manifest.md")


if __name__ == "__main__":
    main()