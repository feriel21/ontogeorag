#!/usr/bin/env python3
"""
m4_calibration.py — V4: Confidence-score calibration
=====================================================

Question answered: does the M4 confidence score (0.7*evidence +
0.3*blind) MEAN anything as a probability — i.e., among triples scored
around c, is the fraction of genuine (non-corrupted) pairs actually
close to c?

Ground truth comes from the V2 negative controls, where labels are known
BY CONSTRUCTION: corrupted items (label 0) are wrong by design; original
items (label 1) are the pipeline's own outputs. Declared caveat, written
into the report: originals are NOT certified correct (the main run finds
~20% of them ungrounded), so measured "accuracy" among high-confidence
originals is a lower bound and the resulting ECE is a CONSERVATIVE
(pessimistic) estimate of calibration.

Outputs:
  calibration_report.json     ECE, MCE, Brier score, per-bin table
  fig_m4_reliability.pdf/.png reliability diagram + confidence histogram
                              (Okabe-Ito, single-column, 300 dpi)

Metrics:
  ECE  = sum_b (n_b/N) * |acc_b - conf_b|     (expected calibration error)
  MCE  = max_b |acc_b - conf_b|               (maximum calibration error)
  Brier = mean (conf - label)^2

Binning: the confidence score is discrete (9 possible values from the
3x3 verdict grid), so bins are the UNIQUE SCORE VALUES (cleaner than
arbitrary equal-width bins for a discrete score); --bins N switches to
N equal-width bins if preferred.

Usage:
    python m4_calibration.py \
        --controls ~/ontogeorag/output/m4/negatives/controls.jsonl \
        --verdicts ~/ontogeorag/output/m4/negatives/m4_verdicts.jsonl \
        --output   ~/ontogeorag/output/m4/calibration
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from m4_aggregate import confidence

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "grey": "#999999"}
MM = 1 / 25.4

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_jsonl(path):
    out = []
    with open(Path(path).expanduser(), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def compute_bins(scores, labels, edges=None):
    """Group by unique score value (edges=None) or by equal-width bins."""
    groups = defaultdict(list)
    if edges is None:
        for s, y in zip(scores, labels):
            groups[round(s, 3)].append(y)
        bins = []
        for s in sorted(groups):
            ys = groups[s]
            bins.append({"conf": s, "n": len(ys),
                         "accuracy": round(sum(ys) / len(ys), 4)})
        return bins
    idx = np.clip(np.digitize(scores, edges) - 1, 0, len(edges) - 2)
    for i, y, s in zip(idx, labels, scores):
        groups[i].append((s, y))
    bins = []
    for i in sorted(groups):
        pairs = groups[i]
        ss = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        bins.append({"conf": round(float(np.mean(ss)), 4), "n": len(ys),
                     "accuracy": round(sum(ys) / len(ys), 4),
                     "bin_range": [round(float(edges[i]), 3),
                                   round(float(edges[i + 1]), 3)]})
    return bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", required=True,
                    help="controls.jsonl from m4_negatives.py generate")
    ap.add_argument("--verdicts", required=True,
                    help="m4_verdicts.jsonl from the V2 verification run")
    ap.add_argument("--output", required=True)
    ap.add_argument("--bins", type=int, default=0,
                    help="0 = one bin per unique score value (default); "
                         "N>0 = N equal-width bins")
    args = ap.parse_args()

    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    controls = load_jsonl(args.controls)
    verdicts = {v["m4_index"]: v for v in load_jsonl(args.verdicts)}

    scores, labels, classes = [], [], []
    for i, c in enumerate(controls):
        v = verdicts.get(i)
        if v is None:
            continue
        bl = v["blind"]["verdict"]
        ev = v["evidence"]["verdict"]
        if ev not in ("SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"):
            continue
        scores.append(confidence(bl, ev))
        labels.append(1 if c["label"] == "original" else 0)
        classes.append(c["corruption_class"])
    scores = np.array(scores)
    labels = np.array(labels)
    n = len(scores)

    edges = (np.linspace(0, 1, args.bins + 1) if args.bins > 0 else None)
    bins = compute_bins(scores, labels, edges)

    ece = sum(b["n"] / n * abs(b["accuracy"] - b["conf"]) for b in bins)
    mce = max(abs(b["accuracy"] - b["conf"]) for b in bins)
    brier = float(np.mean((scores - labels) ** 2))

    report = {
        "n_items": n,
        "n_original": int(labels.sum()),
        "n_corrupted": int(n - labels.sum()),
        "binning": ("unique score values" if edges is None
                    else f"{args.bins} equal-width bins"),
        "ECE": round(float(ece), 4),
        "MCE": round(float(mce), 4),
        "brier_score": round(brier, 4),
        "bins": bins,
        "caveat": ("Labels are known by construction for corrupted items "
                   "only; 'original' items are pipeline outputs, not "
                   "certified correct (the main run finds ~20% of them "
                   "ungrounded). Accuracy among originals is therefore a "
                   "lower bound and the reported ECE is a conservative "
                   "(pessimistic) estimate of calibration."),
    }
    (out_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    # ── figure: reliability diagram + confidence histogram ─────────────
    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(90 * MM, 85 * MM), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})

    ax.plot([0, 1], [0, 1], "--", color=OI["grey"], lw=1,
            label="Perfect calibration")
    confs = [b["conf"] for b in bins]
    accs = [b["accuracy"] for b in bins]
    ns = [b["n"] for b in bins]
    sizes = [20 + 180 * (x / max(ns)) for x in ns]
    ax.scatter(confs, accs, s=sizes, color=OI["blue"], zorder=3,
               edgecolor="white", lw=0.8,
               label="Observed (size ∝ n)")
    for b in bins:
        ax.plot([b["conf"], b["conf"]], [b["conf"], b["accuracy"]],
                color=OI["red"], lw=1, alpha=0.6, zorder=2)
    ax.set_ylabel("Fraction genuine (non-corrupted)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.97, 0.05,
            f"ECE = {report['ECE']:.3f}\nBrier = {report['brier_score']:.3f}",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=OI["grey"], lw=0.5))

    axh.bar(confs, ns, width=0.04, color=OI["grey"], alpha=0.8)
    axh.set_xlabel("M4 confidence score")
    axh.set_ylabel("n")

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_m4_reliability.{ext}",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)

    print(json.dumps({k: v for k, v in report.items() if k != "bins"},
                     indent=2))
    print("\nPer-bin table:")
    for b in bins:
        print(f"  conf={b['conf']:.3f}  n={b['n']:>3d}  "
              f"acc={b['accuracy']:.3f}")
    print(f"\nReport: {out_dir/'calibration_report.json'}")
    print(f"Figure: {out_dir/'fig_m4_reliability.pdf'} (+ .png)")


if __name__ == "__main__":
    main()