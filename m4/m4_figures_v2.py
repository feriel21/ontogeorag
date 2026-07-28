#!/usr/bin/env python3
"""
m4_figures_v2.py — Remaining publication figures (validation battery)
=======================================================================

Complements m4_figures.py with the three figures introduced by V2-V4:

  fig_m4_negatives_sensitivity  detection sensitivity per corruption
                                class, Wilson 95% CIs, AUC annotations
                                (reads negatives_report.json)
  fig_m4_direction              directional check outcome on accepted
                                directional triples (reads
                                m4_direction_summary.json)
  fig_m4_panel_agreement        per-judge evidence distributions +
                                inter-judge kappa vs human inter-expert
                                kappa band (reads m4_panel_report.json)

All values are read from the report files ON DISK — nothing hard-coded.
Okabe-Ito palette, Elsevier single-column, vector PDF + 300-dpi PNG.

Usage:
    python m4_figures_v2.py \
        --negatives ~/ontogeorag/output/m4/negatives/negatives_report.json \
        --direction ~/ontogeorag/output/m4/m4_direction_summary.json \
        --panel     ~/ontogeorag/output/m4_panel/m4_panel_report.json \
        --output    ~/ontogeorag/output/m4/figures
(Any of --negatives/--direction/--panel may be omitted; only the
figures whose input is provided are generated.)
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "sky": "#56B4E9", "yellow": "#F0E442",
      "purple": "#CC79A7", "grey": "#999999"}
MM = 1 / 25.4

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def save(fig, out_dir, name):
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  {name}.pdf / .png")


# ── Fig: sensitivity per corruption class ──────────────────────────────

CLASS_LABELS = {"passage_perm": "Passage\npermutation",
                "entity_sub": "Entity\nsubstitution",
                "relation_sub": "Relation\nsubstitution",
                "inversion": "Direction\ninversion"}


def fig_sensitivity(report_path, out_dir):
    rep = json.loads(Path(report_path).expanduser().read_text(
        encoding="utf-8"))
    per = rep["per_corruption_class"]
    # order by decreasing sensitivity
    classes = sorted(per, key=lambda c: -per[c]["sensitivity"])

    sens, lo, hi, ns, aucs = [], [], [], [], []
    for c in classes:
        m = per[c]
        k, n = m["tp"], m["tp"] + m["fn"]
        s = m["sensitivity"]
        l, h = wilson(k, n)
        sens.append(s)
        lo.append(s - l)
        hi.append(h - s)
        ns.append(n)
        aucs.append(m.get("roc_auc"))

    fig, ax = plt.subplots(figsize=(90 * MM, 55 * MM))
    x = np.arange(len(classes))
    bars = ax.bar(x, sens, width=0.62, color=OI["blue"],
                  edgecolor="white", lw=0.5)
    ax.errorbar(x, sens, yerr=[lo, hi], fmt="none", ecolor="black",
                elinewidth=1, capsize=3)
    for xi, (s, n, a) in enumerate(zip(sens, ns, aucs)):
        ax.text(xi, 0.03, f"n={n}", ha="center", va="bottom",
                fontsize=7, color="white")
        if a is not None:
            ax.text(xi, min(s + hi[xi] + 0.05, 1.06),
                    f"AUC {a:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS.get(c, c) for c in classes])
    ax.set_ylabel("Detection sensitivity")
    ax.set_ylim(0, 1.18)
    ax.axhline(1.0, color=OI["grey"], lw=0.5, ls=":")
    save(fig, out_dir, "fig_m4_negatives_sensitivity")


# ── Fig: direction check outcome ───────────────────────────────────────

DIR_COLORS = {"FORWARD": OI["green"], "REVERSE": OI["red"],
              "UNDIRECTED": OI["orange"], "UNPARSEABLE": OI["grey"],
              "ABSENT": OI["purple"]}


def fig_direction(summary_path, out_dir):
    rep = json.loads(Path(summary_path).expanduser().read_text(
        encoding="utf-8"))
    verdicts = rep["verdicts"]
    n = rep["n_checked"]
    order = [v for v in ("FORWARD", "REVERSE", "UNDIRECTED",
                         "ABSENT", "UNPARSEABLE") if v in verdicts]

    fig, ax = plt.subplots(figsize=(90 * MM, 32 * MM))
    left = 0.0
    for v in order:
        c = verdicts[v]
        ax.barh([0], [c], left=left, color=DIR_COLORS[v], height=0.5,
                edgecolor="white", lw=0.8, label=f"{v} ({c})")
        if c / n > 0.05:
            k = c
            l, h = wilson(k, n)
            ax.text(left + c / 2, 0, f"{c}\n{c/n:.0%}", ha="center",
                    va="center", fontsize=8,
                    color="white" if v == "FORWARD" else "black")
        left += c
    ax.set_xlim(0, n)
    ax.set_yticks([])
    ax.set_xlabel(f"Accepted directional triples (n={n})")
    ax.legend(ncol=len(order), frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.55), fontsize=7,
              columnspacing=0.9, handlelength=1.2)
    save(fig, out_dir, "fig_m4_direction")


# ── Fig: panel agreement vs human reference ────────────────────────────

def fig_panel(panel_path, out_dir):
    rep = json.loads(Path(panel_path).expanduser().read_text(
        encoding="utf-8"))
    names = rep["judges"]
    ev = rep["evidence_distributions"]
    order = ["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"]
    colors = {"SUPPORTED": OI["green"],
              "PARTIALLY_SUPPORTED": OI["sky"],
              "NOT_SUPPORTED": OI["red"]}
    short = {"SUPPORTED": "SUPPORTED", "PARTIALLY_SUPPORTED": "PARTIAL",
             "NOT_SUPPORTED": "NOT SUPP."}

    fig, (ax, axk) = plt.subplots(
        1, 2, figsize=(140 * MM, 50 * MM),
        gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.35})

    # left: per-judge evidence distributions (stacked)
    y = np.arange(len(names))
    left = np.zeros(len(names))
    for v in order:
        vals = np.array([ev[n].get(v, 0) for n in names], dtype=float)
        ax.barh(y, vals, left=left, color=colors[v], height=0.55,
                label=short[v], edgecolor="white", lw=0.5)
        for yi, (val, l) in enumerate(zip(vals, left)):
            if val > 6:
                ax.text(l + val / 2, yi, f"{int(val)}", ha="center",
                        va="center", fontsize=8,
                        color="black" if v == "PARTIALLY_SUPPORTED"
                        else "white")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([n.capitalize() for n in names])
    ax.set_xlabel("Triples")
    ax.invert_yaxis()
    ax.legend(ncol=3, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.32), fontsize=7)

    # right: kappa comparison vs human band
    pairs = rep["inter_judge"]
    pair_key = list(pairs.keys())[0]
    k_ev = pairs[pair_key]["evidence"]
    k_dec = pairs[pair_key]["decisions"]
    labels = ["Evid.\nunw.", "Evid.\nlinear", "Dec.\nunw.", "Dec.\nlinear"]
    values = [k_ev["kappa_unweighted"], k_ev["kappa_linear"],
              k_dec["kappa_unweighted"], k_dec["kappa_linear"]]

    href = rep.get("human_reference", {}).get(
        "inter_expert_kappa_section_4_4")
    if href:
        axk.axhspan(min(href), max(href), color=OI["yellow"], alpha=0.45,
                    label=f"Inter-expert κ\n({min(href)}–{max(href)})")
    xk = np.arange(len(labels))
    axk.bar(xk, values, width=0.6, color=OI["blue"],
            edgecolor="white", lw=0.5)
    for xi, v in zip(xk, values):
        axk.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=7)
    axk.set_xticks(xk)
    axk.set_xticklabels(labels, fontsize=7)
    axk.set_ylabel("Cohen's κ (inter-judge)")
    axk.set_ylim(0, max(values + ([max(href)] if href else [])) + 0.12)
    axk.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, out_dir, "fig_m4_panel_agreement")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", default=None)
    ap.add_argument("--direction", default=None)
    ap.add_argument("--panel", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {out_dir}:")
    if args.negatives:
        fig_sensitivity(args.negatives, out_dir)
    if args.direction:
        fig_direction(args.direction, out_dir)
    if args.panel:
        fig_panel(args.panel, out_dir)


if __name__ == "__main__":
    main()