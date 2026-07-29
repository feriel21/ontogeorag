#!/usr/bin/env python3
"""
m4_figures.py — Publication figures for the M4 validation section
==================================================================

Generates the figures for the manuscript (Computers & Geosciences) from
the M4 output files ON DISK — no hard-coded numbers, every value is read
from m4_decisions.jsonl / m4_verdicts.jsonl at run time.

Figures (each saved as vector PDF + 300-dpi PNG):

  fig_m4_verdicts.pdf      (a) evidence-verdict distribution by tier,
                           stacked horizontal bars — the honest successor
                           of the H_T1 bar (shows the 3-class structure)
  fig_m4_blind_vs_evidence.pdf
                           3x3 heatmap blind x evidence with counts —
                           visualises the over-interpretation index (the
                           PLAUSIBLE x NOT_SUPPORTED cell)
  fig_m4_vs_qwen.pdf       3x3 heatmap Qwen (mapped) x M4 evidence —
                           full-coverage successor of Exp D; shows the
                           degenerate Qwen margin that explains the low
                           kappa
  fig_m4_by_relation.pdf   ACCEPT/UNCERTAIN/REJECT by relation type,
                           stacked bars sorted by triple count —
                           highlights hasDescriptor
  fig_m4_tier_flow.pdf     tier reassignment flow (original -> new),
                           drawn as a simple two-column alluvial

Style: Okabe-Ito colorblind-safe palette, no chartjunk, single-column
width (~90 mm) at 9 pt — Elsevier-compatible.

Usage:
    python m4_figures.py \
        --decisions ~/ontogeorag/output/m4/m4_decisions.jsonl \
        --output    ~/ontogeorag/output/m4/figures
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito colorblind-safe palette
OI = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "purple": "#CC79A7",
    "grey": "#999999",
}

DEC_COLORS = {
    "ACCEPT": OI["green"],
    "UNCERTAIN": OI["orange"],
    "REJECT": OI["red"],
}
EV_COLORS = {
    "SUPPORTED": OI["green"],
    "PARTIALLY_SUPPORTED": OI["sky"],
    "NOT_SUPPORTED": OI["red"],
    "NO_PASSAGE": OI["grey"],
}

EV_ORDER = ["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"]
BLIND_ORDER = ["PLAUSIBLE", "UNCERTAIN", "IMPLAUSIBLE"]
DEC_ORDER = ["ACCEPT", "UNCERTAIN", "REJECT"]
QWEN_TO_M4 = {
    "STRONG_SUPPORT": "SUPPORTED",
    "WEAK_SUPPORT": "PARTIALLY_SUPPORTED",
    "NOT_SUPPORTED": "NOT_SUPPORTED",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

MM = 1 / 25.4  # mm -> inch


def load_jsonl(path):
    """Read `path` as one JSON object per line; no side effects."""
    out = []
    with open(Path(path).expanduser(), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save(fig, out_dir, name):
    """Save `fig` as `name`.pdf (vector) and `name`.png (300dpi) under `out_dir`, close it, and log the filenames."""
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def short(label):
    """Map a verdict/decision `label` to its compact figure-axis form; returns `label` unchanged if not in the lookup, no side effects."""
    return {
        "PARTIALLY_SUPPORTED": "PARTIAL",
        "NOT_SUPPORTED": "NOT SUPP.",
        "STRONG_SUPPORT": "STRONG",
        "WEAK_SUPPORT": "WEAK",
        "IMPLAUSIBLE": "IMPLAUS.",
        "UNCERTAIN": "UNCERT.",
        "SUPPORTED": "SUPPORTED",
        "PLAUSIBLE": "PLAUSIBLE",
        "ACCEPT": "Accept",
        "REJECT": "Reject",
    }.get(label, label)


# ── Fig 1: evidence verdicts by tier ───────────────────────────────────


def fig_verdicts_by_tier(decisions, out_dir):
    """Render the evidence-verdict-distribution-by-tier stacked bar chart from `decisions` and save it as fig_m4_verdicts under `out_dir`."""
    tiers = sorted({str(d.get("tier")) for d in decisions})
    counts = {t: Counter() for t in tiers}
    for d in decisions:
        counts[str(d.get("tier"))][d["evidence_verdict"]] += 1

    fig, ax = plt.subplots(figsize=(90 * MM, 45 * MM))
    y = np.arange(len(tiers))
    left = np.zeros(len(tiers))
    for ev in EV_ORDER + ["NO_PASSAGE"]:
        vals = np.array([counts[t].get(ev, 0) for t in tiers], dtype=float)
        if vals.sum() == 0:
            continue
        ax.barh(
            y,
            vals,
            left=left,
            color=EV_COLORS[ev],
            label=short(ev),
            height=0.55,
            edgecolor="white",
            lw=0.5,
        )
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v > 0:
                ax.text(
                    l + v / 2,
                    yi,
                    f"{int(v)}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if ev != "PARTIALLY_SUPPORTED" else "black",
                )
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"Tier {t}\n(n={sum(counts[t].values())})" for t in tiers]
    )
    ax.set_xlabel("Triples")
    ax.legend(
        ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.28)
    )
    ax.invert_yaxis()
    save(fig, out_dir, "fig_m4_verdicts")


# ── Fig 2 & 3: heatmaps ────────────────────────────────────────────────


def heatmap(matrix, rows, cols, xlabel, ylabel, out_dir, name, highlight=None):
    """Render `matrix` as an annotated Blues heatmap with `rows`/`cols` labels (optionally outlining cell `highlight`) and save it as `name` under `out_dir`."""
    fig, ax = plt.subplots(figsize=(80 * MM, 62 * MM))
    m = np.array(matrix, dtype=float)
    im = ax.imshow(
        m, cmap="Blues", aspect="auto", vmin=0, vmax=max(1, m.max())
    )
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = int(m[i, j])
            color = "white" if m[i, j] > 0.6 * m.max() else "black"
            weight = "bold" if highlight == (i, j) else "normal"
            ax.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                color=color,
                fontsize=9,
                fontweight=weight,
            )
    if highlight is not None:
        i, j = highlight
        ax.add_patch(
            plt.Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor=OI["red"],
                lw=1.8,
            )
        )
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([short(c) for c in cols], rotation=20, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([short(r) for r in rows])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Triples")
    save(fig, out_dir, name)


def fig_blind_vs_evidence(decisions, out_dir):
    """Build the blind x evidence verdict count matrix from `decisions` and render it as the fig_m4_blind_vs_evidence heatmap (highlighting the PLAUSIBLE x NOT_SUPPORTED over-interpretation cell) under `out_dir`."""
    m = [[0] * len(EV_ORDER) for _ in BLIND_ORDER]
    for d in decisions:
        b, e = d["blind_verdict"], d["evidence_verdict"]
        if b in BLIND_ORDER and e in EV_ORDER:
            m[BLIND_ORDER.index(b)][EV_ORDER.index(e)] += 1
    heatmap(
        m,
        BLIND_ORDER,
        EV_ORDER,
        "Evidence judge (with source passage)",
        "Blind judge (no text)",
        out_dir,
        "fig_m4_blind_vs_evidence",
        highlight=(
            BLIND_ORDER.index("PLAUSIBLE"),
            EV_ORDER.index("NOT_SUPPORTED"),
        ),
    )


def fig_m4_vs_qwen(decisions, out_dir):
    """Build the Qwen(mapped) x M4-evidence verdict count matrix from `decisions` and render it as the fig_m4_vs_qwen heatmap under `out_dir`."""
    m = [[0] * len(EV_ORDER) for _ in EV_ORDER]
    for d in decisions:
        q = QWEN_TO_M4.get(d.get("qwen_verdict", ""))
        e = d["evidence_verdict"]
        if q in EV_ORDER and e in EV_ORDER:
            m[EV_ORDER.index(q)][EV_ORDER.index(e)] += 1
    heatmap(
        m,
        EV_ORDER,
        EV_ORDER,
        "M4 independent verifier (Llama-3.1-8B)",
        "Pipeline self-verifier (Qwen-7B, mapped)",
        out_dir,
        "fig_m4_vs_qwen",
    )


# ── Fig 4: decisions by relation ───────────────────────────────────────


def fig_by_relation(decisions, out_dir):
    """Render the ACCEPT/UNCERTAIN/REJECT-by-relation stacked bar chart (relations sorted by triple count) from `decisions` and save it as fig_m4_by_relation under `out_dir`."""
    by_rel = defaultdict(Counter)
    for d in decisions:
        by_rel[d["relation"]][d["m4_decision"]] += 1
    rels = sorted(by_rel, key=lambda r: -sum(by_rel[r].values()))

    fig, ax = plt.subplots(figsize=(90 * MM, 55 * MM))
    y = np.arange(len(rels))
    left = np.zeros(len(rels))
    for dec in DEC_ORDER:
        vals = np.array([by_rel[r].get(dec, 0) for r in rels], dtype=float)
        ax.barh(
            y,
            vals,
            left=left,
            color=DEC_COLORS[dec],
            label=short(dec),
            height=0.6,
            edgecolor="white",
            lw=0.5,
        )
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v > 1:
                ax.text(
                    l + v / 2,
                    yi,
                    f"{int(v)}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                )
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(rels)
    ax.set_xlabel("Triples")
    ax.legend(
        ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18)
    )
    ax.invert_yaxis()
    save(fig, out_dir, "fig_m4_by_relation")


# ── Fig 5: tier flow (two-column alluvial) ─────────────────────────────


def fig_tier_flow(decisions, out_dir):
    """Render the original-tier -> new-tier/quarantine two-column alluvial flow diagram from `decisions` and save it as fig_m4_tier_flow under `out_dir`."""
    flows = Counter()
    for d in decisions:
        orig = f"Tier {d.get('tier')}"
        dec = d["m4_decision"]
        if dec == "REJECT":
            new = "Quarantine"
        elif dec == "ACCEPT" and d.get("tier") == 1:
            new = "Tier 1"
        else:
            new = "Tier 2"
        flows[(orig, new)] += 1

    left_nodes = sorted({k[0] for k in flows})
    right_nodes = [
        n
        for n in ["Tier 1", "Tier 2", "Quarantine"]
        if any(k[1] == n for k in flows)
    ]

    left_tot = {
        n: sum(v for k, v in flows.items() if k[0] == n) for n in left_nodes
    }
    right_tot = {
        n: sum(v for k, v in flows.items() if k[1] == n) for n in right_nodes
    }
    total = sum(flows.values())
    gap = 0.04 * total

    def positions(nodes, tots):
        pos, y = {}, 0.0
        for n in nodes:
            pos[n] = (y, y + tots[n])
            y += tots[n] + gap
        return pos

    lp, rp = positions(left_nodes, left_tot), positions(right_nodes, right_tot)
    node_color = {
        "Tier 1": OI["green"],
        "Tier 2": OI["sky"],
        "Quarantine": OI["red"],
    }

    fig, ax = plt.subplots(figsize=(90 * MM, 60 * MM))
    for n, (y0, y1) in lp.items():
        ax.fill_betweenx(
            [y0, y1],
            0.00,
            0.06,
            color=node_color.get(n, OI["grey"]),
            alpha=0.9,
        )
        ax.text(
            -0.02,
            (y0 + y1) / 2,
            f"{n}\n({int(y1 - y0)})",
            ha="right",
            va="center",
            fontsize=8,
        )
    for n, (y0, y1) in rp.items():
        ax.fill_betweenx(
            [y0, y1],
            0.94,
            1.00,
            color=node_color.get(n, OI["grey"]),
            alpha=0.9,
        )
        ax.text(
            1.02,
            (y0 + y1) / 2,
            f"{n}\n({int(y1 - y0)})",
            ha="left",
            va="center",
            fontsize=8,
        )

    lcur = {n: lp[n][0] for n in left_nodes}
    rcur = {n: rp[n][0] for n in right_nodes}
    xs = np.linspace(0.06, 0.94, 80)
    ease = 0.5 * (1 - np.cos(np.pi * (xs - 0.06) / 0.88))
    for (a, b), v in sorted(flows.items()):
        y0a, y0b = lcur[a], rcur[b]
        top = y0a + (y0b - y0a) * ease
        ax.fill_between(
            xs,
            top,
            top + v,
            color=node_color.get(b, OI["grey"]),
            alpha=0.35,
            lw=0,
        )
        lcur[a] += v
        rcur[b] += v

    ax.set_xlim(-0.18, 1.22)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(
        "Tier reassignment after independent verification", fontsize=9
    )
    save(fig, out_dir, "fig_m4_tier_flow")


def main():
    """CLI entry point: loads --decisions and renders all M4 publication figures (skipping verdict-based ones for multi-judge panel files) to --output."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions = load_jsonl(args.decisions)
    print(f"Loaded {len(decisions)} decisions. Writing figures to {out_dir}:")

    # Panel files carry concatenated per-judge verdicts ("A|B"); the
    # verdict-based figures are only meaningful on single-judge files.
    is_panel = any(
        "|" in str(d.get("evidence_verdict", "")) for d in decisions
    )
    if is_panel:
        print(
            "  [panel file detected: verdict-based figures "
            "(fig_m4_verdicts, fig_m4_blind_vs_evidence, "
            "fig_m4_vs_qwen) are skipped — generate them from a "
            "single-judge m4_decisions.jsonl]"
        )
    else:
        fig_verdicts_by_tier(decisions, out_dir)
        fig_blind_vs_evidence(decisions, out_dir)
        fig_m4_vs_qwen(decisions, out_dir)
    fig_by_relation(decisions, out_dir)
    fig_tier_flow(decisions, out_dir)


if __name__ == "__main__":
    main()
