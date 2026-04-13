"""
plot_retrieval_comparison.py
Bold, fully opaque retrieval comparison figure.
Two panels: recall and silent query rate.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BG = '#FFFFFF'

configs      = ['Memory only\n(no retrieval)', 'C9\n(BM25 only)',
                'C10\n(BM25 +\ncross-encoder\nre-ranking)']
recall       = [34.6, 50.0, 69.2]
silent       = [88.0, 58.2, 41.8]
bar_colors   = ['#9E9E9E', '#90CAF9', '#1565C0']
edge_colors  = ['#000000', '#000000', '#000000']

x     = np.arange(len(configs))
WIDTH = 0.52

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig.patch.set_facecolor(BG)

# ══════════════════════════════════════════════════════════════════════
# Panel 1 — Recall
# ══════════════════════════════════════════════════════════════════════
bars1 = ax1.bar(x, recall, WIDTH,
                color=bar_colors,
                edgecolor=edge_colors,
                linewidth=1.5,
                zorder=3)

# Corpus-gap corrected ceiling
ax1.axhline(75.0, color='#E65100', lw=2.0, ls='--', zorder=4,
            label='Corpus-gap corrected ceiling (75.0%)')

# Value labels on bars
for bar, val in zip(bars1, recall):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 1.2,
             f'{val:.1f}%',
             ha='center', va='bottom',
             fontsize=12, fontweight='bold',
             color='#000000', zorder=5)

# Gain annotations
for i in range(1, len(recall)):
    gain = recall[i] - recall[i-1]
    ax1.annotate('',
        xy=(x[i], recall[i]),
        xytext=(x[i-1], recall[i-1]),
        arrowprops=dict(
            arrowstyle='->', color='#000000',
            lw=1.8, mutation_scale=16,
        ),
        zorder=6)
    mid_x = (x[i] + x[i-1]) / 2
    mid_y = (recall[i] + recall[i-1]) / 2
    ax1.text(mid_x + 0.08, mid_y,
             f'+{gain:.1f} pp',
             ha='left', va='center',
             fontsize=10, fontweight='bold',
             color='#000000',
             bbox=dict(boxstyle='round,pad=0.2',
                       fc='#FFFFFF', ec='#000000',
                       linewidth=1.0, alpha=1.0),
             zorder=7)

ax1.set_ylabel('Reference-edge recall (%)',
               fontsize=12, fontweight='bold', color='#000000')
ax1.set_title('Benchmark recall\nacross retrieval configurations',
              fontsize=12, fontweight='bold', color='#000000', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(configs, fontsize=11, fontweight='bold',
                    color='#000000')
ax1.set_ylim(0, 90)
ax1.tick_params(axis='y', labelsize=10)
for label in ax1.get_yticklabels():
    label.set_fontweight('bold')
    label.set_color('#000000')

legend = ax1.legend(fontsize=10, framealpha=1.0,
                    edgecolor='#000000', loc='upper left')
for text in legend.get_texts():
    text.set_fontweight('bold')
    text.set_color('#000000')

# Gridlines
ax1.yaxis.grid(True, color='#DDDDDD', lw=1.0, zorder=0)
ax1.set_axisbelow(True)

# Spines
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(1.5)
ax1.spines['bottom'].set_linewidth(1.5)
for spine in ax1.spines.values():
    spine.set_edgecolor('#000000')

# ══════════════════════════════════════════════════════════════════════
# Panel 2 — Silent query rate
# ══════════════════════════════════════════════════════════════════════
bars2 = ax2.bar(x, silent, WIDTH,
                color=bar_colors,
                edgecolor=edge_colors,
                linewidth=1.5,
                zorder=3)

# Value labels
for bar, val in zip(bars2, silent):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 1.2,
             f'{val:.1f}%',
             ha='center', va='bottom',
             fontsize=12, fontweight='bold',
             color='#000000', zorder=5)

# Reduction annotations
for i in range(1, len(silent)):
    reduction = silent[i-1] - silent[i]
    ax2.annotate('',
        xy=(x[i], silent[i]),
        xytext=(x[i-1], silent[i-1]),
        arrowprops=dict(
            arrowstyle='->', color='#1B5E20',
            lw=1.8, mutation_scale=16,
        ),
        zorder=6)
    mid_x = (x[i] + x[i-1]) / 2
    mid_y = (silent[i] + silent[i-1]) / 2
    ax2.text(mid_x + 0.08, mid_y,
             f'\u2212{reduction:.1f} pp',
             ha='left', va='center',
             fontsize=10, fontweight='bold',
             color='#1B5E20',
             bbox=dict(boxstyle='round,pad=0.2',
                       fc='#FFFFFF', ec='#1B5E20',
                       linewidth=1.0, alpha=1.0),
             zorder=7)

ax2.set_ylabel('Silent query rate (%)',
               fontsize=12, fontweight='bold', color='#000000')
ax2.set_title('Fraction of queries producing\nno extracted triples',
              fontsize=12, fontweight='bold', color='#000000', pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(configs, fontsize=11, fontweight='bold',
                    color='#000000')
ax2.set_ylim(0, 100)
ax2.tick_params(axis='y', labelsize=10)
for label in ax2.get_yticklabels():
    label.set_fontweight('bold')
    label.set_color('#000000')

# Annotation: what silent rate means
ax2.text(2, 41.8 - 8,
         '39.4% of queries\nstill produce\nno output',
         ha='center', va='top',
         fontsize=9, fontweight='bold',
         color='#B71C1C',
         bbox=dict(boxstyle='round,pad=0.3',
                   fc='#FFFFFF', ec='#B71C1C',
                   linewidth=1.2, alpha=1.0),
         zorder=8)

ax2.yaxis.grid(True, color='#DDDDDD', lw=1.0, zorder=0)
ax2.set_axisbelow(True)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(1.5)
ax2.spines['bottom'].set_linewidth(1.5)
for spine in ax2.spines.values():
    spine.set_edgecolor('#000000')

# ── Shared color legend ───────────────────────────────────────────────
patches = [
    mpatches.Patch(facecolor='#9E9E9E', edgecolor='#000000',
                   linewidth=1.2, label='Memory only (no retrieval)'),
    mpatches.Patch(facecolor='#90CAF9', edgecolor='#000000',
                   linewidth=1.2, label='C9 — BM25 only'),
    mpatches.Patch(facecolor='#1565C0', edgecolor='#000000',
                   linewidth=1.2, label='C10 — BM25 + cross-encoder re-ranking'),
]
fig.legend(handles=patches,
           loc='lower center',
           ncol=3,
           fontsize=10,
           framealpha=1.0,
           edgecolor='#000000',
           bbox_to_anchor=(0.5, -0.04),
           title='Configuration',
           title_fontsize=10)

# Make legend title bold after creation
leg = fig.legends[-1]
leg.get_title().set_fontweight('bold')
leg.get_title().set_color('#000000')
for text in leg.get_texts():
    text.set_fontweight('bold')
    text.set_color('#000000')

plt.suptitle(
    'Effect of retrieval grounding on pipeline performance',
    fontsize=13, fontweight='bold', color='#000000', y=1.01)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('figures/fig_retrieval_comparison.pdf',
            dpi=300, bbox_inches='tight', facecolor=BG)
plt.savefig('figures/fig_retrieval_comparison.png',
            dpi=300, bbox_inches='tight', facecolor=BG)
print('Saved figures/fig_retrieval_comparison.pdf and .png')
