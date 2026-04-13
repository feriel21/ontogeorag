"""
plot_pipeline_overview.py — fixed legend/C9 overlap
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BG = '#FFFFFF'

fig, ax = plt.subplots(figsize=(17, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 17)
ax.set_ylim(0, 8)
ax.axis('off')

C_INPUT    = '#37474F'
C_RETRIEVAL= '#1565C0'
C_ONTOLOGY = '#4A148C'
C_LLM      = '#BF360C'
C_OUTPUT   = '#1B5E20'
C_EXPERT   = '#E65100'

boxes = [
    (1.4,  4.2, 2.2, 1.6, C_INPUT,
     '41-Paper\nCorpus',
     'PDF → plain text\nnormalized chunks'),

    (4.2,  4.2, 2.2, 1.6, C_RETRIEVAL,
     'BM25 Index\n+ Gating',
     '3,386 chunks\nk₁=1.5  b=0.75'),

    (7.0,  4.2, 2.2, 1.6, C_ONTOLOGY,
     'Ontology-Guided\nQuery Gen.',
     '249 queries\n4 strategies'),

    (9.8,  4.2, 2.2, 1.6, C_LLM,
     'LLM Extraction\n(Qwen 7B)',
     'Pass A (T=0)\nPass B (T=0.3)'),

    (12.6, 4.2, 2.2, 1.6, C_LLM,
     'Verification\n& Validation',
     'Strong / Weak /\nNot supported'),

    (15.4, 4.2, 2.2, 1.6, C_OUTPUT,
     'Final KG\n(Tiered)',
     'Tier-1: 101\nTier-1+2: 153'),
]

rerank_box  = (5.6,  2.0, 2.8, 1.2, C_RETRIEVAL,
               'CrossEncoder Re-ranking\n(C10 only)',
               'ms-marco-MiniLM-L-6-v2\nTop-20 → Top-5')

ontology_box= (8.4,  6.8, 2.8, 1.0, C_ONTOLOGY,
               'LB2019 Ontology',
               '88 nodes · 173 edges')

expert_box  = (12.6, 2.0, 2.2, 1.2, C_EXPERT,
               'Expert Validation',
               'κ=0.53  n=50\nrelaxed prec.')

def draw_box(ax, xc, yc, w, h, color, label, sublabel):
    rect = mpatches.FancyBboxPatch(
        (xc - w/2, yc - h/2), w, h,
        boxstyle='round,pad=0.12',
        facecolor=color, edgecolor='#000000',
        linewidth=1.8, zorder=3)
    ax.add_patch(rect)
    ax.text(xc, yc + 0.22, label,
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            color='#FFFFFF', zorder=4, linespacing=1.3)
    ax.text(xc, yc - 0.38, sublabel,
            ha='center', va='center',
            fontsize=7.5, fontweight='bold',
            color='#EEEEEE', zorder=4, linespacing=1.25)

def draw_arrow(ax, x0, y0, x1, y1,
               color='#000000', lw=2.0,
               label='', label_offset=(0, 0.22)):
    ax.annotate('',
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle='->', color=color,
            lw=lw, mutation_scale=20,
            connectionstyle='arc3,rad=0.0'),
        zorder=2)
    if label:
        mx = (x0+x1)/2 + label_offset[0]
        my = (y0+y1)/2 + label_offset[1]
        ax.text(mx, my, label,
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='#000000',
                bbox=dict(boxstyle='round,pad=0.2',
                          fc='#FFFFFF', ec='#000000',
                          linewidth=1.0, alpha=1.0),
                zorder=5)

# Draw all boxes
for b in boxes:
    draw_box(ax, *b)
draw_box(ax, *rerank_box)
draw_box(ax, *ontology_box)
draw_box(ax, *expert_box)

# Main horizontal arrows
main_y = 4.2
for (x0, x1) in [(1.4+1.1, 4.2-1.1),
                  (4.2+1.1, 7.0-1.1),
                  (7.0+1.1, 9.8-1.1),
                  (9.8+1.1, 12.6-1.1),
                  (12.6+1.1,15.4-1.1)]:
    draw_arrow(ax, x0, main_y, x1, main_y, lw=2.2)

# Ontology arrows
draw_arrow(ax, 7.6, 6.8-0.5, 7.0, 4.2+0.8,
           color='#4A148C', lw=1.8,
           label='schema', label_offset=(-0.55, 0))
draw_arrow(ax, 9.2, 6.8-0.5, 12.6, 4.2+0.8,
           color='#4A148C', lw=1.8,
           label='constraints', label_offset=(0.7, 0))

# Re-ranking arrows
draw_arrow(ax, 4.2, 4.2-0.8, 4.2, 2.0+0.6,
           color='#1565C0', lw=1.8,
           label='Top-20', label_offset=(0.5, 0))
draw_arrow(ax, 5.6+1.4, 2.0, 7.0, 4.2-0.8,
           color='#1565C0', lw=1.8,
           label='Top-5\nre-ranked', label_offset=(0.65, 0))

# C10 badge
ax.text(5.6, 2.0+0.82,
        'C10 only',
        ha='center', va='center',
        fontsize=8.5, fontweight='bold',
        color='#1565C0',
        bbox=dict(boxstyle='round,pad=0.18',
                  fc='#E3F2FD', ec='#1565C0',
                  linewidth=1.5, alpha=1.0),
        zorder=6)

# Expert validation arrow
draw_arrow(ax, 12.6, 4.2-0.8, 12.6, 2.0+0.6,
           color='#E65100', lw=1.8,
           label='sample', label_offset=(0.5, 0))

# ── C9 / C10 config bar — TOP of figure, clear of legend ─────────────
config_y = 7.65
ax.text(8.5, config_y,
        'C9: BM25 only (top-5 direct)                    '
        'C10: BM25 top-20  →  CrossEncoder re-ranking  →  top-5',
        ha='center', va='center',
        fontsize=10, fontweight='bold',
        color='#000000',
        bbox=dict(boxstyle='round,pad=0.35',
                  fc='#F5F5F5', ec='#000000',
                  linewidth=1.5, alpha=1.0),
        zorder=5)

# ── Title ─────────────────────────────────────────────────────────────
ax.set_title('OntoGeoRAG Pipeline Architecture',
             fontsize=14, fontweight='bold',
             color='#000000', pad=6, y=0.98)

# ── Legend — bottom LEFT, compact, 2 columns ──────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_INPUT,     edgecolor='#000000',
                   lw=1.2, label='Corpus / data'),
    mpatches.Patch(facecolor=C_RETRIEVAL, edgecolor='#000000',
                   lw=1.2, label='Retrieval'),
    mpatches.Patch(facecolor=C_ONTOLOGY,  edgecolor='#000000',
                   lw=1.2, label='Ontology'),
    mpatches.Patch(facecolor=C_LLM,       edgecolor='#000000',
                   lw=1.2, label='Language model'),
    mpatches.Patch(facecolor=C_OUTPUT,    edgecolor='#000000',
                   lw=1.2, label='Output KG'),
    mpatches.Patch(facecolor=C_EXPERT,    edgecolor='#000000',
                   lw=1.2, label='Expert validation'),
]
legend = ax.legend(
    handles=legend_items,
    loc='lower left',
    ncol=2,
    fontsize=9.5,
    framealpha=1.0,
    edgecolor='#000000',
    bbox_to_anchor=(0.0, 0.0),
    title='Component type',
    title_fontsize=10,
    borderpad=0.6,
    labelspacing=0.5)
legend.get_title().set_fontweight('bold')
for text in legend.get_texts():
    text.set_fontweight('bold')
    text.set_color('#000000')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('figures/fig01_pipeline_overview.pdf',
            dpi=300, bbox_inches='tight', facecolor=BG)
plt.savefig('figures/fig01_pipeline_overview.png',
            dpi=300, bbox_inches='tight', facecolor=BG)
print('Saved figures/fig01_pipeline_overview.pdf and .png')
