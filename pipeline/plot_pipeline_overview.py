"""
pipeline/plot_pipeline_overview.py — Pipeline Architecture Diagram
=======================================================================
WHY
    The paper needs a single figure explaining the end-to-end pipeline
    (corpus -> retrieval -> extraction -> verification -> KG) alongside
    the ontology and expert-validation side-paths, for readers who won't
    read the full methods section.

WHAT
    Hand-laid-out box-and-arrow diagram of the OntoGeoRAG pipeline,
    rendered to ontogeorag_pipeline_v3.png. Run directly (no CLI args).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- CONFIGURATION & STYLING ---
BG = '#FFFFFF'
C_INPUT     = '#37474F'
C_RETRIEVAL = '#1565C0'
C_ONTOLOGY  = '#4A148C'
C_LLM       = '#B71C1C' 
C_OUTPUT    = '#004D40' 
C_EXPERT    = '#E65100'

# Increase figure size for better spacing
fig, ax = plt.subplots(figsize=(18, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 17)
ax.set_ylim(0, 9.5) 
ax.axis('off')

def draw_box(ax, xc, yc, w, h, color, label, sublabel, title_fs=14, sub_fs=10.5):
    """Draws a node with optimized text filling and centering."""
    rect = mpatches.FancyBboxPatch(
        (xc - w/2, yc - h/2), w, h,
        boxstyle='round,pad=0.08',
        facecolor=color, edgecolor='none',
        zorder=3)
    ax.add_patch(rect)
    
    # Title - Positioned higher with tighter line spacing
    ax.text(xc, yc + 0.28, label,
            ha='center', va='center',
            fontsize=title_fs, fontweight='bold',
            color='#FFFFFF', zorder=4, linespacing=1.1)
    
    # Subtitle - Positioned lower to balance the box
    ax.text(xc, yc - 0.38, sublabel,
            ha='center', va='center',
            fontsize=sub_fs, fontweight='normal',
            color='#EEEEEE', zorder=4, linespacing=1.1)

def draw_bold_arrow(ax, x0, y0, x1, y1, color='#000000', lw=3.0, ms=25, rad=0.0):
    """Draws high-visibility bold arrows."""
    ax.annotate('',
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle='-|>, head_width=0.5, head_length=0.8', 
            color=color,
            lw=lw, 
            mutation_scale=ms,
            connectionstyle=f'arc3,rad={rad}'),
        zorder=2)

# --- 1. PRIMARY PIPELINE NODES (Centered on y=5.2) ---
main_y = 5.2
box_w, box_h = 2.4, 1.9

nodes = [
    (1.4,  main_y, box_w, box_h, C_INPUT, '41-Paper\nCorpus', 'PDF → plain text\nnormalized chunks'),
    (4.2,  main_y, box_w, box_h, C_RETRIEVAL, 'BM25 Index\n+ Gating', '3,386 chunks\nk₁=1.5  b=0.75'),
    (7.0,  main_y, box_w, box_h, C_ONTOLOGY, 'Ontology-Guided\nQuery Gen.', '249 queries\n4 strategies'),
    (9.8,  main_y, box_w, box_h, C_LLM, 'LLM Extraction\n(Qwen 7B)', 'Pass A (T=0)\nPass B (T=0.3)'),
    (12.6, main_y, box_w, box_h, C_LLM, 'Verification\n& Validation', 'Strong / Weak /\nNot supported'),
    (15.4, main_y, box_w, box_h, C_OUTPUT, 'Final KG\n(Tiered)', 'Tier-1: 101\nTier-1+2: 153'),
]

for n in nodes:
    draw_box(ax, *n)

# --- 2. SECONDARY NODES (Re-ranking, Ontology, Expert) ---
# Re-ranking Box (Lower)
draw_box(ax, 5.6, 2.5, 3.4, 1.6, C_RETRIEVAL, 'CrossEncoder Re-ranking\n(LLM-Rerank only)', 'ms-marco-MiniLM-L-6-v2\nTop 20 → Top-5', title_fs=12)
# Internal Badge for LLM-Rerank
ax.text(5.6, 1.95, 'LLM-Rerank only', ha='center', fontsize=10.5, fontweight='bold', 
        color=C_RETRIEVAL, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C_RETRIEVAL, lw=2), zorder=5)

# Ontology Definition (Upper)
draw_box(ax, 8.4, 8.0, 3.2, 1.3, C_ONTOLOGY, 'LB2019 Ontology', '88 nodes · 173 edges', title_fs=15)

# Expert Validation (Lower Right)
draw_box(ax, 13.2, 2.5, 2.8, 1.6, C_EXPERT, 'Expert Validation', 'κ=0.53  n=50\nrelaxed prec.', title_fs=13)

# --- 3. BOLD FLOW ARROWS ---
# Main horizontal flow
for (x0, x1) in [(2.6, 3.0), (5.4, 5.8), (8.2, 8.6), (11.0, 11.4), (13.8, 14.2)]:
    draw_bold_arrow(ax, x0, main_y, x1, main_y)

# Retrieval Logic (Labels outside to keep boxes clean)
draw_bold_arrow(ax, 4.2, 4.2, 4.2, 3.35, color=C_RETRIEVAL)
ax.text(3.5, 3.8, 'Top-20', fontsize=11, fontweight='bold', color=C_RETRIEVAL, ha='center')

draw_bold_arrow(ax, 7.0, 3.35, 7.0, 4.2, color=C_RETRIEVAL)
ax.text(7.7, 3.8, 'Top-5\nre-ranked', fontsize=11, fontweight='bold', color=C_RETRIEVAL, ha='center')

# Ontology Constraints (Connecting from top)
draw_bold_arrow(ax, 7.8, 7.3, 7.0, 6.2, color=C_ONTOLOGY, lw=2.0)
draw_bold_arrow(ax, 9.0, 7.3, 12.6, 6.2, color=C_ONTOLOGY, lw=2.0)

# Expert Sampling (Curved arrow)
draw_bold_arrow(ax, 12.6, 4.2, 13.2, 3.35, color=C_EXPERT, rad=-0.2)
ax.text(14.0, 4.0, 'sample', fontsize=11, fontweight='bold', color=C_EXPERT, ha='center')

# --- 4. TITLES, HEADERS & LEGEND ---
# Bold Black Title
ax.text(8.5, 9.2, 'OntoGeoRAG Pipeline Architecture', ha='center', va='center', 
        fontsize=24, fontweight='black', color='black')

# Sub-headers (Configuration modes)
ax.text(4.2, 8.6, 'LLM-BM25: word-matching search (top-5 direct)', ha='center', fontsize=11, fontweight='bold', color='#444444')
ax.text(12.6, 8.6, 'LLM-Rerank: word-matching + semantic re-ranking (top-20 → top-5)', ha='center', fontsize=11, fontweight='bold', color='#444444')

# Legend
legend_items = [
    mpatches.Patch(facecolor=C_INPUT, label='Corpus / Data'),
    mpatches.Patch(facecolor=C_LLM, label='Language Model'),
    mpatches.Patch(facecolor=C_RETRIEVAL, label='Retrieval (Search)'),
    mpatches.Patch(facecolor=C_OUTPUT, label='Knowledge Graph'),
    mpatches.Patch(facecolor=C_ONTOLOGY, label='Ontology Schema'),
    mpatches.Patch(facecolor=C_EXPERT, label='Expert Validation'),
]
legend = ax.legend(handles=legend_items, loc='lower left', ncol=3, 
                   bbox_to_anchor=(0.08, 0.05), title='Component Classification',
                   title_fontsize=13, frameon=True, prop={'weight':'bold', 'size':11})
plt.setp(legend.get_title(), fontweight='bold')

# Footnote Summary
footer = ("The 41 papers are split into normalized text windows and queried using 249 ontology-derived questions.\n"
          "The LLM (Qwen 7B) performs dual-pass extraction of subject-relation-object triples, verified against\n"
          "the source text before populating the tiered knowledge graph.")
ax.text(8.5, 0.05, footer, ha='center', fontsize=10.5, fontweight='bold', linespacing=1.3)

plt.savefig('ontogeorag_pipeline_v3.png', dpi=300, bbox_inches='tight', facecolor=BG)
print("Updated pipeline diagram saved successfully.")