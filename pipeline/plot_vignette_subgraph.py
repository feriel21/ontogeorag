"""
plot_vignette_subgraph.py
Vignette subgraph — bold labels, bold edges, bold everything.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

COLORS = {
    'SeismicObject': '#1565C0',
    'Process':       '#BF360C',
    'Descriptor':    '#1B5E20',
    'Setting':       '#4A148C',
}
BG = '#FFFFFF'

nodes = {
    'MTD':         ('SeismicObject', 'Mass transport\ndeposit'),
    'BSS':         ('SeismicObject', 'Basal shear\nsurface'),
    'SF':          ('Process',       'Slope\nfailure'),
    'FO':          ('Process',       'Fluid\noverpressure'),
    'GC':          ('Process',       'Gravitational\ncollapse'),
    'DFT':         ('Process',       'Debris flow\ntransformation'),
    'CHAOTIC':     ('Descriptor',    'chaotic'),
    'TRANSPARENT': ('Descriptor',    'transparent'),
    'LOWAMP':      ('Descriptor',    'low-amplitude'),
    'TOPO':        ('Setting',       'Rough\ntopography'),
}

pos = {
    'FO':          (0.08, 0.82),
    'GC':          (0.08, 0.57),
    'SF':          (0.08, 0.32),
    'MTD':         (0.45, 0.57),
    'BSS':         (0.45, 0.20),
    'CHAOTIC':     (0.84, 0.82),
    'TRANSPARENT': (0.84, 0.57),
    'LOWAMP':      (0.84, 0.32),
    'TOPO':        (0.25, 0.07),
    'DFT':         (0.65, 0.07),
}

edges = [
    ('FO',   'GC',          'causes',         (-0.09,  0.00)),
    ('GC',   'SF',          'causes',         (-0.09,  0.00)),
    ('SF',   'MTD',         'causes',         ( 0.00,  0.07)),
    ('MTD',  'BSS',         'partOf',         ( 0.07,  0.00)),
    ('MTD',  'CHAOTIC',     'hasDescriptor',  ( 0.00,  0.07)),
    ('MTD',  'TRANSPARENT', 'hasDescriptor',  ( 0.00,  0.07)),
    ('MTD',  'LOWAMP',      'hasDescriptor',  ( 0.00, -0.07)),
    ('TOPO', 'DFT',         'controls',       ( 0.00,  0.06)),
]

NODE_R = 0.075

fig, ax = plt.subplots(figsize=(15, 9))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(-0.05, 1.08)
ax.set_ylim(-0.04, 1.02)
ax.axis('off')

# ── Draw edges ────────────────────────────────────────────────────────
for src, tgt, label, offset in edges:
    x0, y0 = pos[src]
    x1, y1 = pos[tgt]
    dx, dy = x1 - x0, y1 - y0
    dist = np.sqrt(dx**2 + dy**2)
    ux, uy = dx / dist, dy / dist
    xs = x0 + ux * NODE_R
    ys = y0 + uy * NODE_R
    xe = x1 - ux * NODE_R
    ye = y1 - uy * NODE_R

    ax.annotate('',
        xy=(xe, ye), xytext=(xs, ys),
        arrowprops=dict(
            arrowstyle='->', color='#000000',
            lw=2.2,
            mutation_scale=22,
            connectionstyle='arc3,rad=0.06',
        ),
        zorder=2)

    mx = (xs + xe) / 2 + offset[0]
    my = (ys + ye) / 2 + offset[1]

    ax.text(mx, my, label,
            ha='center', va='center',
            fontsize=10,
            fontweight='bold',
            color='#000000',
            bbox=dict(
                boxstyle='round,pad=0.25',
                fc='#FFFFFF',
                ec='#000000',
                linewidth=1.5,
                alpha=1.0,
            ),
            zorder=5)

# ── Draw nodes ────────────────────────────────────────────────────────
for key, (ntype, label) in nodes.items():
    x, y = pos[key]
    color = COLORS[ntype]

    # Outer ring for contrast
    ring = plt.Circle((x, y), NODE_R + 0.005,
                       color='#000000', zorder=3)
    ax.add_patch(ring)

    circle = plt.Circle((x, y), NODE_R,
                         color=color, zorder=4)
    ax.add_patch(circle)

    ax.text(x, y, label,
            ha='center', va='center',
            fontsize=9.5,
            fontweight='bold',
            color='#FFFFFF',
            linespacing=1.4,
            zorder=5)

# ── Zone headers ──────────────────────────────────────────────────────
zone_style = dict(
    ha='center', va='center',
    fontsize=11, fontweight='bold',
    color='#000000',
    bbox=dict(boxstyle='round,pad=0.3',
              fc='#F5F5F5', ec='#AAAAAA',
              linewidth=1.2, alpha=1.0)
)

ax.text(0.08, 0.97, 'TRIGGER CHAIN', **zone_style)
ax.text(0.45, 0.97, 'IDENTIFIED OBJECT', **zone_style)
ax.text(0.84, 0.97, 'SEISMIC DESCRIPTORS', **zone_style)
ax.text(0.45, -0.02, 'DOWNSLOPE EVOLUTION', **zone_style)

# ── Dividers ──────────────────────────────────────────────────────────
for xv in [0.28, 0.65]:
    ax.axvline(xv, color='#AAAAAA', lw=1.2,
               ls='--', zorder=0)
ax.axhline(0.14, color='#AAAAAA', lw=1.2,
           ls='--', zorder=0)

# ── Legend ────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(facecolor=COLORS[t],
                   edgecolor='#000000',
                   linewidth=1.2,
                   label=lbl)
    for t, lbl in [
        ('SeismicObject', 'Seismic object'),
        ('Process',       'Process'),
        ('Descriptor',    'Seismic descriptor'),
        ('Setting',       'Setting / condition'),
    ]
]
legend = ax.legend(
    handles=patches,
    loc='lower right',
    fontsize=10,
    framealpha=1.0,
    edgecolor='#000000',
    title='Node type',
    title_fontsize=10,
)
legend.get_title().set_fontweight('bold')
for text in legend.get_texts():
    text.set_fontweight('bold')

# ── Title ─────────────────────────────────────────────────────────────
ax.set_title(
    'Tier-1 knowledge graph relations supporting a mass-transport deposit\n'
    'interpretation scenario — each edge verified against a source passage',
    fontsize=12, fontweight='bold',
    pad=14, color='#000000')

plt.tight_layout()
plt.savefig('figures/fig_vignette_subgraph.pdf',
            dpi=300, bbox_inches='tight', facecolor=BG)
plt.savefig('figures/fig_vignette_subgraph.png',
            dpi=300, bbox_inches='tight', facecolor=BG)
print('Saved figures/fig_vignette_subgraph.pdf and .png')
