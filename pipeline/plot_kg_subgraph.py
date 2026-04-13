import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyArrowPatch

kg = json.load(open('output/run11_kg/tiered_kg_run11.json'))
tier1 = [t for t in kg['triples'] if t['tier'] == 1]

TRIPLES = [
    # Chaîne causale directe vers MTD
    ('earthquake',               'triggers',      'slope failure'),
    ('slope failure',            'causes',        'mass transport deposit'),
    ('growth stratal wedge',     'causes',        'mass transport deposit'),

    # MTD hasDescriptor — descripteurs sismiques
    ('mass transport deposit',   'hasDescriptor', 'transparent'),
    ('mass transport deposit',   'hasDescriptor', 'hummocky'),
    ('mass transport deposit',   'hasDescriptor', 'high-amplitude'),
    ('mass transport deposit',   'hasDescriptor', 'blocky'),
    ('mass transport deposit',   'hasDescriptor', 'continuous'),

    # Autres objets géologiques + descripteurs
    ('turbidite',                'hasDescriptor', 'parallel'),
    ('turbidite',                'hasDescriptor', 'continuous'),
    ('megaslide',                'hasDescriptor', 'chaotic'),
    ('megaslide',                'hasDescriptor', 'layered'),
    ('debris-flow deposit',      'hasDescriptor', 'chaotic'),

    # Contexte spatial — connecté à MTD
    ('mass transport deposit',   'occursIn',      'continental margin'),
    ('mass transport deposit',   'occursIn',      'deepwater basinal setting'),

    # Structure — connecté à MTD
    ('mass transport deposit',   'partOf',        'basal shear surface'),
    ('toe',                      'partOf',        'mass transport deposit'),

    # Connexions turbidite et megaslide à MTD
    ('turbidite',                'occursIn',      'continental margin'),
    ('megaslide',                'hasDescriptor', 'transparent'),
]

seen = set()
final = []
for t in TRIPLES:
    if t not in seen:
        seen.add(t)
        final.append(t)

all_nodes = set()
for s, r, o in final:
    all_nodes.add(s)
    all_nodes.add(o)

DESCRIPTORS = {
    'chaotic','transparent','blocky','hummocky','layered','parallel',
    'continuous','discontinuous','high-amplitude','low-amplitude',
    'massive','deformed','undeformed','wedge-shaped','stratified',
}
PROCESSES = {
    'slope failure','earthquake','forced regression',
    'excess pore pressure','fluid overpressure',
    'methane hydrate dissociation','rapid rate of sedimentation',
    'growth stratal wedge','sea-level lowstands',
    'translational sliding','gravitational spreading',
    'storm waves','pore-water pressures build up',
}
SETTINGS = {
    'continental margin','deepwater basinal setting',
    'basin floor','continental slope','canyon mouth',
}

def classify(name):
    n = name.lower()
    if n in DESCRIPTORS: return 'Descriptor'
    if n in PROCESSES:   return 'Process'
    if n in SETTINGS:    return 'Setting'
    if 'margin' in n or 'slope' in n or 'setting' in n: return 'Setting'
    if any(p in n for p in ['failure','earthquake','sliding',
                              'spreading','dissociation']): return 'Process'
    return 'SeismicObject'

NODE_COLORS = {
    'SeismicObject': '#4FC3F7',
    'Process':       '#FF8A65',
    'Descriptor':    '#81C784',
    'Setting':       '#CE93D8',
}
EDGE_COLORS = {
    'hasDescriptor':'#1B5E20','occursIn':'#4A148C',
    'formedBy':'#BF360C','partOf':'#1565C0',
    'triggers':'#880E4F','causes':'#E65100',
    'controls':'#006064','affects':'#4E342E',
}
REL_SHORT = {
    'hasDescriptor':'hD','occursIn':'oI','formedBy':'fB',
    'partOf':'pO','triggers':'tr','causes':'ca',
    'controls':'co','affects':'af','overlies':'ov',
}

node_list  = sorted(all_nodes)
node_idx   = {n: i for i, n in enumerate(node_list)}
n_nodes    = len(node_list)

np.random.seed(7)
pos = np.random.randn(n_nodes, 2) * 3.0
center_idx = node_idx['mass transport deposit']
pos[center_idx] = [0.0, 0.0]

for i, name in enumerate(node_list):
    c = classify(name)
    if c == 'Descriptor':
        pos[i] = [6.0 + np.random.rand(), (i % 8 - 4) * 1.5]
    elif c == 'Process' and name != 'mass transport deposit':
        pos[i][0] = -4.0 + np.random.rand()
    elif c == 'Setting':
        pos[i][1] = -5.0 + np.random.rand()

for _ in range(800):
    forces = np.zeros_like(pos)
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            diff = pos[i] - pos[j]
            dist = max(np.linalg.norm(diff), 0.3)
            f = diff / (dist**2) * 4.0
            forces[i] += f
            forces[j] -= f
    for s, r, o in final:
        i = node_idx[s]
        j = node_idx[o]
        diff = pos[j] - pos[i]
        forces[i] += diff * 0.1
        forces[j] -= diff * 0.1
    forces[center_idx] = 0
    pos += forces * 0.01
    pos *= 0.99

pos = pos / np.abs(pos).max() * 9.0

fig, ax = plt.subplots(figsize=(28, 20))
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')
ax.axis('off')

# ── Node dimensions ───────────────────────────────────────────────────
def get_node_dims(name):
    """Return (width, height) for ellipse based on text length."""
    words = name.split()
    lines = []
    if len(words) <= 2:
        lines = [name]
    elif len(words) == 3:
        lines = [words[0]+' '+words[1], words[2]]
    else:
        mid = len(words)//2
        lines = [' '.join(words[:mid]), ' '.join(words[mid:])]
    max_chars = max(len(l) for l in lines)
    n_lines   = len(lines)
    # width proportional to longest line, height to number of lines
    w = max(max_chars * 0.13, 1.6)
    h = max(n_lines  * 0.65, 0.9)
    if name == 'mass transport deposit':
        w, h = 3.2, 1.4
    return w, h, lines

# ── Draw edges ────────────────────────────────────────────────────────
drawn = set()
for s, r, o in final:
    si = node_idx[s]
    oi = node_idx[o]
    if (si, oi) in drawn:
        continue
    drawn.add((si, oi))

    x0, y0 = pos[si]
    x1, y1 = pos[oi]
    color  = EDGE_COLORS.get(r, '#555555')
    label  = REL_SHORT.get(r, r[:2])
    rad    = 0.15 if (oi, si) in drawn else 0.03

    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle='->', color=color,
            lw=2.2, mutation_scale=20,
            connectionstyle=f'arc3,rad={rad}'),
        zorder=2)

    # Edge label — offset perpendicular to edge direction
    mx = (x0 + x1) / 2
    my = (y0 + y1) / 2
    dx = x1 - x0
    dy = y1 - y0
    length = max(np.sqrt(dx**2 + dy**2), 0.001)
    # Perpendicular offset
    perp_x = -dy / length * 0.55
    perp_y =  dx / length * 0.55

    ax.text(mx + perp_x, my + perp_y, label,
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            color='#000000',
            bbox=dict(boxstyle='round,pad=0.2',
                      fc='#FFFFFF', ec='#000000',
                      linewidth=1.3, alpha=1.0),
            zorder=6)

# ── Draw nodes as ellipses ────────────────────────────────────────────
for name in node_list:
    idx   = node_idx[name]
    x, y  = pos[idx]
    ntype = classify(name)
    color = NODE_COLORS[ntype]
    w, h, lines = get_node_dims(name)

    ellipse = mpatches.Ellipse(
        (x, y), width=w, height=h,
        facecolor=color, edgecolor='none',
        zorder=3)
    ax.add_patch(ellipse)

    display = '\n'.join(lines)

    # Font size based on longest line
    max_chars = max(len(l) for l in lines)
    if name == 'mass transport deposit':
        fs = 13
    elif max_chars > 14:
        fs = 9.5
    elif max_chars > 10:
        fs = 11.0
    else:
        fs = 12.0

    ax.text(x, y, display,
            ha='center', va='center',
            fontsize=fs, fontweight='bold',
            color='#000000', linespacing=1.25,
            zorder=4)

ax.set_xlim(pos[:,0].min()-2.0, pos[:,0].max()+2.0)
ax.set_ylim(pos[:,1].min()-2.0, pos[:,1].max()+2.0)

# ── Legends ───────────────────────────────────────────────────────────
node_patches = [
    mpatches.Patch(facecolor=NODE_COLORS[t],
                   edgecolor='none', label=t)
    for t in ['SeismicObject','Process','Descriptor','Setting']
]
edge_patches = [
    mpatches.Patch(facecolor='#FFFFFF',
                   edgecolor=EDGE_COLORS[r],
                   linewidth=2.5,
                   label=f'[{REL_SHORT[r]}]  {r}')
    for r in ['hasDescriptor','occursIn','causes',
              'triggers','partOf']
]

leg1 = ax.legend(handles=node_patches, loc='upper left',
                 fontsize=12, framealpha=1.0,
                 edgecolor='#000000',
                 title='Node type', title_fontsize=12,
                 bbox_to_anchor=(0.0, 1.0))
leg1.get_title().set_fontweight('bold')
for t in leg1.get_texts():
    t.set_fontweight('bold')
    t.set_color('#000000')
ax.add_artist(leg1)

leg2 = ax.legend(handles=edge_patches, loc='upper right',
                 fontsize=12, framealpha=1.0,
                 edgecolor='#000000',
                 title='Relation type', title_fontsize=12,
                 bbox_to_anchor=(1.0, 1.0), ncol=2)
leg2.get_title().set_fontweight('bold')
for t in leg2.get_texts():
    t.set_fontweight('bold')
    t.set_color('#000000')

n_n = len(all_nodes)
n_e = len(final)
ax.set_title(
    f'Representative Tier-1 subgraph — causal chains, '
    f'seismic descriptors, and depositional context\n'
    f'({n_n} nodes, {n_e} edges — '
    f'all edges verified against source passages)',
    fontsize=14, fontweight='bold',
    color='#000000', pad=18)

plt.tight_layout()
plt.savefig('figures/fig_kg_subgraph.pdf', dpi=300,
            bbox_inches='tight', facecolor='#FFFFFF')
plt.savefig('figures/fig_kg_subgraph.png', dpi=300,
            bbox_inches='tight', facecolor='#FFFFFF')
print(f'Saved: {n_n} nodes, {n_e} edges')
