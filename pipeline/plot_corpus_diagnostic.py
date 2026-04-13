"""
plot_corpus_diagnostic.py — bold, fully opaque, publication-ready.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Load and correct data ─────────────────────────────────────────────
data = json.load(open('output/diagnostics/corpus_diagnostic_corrected.json'))

# Manual outcome corrections from passage inspection
corrections = {
    ('debris flow',          'hasDescriptor', 'chaotic'):            'RECOVERED',
    ('debris flow',          'hasDescriptor', 'hummocky'):           'PIPELINE_FAILURE_RETRIEVAL',
    ('slide',                'hasDescriptor', 'blocky'):             'PIPELINE_FAILURE_RETRIEVAL',
    ('turbidite',            'hasDescriptor', 'layered'):            'RECOVERED_VIA_NORMALIZATION',
    ('mass transport deposit','hasDescriptor', 'discontinuous'):     'RECOVERED',
    ('hemipelagite',         'hasDescriptor', 'parallel'):           'PIPELINE_FAILURE_RETRIEVAL',
    ('pore pressure',        'controls',      'slope failure'):      'PIPELINE_FAILURE_RETRIEVAL',
    ('turbidity current',    'formedBy',      'debris flow'):        'PIPELINE_FAILURE_EXTRACTION',
    ('mass transport deposit','occursIn',     'abyssal plain'):      'PIPELINE_FAILURE_RETRIEVAL',
    ('debris flow',          'occursIn',      'continental slope'):  'PIPELINE_FAILURE_EXTRACTION',
    ('slide',                'overlies',      'hemipelagite'):       'PIPELINE_FAILURE_EXTRACTION',
}

for e in data:
    key = (e['subject'], e['relation'], e['object'])
    if key in corrections:
        e['outcome'] = corrections[key]

# ── Color scheme — all fully opaque ──────────────────────────────────
COLORS = {
    'RECOVERED':                   '#1B5E20',   # dark green
    'RECOVERED_VIA_NORMALIZATION': '#66BB6A',   # medium green
    'PIPELINE_FAILURE_RETRIEVAL':  '#E65100',   # dark orange
    'PIPELINE_FAILURE_EXTRACTION': '#FF8F00',   # amber
    'CORPUS_GAP':                  '#B71C1C',   # dark red
}

LABELS = {
    'RECOVERED':                   'Recovered',
    'RECOVERED_VIA_NORMALIZATION': 'Recovered via normalization',
    'PIPELINE_FAILURE_RETRIEVAL':  'Pipeline failure — retrieval',
    'PIPELINE_FAILURE_EXTRACTION': 'Pipeline failure — extraction',
    'CORPUS_GAP':                  'Corpus gap',
}

# ── Sort: corpus gaps, then failures, then recovered ─────────────────
ORDER = {
    'CORPUS_GAP':                  0,
    'PIPELINE_FAILURE_RETRIEVAL':  1,
    'PIPELINE_FAILURE_EXTRACTION': 2,
    'RECOVERED_VIA_NORMALIZATION': 3,
    'RECOVERED':                   4,
}

data_sorted = sorted(
    data,
    key=lambda x: (ORDER.get(x['outcome'], 3), -x['corpus_chunks'])
)

# ── Relation type abbreviations ───────────────────────────────────────
REL_SHORT = {
    'hasDescriptor': 'hD',
    'occursIn':      'oI',
    'causes':        'ca',
    'triggers':      'tr',
    'controls':      'co',
    'formedBy':      'fB',
    'overlies':      'ov',
    'partOf':        'pO',
}

def make_label(e):
    subj = e['subject']
    obj  = e['object']
    rel  = REL_SHORT.get(e['relation'], e['relation'][:2])
    # Truncate if too long
    if len(subj) > 24: subj = subj[:22] + '…'
    if len(obj)  > 18: obj  = obj[:16]  + '…'
    return f'{subj}  [{rel}]  {obj}'

labels  = [make_label(e)             for e in data_sorted]
chunks  = [e['corpus_chunks']        for e in data_sorted]
outcomes= [e['outcome']              for e in data_sorted]
c9_rec  = [e.get('c9',  e.get('recovered_c9',  False)) for e in data_sorted]
c10_rec = [e.get('c10', e.get('recovered_c10', False)) for e in data_sorted]

MAX_CHUNKS = max(chunks) if max(chunks) > 0 else 1

# ── Figure ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 10))
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

y = np.arange(len(data_sorted))
BAR_H = 0.62

for i, (e, outcome, chunk, c9, c10) in enumerate(
        zip(data_sorted, outcomes, chunks, c9_rec, c10_rec)):

    bar_color = COLORS.get(outcome, '#999999')

    # ── Chunk count bar (light grey background) ───────────────────
    ax.barh(i, chunk, BAR_H,
            color='#E0E0E0', edgecolor='#AAAAAA',
            linewidth=0.8, zorder=1)

    # ── Colored outcome stripe on the left ────────────────────────
    ax.barh(i, MAX_CHUNKS * 0.018, BAR_H,
            left=0, color=bar_color,
            edgecolor='#000000', linewidth=0.6,
            zorder=3)

    # ── C9 / C10 tags ─────────────────────────────────────────────
    tag_x = chunk + MAX_CHUNKS * 0.025
    for tag, flag in [('C9', c9), ('C10', c10)]:
        if flag:
            ax.text(tag_x, i, tag,
                    va='center', ha='left',
                    fontsize=8.5, fontweight='bold',
                    color='#FFFFFF',
                    bbox=dict(boxstyle='round,pad=0.22',
                              fc='#1565C0',
                              ec='#000000',
                              linewidth=0.8,
                              alpha=1.0),
                    zorder=6)
            tag_x += MAX_CHUNKS * 0.085

    # ── Chunk count number inside bar ─────────────────────────────
    if chunk > 0:
        ax.text(chunk - MAX_CHUNKS * 0.01, i,
                str(chunk),
                va='center', ha='right',
                fontsize=8, fontweight='bold',
                color='#333333', zorder=4)

# ── Y-axis labels ─────────────────────────────────────────────────────
ax.set_yticks(y)
ax.set_yticklabels(labels,
                   fontsize=9.5,
                   fontweight='bold',
                   color='#000000')

# ── Divider lines between groups ─────────────────────────────────────
gap_end  = sum(1 for e in data_sorted
               if e['outcome'] == 'CORPUS_GAP')
ret_end  = gap_end + sum(1 for e in data_sorted
               if e['outcome'] == 'PIPELINE_FAILURE_RETRIEVAL')
ext_end  = ret_end + sum(1 for e in data_sorted
               if e['outcome'] == 'PIPELINE_FAILURE_EXTRACTION')

for boundary in [gap_end - 0.5, ret_end - 0.5, ext_end - 0.5]:
    ax.axhline(boundary, color='#000000',
               lw=1.5, ls='--', zorder=2)

# ── Section labels on the right ───────────────────────────────────────
section_style = dict(
    ha='left', va='center',
    fontsize=10, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.3',
              fc='#FFFFFF',
              ec='#000000',
              linewidth=1.2,
              alpha=1.0)
)

right_x = MAX_CHUNKS * 1.38

sections = [
    (gap_end / 2 - 0.5,
     'CORPUS\nGAPS\n(n=2)',
     COLORS['CORPUS_GAP']),

    ((gap_end + ret_end) / 2 - 0.5,
     'RETRIEVAL\nFAILURES\n(n=5)',
     COLORS['PIPELINE_FAILURE_RETRIEVAL']),

    ((ret_end + ext_end) / 2 - 0.5,
     'EXTRACTION\nFAILURES\n(n=4)',
     COLORS['PIPELINE_FAILURE_EXTRACTION']),

    ((ext_end + len(data_sorted)) / 2 - 0.5,
     'RECOVERED\n(n=18)',
     COLORS['RECOVERED']),
]

for ypos, txt, col in sections:
    ax.text(right_x, ypos, txt,
            color=col, **section_style)

# ── X axis ────────────────────────────────────────────────────────────
ax.set_xlabel(
    'Number of corpus chunks containing both subject and object entities',
    fontsize=11, fontweight='bold', color='#000000', labelpad=8)

ax.tick_params(axis='x', labelsize=10, width=1.2)
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
    label.set_color('#000000')

ax.set_xlim(-MAX_CHUNKS * 0.02, MAX_CHUNKS * 1.62)
ax.invert_yaxis()

# ── Spines ────────────────────────────────────────────────────────────
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['bottom'].set_color('#000000')

# ── Legend ────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(facecolor=COLORS[k],
                   edgecolor='#000000',
                   linewidth=1.2,
                   label=LABELS[k])
    for k in [
        'CORPUS_GAP',
        'PIPELINE_FAILURE_RETRIEVAL',
        'PIPELINE_FAILURE_EXTRACTION',
        'RECOVERED_VIA_NORMALIZATION',
        'RECOVERED',
    ]
]
# C9/C10 tag example
import matplotlib.lines as mlines
c9_patch = mlines.Line2D([], [],
    color='#1565C0', marker='s',
    markersize=10, linewidth=0,
    label='Retrieved by C9 or C10')

legend = ax.legend(
    handles=patches + [c9_patch],
    loc='lower right',
    fontsize=9.5,
    framealpha=1.0,
    edgecolor='#000000',
    title='Outcome',
    title_fontsize=10,
)
legend.get_title().set_fontweight('bold')
for text in legend.get_texts():
    text.set_fontweight('bold')
    text.set_color('#000000')

# ── Title ─────────────────────────────────────────────────────────────
ax.set_title(
    'Corpus presence and retrieval coverage — 26 LB2019 benchmark edges\n'
    'Failure mode classified by manual inspection of retrieved passages',
    fontsize=12, fontweight='bold',
    pad=14, color='#000000')

plt.tight_layout()
plt.savefig('figures/fig_corpus_diagnostic.pdf',
            dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
plt.savefig('figures/fig_corpus_diagnostic.png',
            dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
print('Saved figures/fig_corpus_diagnostic.pdf and .png')
