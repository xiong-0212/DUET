from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
T = ['TE18', 'TL12', 'JL10', 'RB9']


ARMS = [
    ('mlp',           'MLP only (CoBRA-M2 equiv.)',   'base'),
    ('seq_ss',        'seq+ss (current best)',        'base'),
    ('seq_ss_alt',    'seq+ss+ss_alt binary P≥0.3',   'base'),
    ('ssalt_p10',     'ss_alt binary P≥0.1',          'thr'),
    ('ssalt_p20',     'ss_alt binary P≥0.2',          'thr'),
    ('ssalt_p50',     'ss_alt binary P≥0.5',          'thr'),
    ('ssalt_p10w',    'ss_alt weighted P≥0.1',        'w'),
    ('ssalt_p30w',    'ss_alt weighted P≥0.3',        'w'),
    ('ssalt_p01w',    'ss_alt weighted P≥0.01',       'w'),
    ('ss_pw',         'ss weighted (MFE, w=P)',       'w'),
    ('bp10w',         'seq+bp weighted P≥0.1',        'w'),
    ('bplow_w',       'seq+bp weighted P≥0.01',       'w'),
    ('ssalt_p10rand', 'ss_alt random edges (ctrl)',   'ctrl'),
    ('ssalt_p10shuf', 'ss_alt shuffled w (ctrl)',     'ctrl'),
    ('ss_pwshuf',     'ss shuffled w (ctrl)',         'ctrl'),
    ('ss_w1',         'ss all-ones w (ctrl)',         'ctrl'),
    ('bp10',          'seq+bp binary P≥0.1 (ctrl)',   'ctrl'),
    ('bplow_bin',     'seq+bp binary P≥0.01 (ctrl)',  'ctrl'),
    ('bplow_shuf',    'seq+bp shuffled w (ctrl)',     'ctrl'),
    ('pf_ss',         'prob node-features (5-d)',     'w'),
    ('pf_ss_shuf',    'shuffled node-features (ctrl)', 'ctrl'),
    ('ss_prune90',    'MFE pruned P≥0.9',             'w'),
    ('ss_prune70',    'MFE pruned P≥0.7',             'w'),
]
COLOR = {'base': '#4c72b0', 'thr': '#dd8452', 'w': '#55a868', 'ctrl': '#9e9e9e'}
HATCH = {'ctrl': '//'}


def collect(tag):
    vals, per = [], {s: [] for s in T}
    for f in glob.glob(os.path.join(ROOT, 'results', 'controlled',
                                    f'train_clean_{tag}_seed*.json')):
        r = json.load(open(f, encoding='utf-8'))
        tm = r.get('test_metrics', {})
        if not all(s in tm for s in T):
            return None
        vals.append(float(np.mean([tm[s]['at_threshold']['MCC'] for s in T])))
        for s in T:
            per[s].append(tm[s]['at_threshold']['MCC'])
    if not vals:
        return None
    return {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)),
            'n': len(vals), 'per': {s: float(np.mean(v)) for s, v in per.items()}}


def main():
    rows = [(tag, label, cat, collect(tag)) for tag, label, cat in ARMS]
    rows = [r for r in rows if r[3] is not None]
    rows.sort(key=lambda r: r[3]['mean'])
    ref = [r for r in rows if r[0] == 'seq_ss']
    ref_val = ref[0][3]['mean'] if ref else 0.343

    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(rows) + 1.5), dpi=300)
    y = np.arange(len(rows))
    for i, (tag, label, cat, d) in enumerate(rows):
        ax.barh(i, d['mean'], xerr=d['std'], height=0.66, capsize=2.5,
                color=COLOR[cat], edgecolor='white', linewidth=0.8,
                hatch=HATCH.get(cat, ''))
        ax.text(d['mean'] + d['std'] + 0.006, i, f"{d['mean']:.3f}",
                va='center', fontsize=7.5)
    ax.axvline(ref_val, color='#c44e52', ls='--', lw=1)
    ax.text(ref_val + 0.002, len(rows) - 0.3, f'seq+ss = {ref_val:.3f}',
            color='#c44e52', fontsize=8, va='top')
    ax.set_yticks(y)
    ax.set_yticklabels([f'{label}' for _, label, _, _ in rows], fontsize=8.2)
    ax.set_xlabel('Average MCC over TE18/TL12/JL10/RB9 (5 seeds, mean ± s.d.)',
                  fontsize=9)
    ax.set_xlim(0, max(d['mean'] + d['std'] for *_, d in rows) + 0.055)
    ax.spines[['top', 'right']].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[c],
                             hatch=HATCH.get(c, ''), edgecolor='white')
               for c in ['base', 'thr', 'w', 'ctrl']]
    ax.legend(handles, ['reference arms', 'threshold scan', 'probability-weighted',
                        'controls'], frameon=False, fontsize=8, loc='lower right')
    ax.set_title('Probability-weighted structural edges: controlled comparison '
                 '(de-leaked train=314)', fontsize=10)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(ROOT, 'figures', f'Fig7_weighted_grid.{ext}'))
    print(f'Saved figures/Fig7_weighted_grid.png/.pdf ({len(rows)} arms)')

    for tag, label, cat, d in reversed(rows):
        print(f"  {d['mean']:.3f} ± {d['std']:.3f}  {label}")


if __name__ == '__main__':
    main()
