from __future__ import annotations

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


def main():
    d = json.load(open(os.path.join(ROOT, 'results', 'gate_analysis.json'),
                       encoding='utf-8'))
    cf = json.load(open(os.path.join(ROOT, 'results', 'gate_counterfactual.json'),
                        encoding='utf-8'))
    ex = d.get('examples', {})
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6), dpi=300)


    ax = axes[0, 0]
    gb, gn = [], []
    for rid, e in ex.items():
        g = np.array(e['g']); y = np.array(e['y'])
        gb += g[y == 1].tolist(); gn += g[y == 0].tolist()
    bins = np.linspace(0, 0.7, 40)
    ax.hist(gn, bins=bins, alpha=0.6, label=f'non-binding (μ={np.mean(gn):.3f})',
            color='#9e9e9e', density=True)
    ax.hist(gb, bins=bins, alpha=0.6, label=f'binding (μ={np.mean(gb):.3f})',
            color='#1D9E8F', density=True)
    ax.axvline(0.5, ls='--', lw=1, color='#c44e52')
    ax.text(0.5, ax.get_ylim()[1] * 0.9, ' equal weight', color='#c44e52', fontsize=7)
    ax.set_xlabel('Gate value $g$ (→1: graph tower, →0: MLP tower)')
    ax.set_ylabel('Density')
    ax.set_title('(a) Gate opens toward the graph tower at binding sites', fontsize=9.5,
                 fontweight='bold')
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)


    ax = axes[0, 1]
    picked = [k for k in ex if len(ex[k]['g']) >= 30][:3]
    for i, rid in enumerate(picked):
        g = np.array(ex[rid]['g']); y = np.array(ex[rid]['y'])
        off = i * 1.15
        ax.plot(np.arange(len(g)), g + off, lw=0.9, color='#4c72b0')
        ax.fill_between(np.arange(len(g)), off, off + 0.05, where=y == 1,
                        color='#1D9E8F', alpha=0.5, lw=0)
        ax.text(-2, off + 0.15, rid, ha='right', fontsize=6.5)
    ax.set_xlabel('Residue index')
    ax.set_ylabel('Gate value (+offset per chain)')
    ax.set_title('(b) Per-chain gate profiles (shaded = binding sites)', fontsize=9.5,
                 fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)


    ax = axes[1, 0]
    modes = [('learned', 'learned\n(natural)', '#1D9E8F'), ('const025', 'fixed 0.25\n(= mean)', '#9e9e9e'),
             ('const05', 'fixed 0.5\n(equal weight)', '#bbbbbb'), ('shuffled', 'shuffled\n(same dist.)', '#c44e52')]
    x = np.arange(len(T)); w = 0.2
    for i, (k, lab, col) in enumerate(modes):
        vals = [cf[k][s] for s in T]
        ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=col, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(T); ax.set_ylabel('MCC')
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_title('(c) Replacing the learned gate at inference (post-hoc)', fontsize=9.5,
                 fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)


    ax = axes[1, 1]
    ps = d['per_split']
    xs = np.arange(len(T)); w2 = 0.36
    b = [ps[s]['gate_binding'] for s in T]
    n = [ps[s]['gate_nonbinding'] for s in T]
    ax.bar(xs - w2/2, b, w2, label='binding', color='#1D9E8F', edgecolor='white')
    ax.bar(xs + w2/2, n, w2, label='non-binding', color='#9e9e9e', edgecolor='white')
    for xi, (bv, nv) in zip(xs, zip(b, n)):
        ax.text(xi, max(bv, nv) + 0.008, f'Δ={bv-nv:+.3f}', ha='center', fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels(T); ax.set_ylabel('Mean gate value')
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('(d) Gate gap is consistent across test sets', fontsize=9.5,
                 fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(ROOT, 'figures', f'Fig10_gate_analysis.{ext}'))
    print('saved figures/Fig10_gate_analysis.png/.pdf')


if __name__ == '__main__':
    main()
