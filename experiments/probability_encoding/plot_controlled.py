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
MODELS = ['mlp', 'seq_ss', 'seq_ss_alt', 'seq_ss_sim', 'seq_ss_alt_sim']
LABELS = ['MLP only\n(CoBRA-M2 equiv.)', 'seq+ss', 'seq+ss+ss_alt\n(PRISM main)', 'seq+ss+sim', 'seq+ss+ss_alt+sim']


def collect(ds):
    means, stds = [], []
    for m in MODELS:
        vals = []
        for f in glob.glob(os.path.join(ROOT, 'results', 'controlled', f'train_{ds}_{m}_seed*.json')):
            r = json.load(open(f, encoding="utf-8"))
            tm = r['test_metrics']
            vals.append(np.mean([tm[s]['at_threshold']['MCC'] for s in T]))
        means.append(np.mean(vals)); stds.append(np.std(vals, ddof=1))
    return np.array(means), np.array(stds)


def main():
    mo, so = collect('orig')
    mc, sc = collect('clean')
    x = np.arange(len(MODELS)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=300)
    b1 = ax.bar(x - w/2, mo, w, yerr=so, capsize=3, label='Original benchmark (train=345)',
                color='#9ecae1', edgecolor='#3182bd', linewidth=1)
    b2 = ax.bar(x + w/2, mc, w, yerr=sc, capsize=3, label='De-leaked (train=314)',
                color='#fdae6b', edgecolor='#e6550d', linewidth=1)
    for bars, vals in [(b1, mo), (b2, mc)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.008, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Average MCC (4 benchmarks)')
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=8.5)
    ax.set_ylim(0, 0.52)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('Controlled comparison: identical loss / splits / seeds (5 seeds, mean ± s.d.)',
                 fontsize=10)

    ax.annotate('', xy=(1 + w/2, mc[1]), xytext=(2 + w/2, mc[2]),
                arrowprops=dict(arrowstyle='-', color='#888', ls='--', lw=0.8))
    ax.text(1.5 + w/2, min(mc[1], mc[2]) - 0.035,
            'ss_alt vs seq+ss:\n−0.027 (p=0.045)', ha='center', fontsize=7.5, color='#b30000')
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, 'figures'), exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(ROOT, 'figures', f'Fig6_controlled_grid.{ext}'))
    print('Saved figures/Fig6_controlled_grid.png/.pdf')


if __name__ == '__main__':
    main()
