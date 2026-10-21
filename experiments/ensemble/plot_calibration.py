from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
T = ['TE18', 'TL12', 'JL10', 'RB9']
C = {'uncal': '#9e9e9e', 'cal': '#4c72b0', 'ens': '#55a868'}


def main():
    d = json.load(open(os.path.join(ROOT, 'results', 'calibration.json'),
                       encoding='utf-8'))
    r, rel, ch = d['result'], d['reliability'], d['chains']

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.2), dpi=300)


    ax = axes[0, 0]
    for key, lab, col in [('single_uncal', 'Single model (uncalibrated)', C['uncal']),
                          ('ensemble', '5-seed ensemble', C['ens'])]:
        bins = [b for b in rel[key]['bins'] if b['n'] > 0]
        x = np.array([b['conf'] for b in bins])
        y = np.array([b['acc'] for b in bins])
        n = np.array([b['n'] for b in bins], dtype=float)
        ax.plot(x, y, '-', lw=1.2, color=col,
                label=f'{lab} (ECE={rel[key]["ece"]:.3f})')
        ax.scatter(x, y, s=np.clip(n / n.max() * 45, 8, 45), color=col, zorder=3)
    ax.plot([0, 1], [0, 1], '--', color='#c44e52', lw=1, label='perfect calibration')
    ax.set_xlabel('Predicted probability (binned)')
    ax.set_ylabel('Empirical positive rate')
    ax.set_title('(a) Reliability, pooled test sets', fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)


    ax = axes[0, 1]
    x = np.arange(len(T))
    e0 = [float(np.mean(r['per_seed']['uncal'][s]['ECE'])) for s in T]
    e1 = [float(np.mean(r['per_seed']['cal'][s]['ECE'])) for s in T]
    ax.bar(x - 0.19, e0, 0.38, label='uncalibrated', color=C['uncal'], edgecolor='white')
    ax.bar(x + 0.19, e1, 0.38, label='temperature-scaled', color=C['cal'], edgecolor='white')
    for xi, v in zip(x - 0.19, e0):
        ax.text(xi, v + 0.004, f'{v:.3f}', ha='center', fontsize=7)
    for xi, v in zip(x + 0.19, e1):
        ax.text(xi, v + 0.004, f'{v:.3f}', ha='center', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(T)
    ax.set_ylabel('ECE (15 bins)')
    ax.set_title('(b) Calibration by test set (T fit on val)', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)


    ax = axes[1, 0]
    rows = r['selective']['pooled']
    cov = [x['coverage'] for x in rows]
    ax.plot(cov, [x['MCC'] for x in rows], 'o-', color='#4c72b0', label='MCC')
    ax.plot(cov, [x['Precision'] for x in rows], 's-', color='#dd8452', label='Precision')
    ax.plot(cov, [x['Recall'] for x in rows], '^-', color='#55a868', label='Recall')
    for xi, row in zip(cov, rows):
        ax.annotate(f"{row['MCC']:.2f}", (xi, row['MCC']), textcoords='offset points',
                    xytext=(0, 6), ha='center', fontsize=7)
    ax.invert_xaxis()
    ax.set_xlabel('Retained fraction (highest confidence first)')
    ax.set_ylabel('Metric')
    ax.set_title('(c) Selective prediction, pooled residues', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)


    ax = axes[1, 1]
    markers = {'TE18': 'o', 'TL12': 's', 'JL10': '^', 'RB9': 'D'}
    for split in T:
        pts = [(v['mean_conf'], v['mcc']) for v in ch.values()
               if v['split'] == split and np.isfinite(v['mcc'])]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=22,
                   marker=markers[split], alpha=0.75, label=split)
    ok = [(v['mean_conf'], v['mcc']) for v in ch.values() if np.isfinite(v['mcc'])]
    xs, ys = np.array([p[0] for p in ok]), np.array([p[1] for p in ok])
    rho_c, p_c = stats.spearmanr(xs, ys)
    k, b = np.polyfit(xs, ys, 1)
    xr = np.linspace(xs.min(), xs.max(), 20)
    ax.plot(xr, k * xr + b, '--', color='#c44e52', lw=1.2,
            label=f'ρ={rho_c:.2f} (p={p_c:.3f})')
    ax.set_xlabel('Mean predictive confidence per chain')
    ax.set_ylabel('Chain MCC (ensemble)')
    ax.set_title('(d) Confidence predicts per-chain accuracy', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(ROOT, 'figures', f'Fig8_uncertainty.{ext}'))
    print('saved figures/Fig8_uncertainty.png/.pdf')
    print('Selective prediction (pooled):')
    for row in rows:
        print(f"  retain {row['coverage']*100:.0f}%: MCC={row['MCC']:.3f} "
              f"Prec={row['Precision']:.3f} Rec={row['Recall']:.3f}")


if __name__ == '__main__':
    main()
