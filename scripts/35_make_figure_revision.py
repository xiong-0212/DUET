from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from audit_panel_alignment import require_matplotlib_panel_alignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)

R = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
BLUE, ORANGE, GREEN, GREY = '#0072B2', '#D55E00', '#009E73', '#4D4D4D'
plt.rcParams.update({
    'font.size': 7,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'axes.titlesize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})


def load(name):
    return json.load(open(os.path.join(R, name), encoding='utf-8'))


def panel_label(ax, letter, title):
    ax.text(-0.16, 1.02, letter, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=8, fontweight='bold')
    ax.set_title(title, loc='left', pad=7)


def save(fig, name):
    require_matplotlib_panel_alignment(
        fig,
        json_out=os.path.join(FIG, f'{name}.alignment.json'),
        overlay_svg=os.path.join(FIG, f'{name}.alignment.svg'),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(os.path.join(FIG, f'{name}.pdf'))
    fig.savefig(os.path.join(FIG, f'{name}.svg'))
    fig.savefig(os.path.join(FIG, f'{name}.png'), dpi=600)
    fig.savefig(os.path.join(FIG, f'{name}.tiff'), dpi=600)
    plt.close(fig)


def main():

    psm = load('per_set_metrics.json')['per_set']
    duet = next(k for k in psm if k.startswith('DUET'))
    base = next(k for k in psm if not k.startswith('DUET'))
    T4 = ['TE18', 'TL12', 'JL10', 'RB9']
    T3 = ['TE18', 'TL12', 'JL10']
    id_duet4 = np.mean([psm[duet]['MCC'][s]['mean'] for s in T4])
    id_base4 = np.mean([psm[base]['MCC'][s]['mean'] for s in T4])
    id_duet3 = np.mean([psm[duet]['MCC'][s]['mean'] for s in T3])
    id_base3 = np.mean([psm[base]['MCC'][s]['mean'] for s in T3])
    gc = load('grassp_vs_duet.json')['average']
    id_gc = gc['GRASSP (sequence + 3D structure)']['MCC']['mean']

    st = load('strict_summary.json')['thresholds']
    labels = ['identical\nsequences\nonly', '≥ 80 %\nidentity', '≥ 70 %\nidentity',
              '≥ 50 %\nidentity']
    keys = ['p80', 'p70', 'p50']
    duet4 = [id_duet4] + [st[k]['duet_mean'] for k in keys]
    base4 = [id_base4] + [st[k]['baseline_mean'] for k in keys]
    duet3 = [id_duet3] + [st[k]['grassp']['duet_3set'] for k in keys]
    gc3 = [id_gc] + [st[k]['grassp']['mean'] for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.24, top=0.88, wspace=0.50)

    x = np.arange(4)
    ax = axes[0]
    ax.plot(x, duet4, 'o-', color=GREEN, lw=1.2, ms=3.5, label='DUET')
    ax.plot(x, base4, 's--', color=GREY, lw=1.0, ms=3.0, label='CoBRA-M3')
    for i, (a, b) in enumerate(zip(duet4, base4)):
        label_y = max(a, b) + 0.035
        if i == 3:
            label_y += 0.025
        ax.text(i + 0.06, label_y, f'{a - b:+.3f}', ha='left', va='bottom',
                fontsize=5.8, color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel('MCC (four benchmarks)')
    ax.set_ylim(0, 0.50)
    ax.set_xlim(-0.20, 3.80)
    ax.legend(frameon=False, fontsize=6.5, loc='lower left')
    panel_label(ax, 'a', 'Ensemble advantage')

    ax = axes[1]
    ax.plot(x[:3], duet3[:3], 'o-', color=GREEN, lw=1.2, ms=3.5, label='DUET')
    ax.plot(x[:3], gc3[:3], '^-', color=BLUE, lw=1.2, ms=3.5, label='GRASSP')
    for i, (a, b) in enumerate(zip(duet3[:3], gc3[:3])):
        ax.text(i + 0.06, max(a, b) + 0.035, f'{a - b:+.3f}', ha='left', va='bottom',
                fontsize=5.8, color=GREEN if a > b else BLUE)
    ax.set_xticks(x[:3])
    ax.set_xticklabels(labels[:3], fontsize=6)
    ax.set_ylabel('MCC (three shared benchmarks)')
    ax.set_ylim(0, 0.50)
    ax.set_xlim(-0.20, 2.75)
    ax.legend(frameon=False, fontsize=6.5, loc='lower left')
    panel_label(ax, 'b', 'Sequence vs structure')

    ax = axes[2]
    ev = load('external_eval.json')['per_set']
    eb = load('external_eval_baseline.json')['per_set']
    order = [('time_dependent', 'time\ndependent'), ('external_jiang', 'Jiang'),
             ('hariboss_test', 'HARIBOSS\ntest')]
    order = [(k, lab) for k, lab in order if k in ev]
    xs = np.arange(len(order))
    ax.bar(xs - 0.19, [ev[k]['MCC']['mean'] for k, _ in order], width=0.36, color=GREEN,
           label='DUET')
    ax.bar(xs + 0.19, [eb[k]['MCC']['mean'] for k, _ in order], width=0.36, color=GREY,
           label='Baseline')
    for i, (k, _) in enumerate(order):
        ax.text(i, max(ev[k]['MCC']['mean'], eb[k]['MCC']['mean']) + 0.015,
                f"{ev[k]['n_chains']}", ha='center', fontsize=5.8, color=GREY)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in order], fontsize=6)
    ax.set_ylabel('MCC (external)')
    ax.set_ylim(0, 0.50)
    ax.legend(frameon=False, fontsize=6.5, loc='upper right')
    panel_label(ax, 'c', 'External validation')

    os.makedirs(FIG, exist_ok=True)
    save(fig, 'Fig4_deleaking_criterion')
    print('[written] figures/Fig4_deleaking_criterion.pdf / .svg / .png / .tiff')
    print('  panel a:', [round(v, 3) for v in duet4], [round(v, 3) for v in base4])
    print('  panel b:', [round(v, 3) for v in duet3], [round(v, 3) for v in gc3])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
