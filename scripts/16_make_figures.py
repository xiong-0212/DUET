from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from audit_panel_alignment import require_matplotlib_panel_alignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)

FIG = os.path.join(ROOT, 'figures')

BLUE, ORANGE, GREEN, GREY = '#0072B2', '#D55E00', '#009E73', '#4D4D4D'
plt.rcParams.update({
    'font.size': 7,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'axes.titlesize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})


def save(fig, name, *, require_panel_labels=True, exemptions=()):
    os.makedirs(FIG, exist_ok=True)
    require_matplotlib_panel_alignment(
        fig,
        json_out=os.path.join(FIG, f'{name}.alignment.json'),
        overlay_svg=os.path.join(FIG, f'{name}.alignment.svg'),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=require_panel_labels,
        strict=True,
        exemptions=exemptions,
    )
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'), dpi=600)
    fig.savefig(os.path.join(FIG, f'{name}.tiff'), dpi=600)
    fig.savefig(os.path.join(FIG, f'{name}.svg'))
    plt.close(fig)
    print(f'[written] figures/{name}.pdf / .svg / .png')


def panel_label(ax, letter, title):
    ax.text(-0.16, 1.02, letter, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=8, fontweight='bold')
    ax.set_title(title, loc='left', pad=7)


def fig_leakage():
    audit = json.load(open(os.path.join(ROOT, 'results', 'homology_audit.json'), encoding='utf-8'))
    towers = json.load(open(os.path.join(ROOT, 'results', 'tower_analysis.json'), encoding='utf-8'))
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.75),
        gridspec_kw={'width_ratios': [1.0, 1.35, 1.0]},
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.30, top=0.88, wspace=0.58)


    ax = axes[0]
    order = ['exact_equal', '≥95%', '≥90%', '≥80%', '≥70%']
    labels = ['identical', '≥95%', '≥90%', '≥80%', '≥70%']
    hits = [audit['summary'][k]['hit'] for k in order]



    census = json.load(open(os.path.join(ROOT, 'results', 'overlap_census.json'),
                            encoding='utf-8'))
    total = census['total']['n_chains']
    bars = ax.bar(range(len(order)), hits, color=[ORANGE] + [BLUE] * (len(order) - 1), width=0.65)
    for i, (b, h) in enumerate(zip(bars, hits)):
        ax.text(b.get_x() + b.get_width() / 2, h + 0.7, str(h), ha='center', fontsize=7)
    ax.axhline(total, ls=':', lw=0.8, color=GREY)
    ax.text(len(order) - 0.5, total + 0.8, f'{total} test chains', ha='right', fontsize=6.2,
            color=GREY)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor',
                       fontstyle='italic', fontsize=6.0)
    ax.tick_params(axis='x', pad=1.5)
    ax.set_ylabel('Test chains with a\ntraining-set match', fontsize=6.8)
    ax.set_ylim(0, total * 1.25)
    panel_label(ax, 'a', 'Benchmark overlap')


    ax = axes[1]
    strata = [('<70%', '<70%'), ('70-79%', '70-79%'), ('90-94%', '90-94%'),
              ('95-99%', '95-99%'), ('100%', 'identical')]
    x = np.arange(len(strata))
    for off, key, col, lab in [(-0.19, 'seq+ss', BLUE, 'graph tower'),
                               (0.19, 'MLP', ORANGE, 'MLP on RNA-FM')]:
        vals = [audit['strata'][k][key + '_mean'] for k, _ in strata]
        ax.bar(x + off, vals, width=0.36, color=col, label=lab)
    ax.text(0.02, 0.95, 'Graph', color=BLUE, transform=ax.transAxes,
            ha='left', va='top', fontsize=6.0, fontweight='bold')
    ax.text(0.27, 0.95, 'MLP', color=ORANGE, transform=ax.transAxes,
            ha='left', va='top', fontsize=6.0, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in strata],
                       rotation=45, ha='right', rotation_mode='anchor',
                       fontstyle='italic', fontsize=5.8)
    ax.tick_params(axis='x', pad=1.5)
    ax.set_ylabel('Per-chain MCC', fontsize=6.8)
    ax.set_ylim(0, 0.85)
    ax.set_xlabel('Maximum identity to a training sequence', fontsize=6.5,
                  fontstyle='italic', labelpad=1)
    panel_label(ax, 'b', 'Performance by homology')


    ax = axes[2]
    order = [('MLP on RNA-FM', 'MLP on\nRNA-FM'), ('graph tower', 'graph\ntower'),
             ('CoBRA-M3 equivalent', 'CoBRA-M3')]
    x = np.arange(len(order))
    leaked = [towers['leakage'][k]['leaked'] for k, _ in order]
    clean = [towers['leakage'][k]['deleaked'] for k, _ in order]
    ax.bar(x - 0.19, leaked, width=0.36, color=GREY)
    ax.bar(x + 0.19, clean, width=0.36, color=GREEN)
    for i, (l, c) in enumerate(zip(leaked, clean)):
        ax.text(i, max(l, c) + 0.030, f'{c - l:+.3f}', ha='center', va='bottom',
                fontsize=6.2)
    ax.text(0.02, 0.95, 'Original', color=GREY, transform=ax.transAxes,
            ha='left', va='top', fontsize=6.0)
    ax.text(0.30, 0.95, 'De-leaked', color=GREEN, transform=ax.transAxes,
            ha='left', va='top', fontsize=6.0)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in order], fontsize=6.0)
    ax.set_ylabel('Average MCC\n(four benchmarks)', fontsize=6.8)
    ax.set_ylim(0, 0.60)
    panel_label(ax, 'c', 'Leaked vs de-leaked')

    save(
        fig,
        'Fig2_leakage',
        exemptions=[{
            'panels': ['a', 'b', 'c'],
            'checks': ['panel-width'],
            'reason': 'Panel b is widened to keep 45-degree homology labels legible.',
        }],
    )


def fig_main_results():
    v = json.load(open(os.path.join(ROOT, 'results', 'three_tower_tables.json'), encoding='utf-8'))
    t2 = v['table2']
    sets = ['TE18', 'TL12', 'JL10', 'RB9']
    methods = [('CoBRA-M3 equiv. (retrained baseline)', 'CoBRA-M3', GREY),
               ('DUET graph tower (R-GCN)', 'graph\ntower', BLUE),
               ('DUET sequence tower A (TCN)', 'TCN', '#56B4E9'),
               ('DUET sequence tower B (DenseNet)', 'DenseNet', ORANGE),
               ('DUET (three towers)', 'DUET', GREEN)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6),
                             gridspec_kw={'width_ratios': [3, 1.15]})

    ax = axes[0]
    x = np.arange(len(sets))
    w = 0.16
    for i, (key, lab, col) in enumerate(methods):
        means = [t2[key]['per_set_mean'][s] for s in sets]
        sds = [t2[key]['per_set_sd'][s] for s in sets]
        ax.bar(x + (i - 2) * w, means, width=w, yerr=sds, color=col, label=lab.replace('\n', ' '),
               error_kw=dict(lw=0.6, capsize=1.2))
    ax.set_xticks(x)
    ax.set_xticklabels(sets)
    ax.set_ylabel('MCC')
    ax.set_ylim(0, 0.72)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc='upper left')
    panel_label(ax, 'a', 'Performance per benchmark')

    ax = axes[1]
    names = ['TE18', 'TL12', 'JL10', 'RB9', 'average']
    deltas = [v['vs_baseline'][s]['delta'] for s in sets] + [v['vs_baseline']['average']['delta']]
    ps = [v['vs_baseline'][s]['p'] for s in sets] + [v['vs_baseline']['average']['p']]
    cols = [GREEN if p < 0.05 else GREY for p in ps]
    bars = ax.bar(range(5), deltas, color=cols, width=0.6)
    for i, (b, d, p) in enumerate(zip(bars, deltas, ps)):
        p_text = '< 0.001' if p < 0.001 else f'= {p:.3f}'
        label_x = b.get_x() + b.get_width() / 2
        align = 'center'
        if i == 0:
            label_x = 0.25
        label_y = d + (0.006 if d >= 0 else -0.012)
        if i == 0:
            label_y = 0.105
        if i == 1:
            label_y += 0.020
        ax.text(label_x, label_y,
                f'{d:+.3f}\np {p_text}', ha=align, fontsize=5.7,
                va='bottom' if d >= 0 else 'top')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(range(5))
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylim(-0.04, 0.145)
    ax.set_xlim(-0.50, 4.75)
    ax.set_ylabel('Δ MCC vs baseline')
    panel_label(ax, 'b', 'DUET - baseline')
    ax.text(0.98, 0.95, 'green: p < 0.05', transform=ax.transAxes, ha='right', va='top',
            fontsize=6, color=GREEN)

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.16, top=0.88, wspace=0.38)
    save(
        fig,
        'Fig3_main_results',
        exemptions=[{
            'panels': ['b'],
            'checks': ['panel-width'],
            'reason': 'Compact paired-difference panel beside the hero per-benchmark panel.',
        }],
    )


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    fig_leakage()
    fig_main_results()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
