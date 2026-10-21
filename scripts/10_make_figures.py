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
FIG = os.path.join(ROOT, 'figures')
T = ['TE18', 'TL12', 'JL10', 'RB9']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'legend.fontsize': 8,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

C = {'cobra_paper': '#9e9e9e', 'cobra_equiv': '#4c72b0', 'mlp': '#8c8c8c',
     'seq_ss': '#1D9E8F', 'ss_alt': '#7FBA6E', 'leak': '#C44E52', 'clean': '#4C9E6E'}


def arm_from_results(tag, ds='clean'):
    
    per = {s: [] for s in T}
    for f in glob.glob(os.path.join(ROOT, 'results', 'controlled',
                                    f'train_{ds}_{tag}_seed*.json')):
        r = json.load(open(f, encoding='utf-8'))
        tm = r.get('test_metrics', {})
        if not all(s in tm for s in T):
            continue
        for s in T:
            per[s].append(tm[s]['at_threshold']['MCC'])
    if not per[T[0]]:
        return None
    return {s: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
            for s, v in per.items()}


def m3_from_json(mode='clean', loss='bce'):
    p = os.path.join(ROOT, 'results', 'cobra_m3_equiv.json')
    if not os.path.exists(p):
        return None
    rows = json.load(open(p, encoding='utf-8')).get(mode, {}).get(loss, [])
    if not rows:
        return None
    return {s: (float(np.mean([r['test'][s]['at_threshold']['MCC'] for r in rows
                               if s in r['test']])),
                float(np.std([r['test'][s]['at_threshold']['MCC'] for r in rows
                              if s in r['test']], ddof=1)))
            for s in T}


def avg_of(d):
    return float(np.mean([d[s][0] for s in T])), float(
        np.mean([d[s][1] for s in T]))



PAPER = {
    'CoBRA-M3 (paper)': dict(TE18=0.190, TL12=0.546, JL10=0.293, RB9=0.557),
    'GRASSP (paper, 3 sets)': dict(TE18=0.359, TL12=0.611, JL10=0.343),
}


def fig4():
    arms = []
    m3c = m3_from_json('clean')
    for tag, label, color in [('seq_ss', 'R-GCN seq+ss (de-leaked)', C['seq_ss']),
                              ('seq_ss_alt', 'R-GCN seq+ss+ss_alt (de-leaked)', C['ss_alt']),
                              ('mlp', 'MLP on RNA-FM (de-leaked)', C['mlp'])]:
        d = arm_from_results(tag, 'clean')
        if d:
            arms.append((label, d, color, False))
    if m3c:
        arms.append(('CoBRA-M3 equiv retrained (de-leaked)', m3c, C['cobra_equiv'], False))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4),
                                  gridspec_kw={'width_ratios': [2.4, 1]})

    x = np.arange(len(T))
    n = len(arms) + 1
    w = 0.8 / n
    for i, (label, d, color, hatch) in enumerate(arms):
        pos = x + (i - (n - 1) / 2) * w
        vals = [d[s][0] for s in T]
        errs = [d[s][1] for s in T]
        ax.bar(pos, vals, w * 0.88, color=color, edgecolor='white', linewidth=0.5,
               yerr=errs, capsize=2, error_kw={'linewidth': 0.7})

    pos = x + (len(arms) - (n - 1) / 2) * w
    pv = [PAPER['CoBRA-M3 (paper)'][s] for s in T]
    ax.bar(pos, pv, w * 0.88, color=C['cobra_paper'], edgecolor='white',
           linewidth=0.5, hatch='//')
    ax.set_xticks(x)
    ax.set_xticklabels(T)
    ax.set_ylabel('MCC')
    ax.set_ylim(0, 0.75)
    ax.set_title('(A) Per benchmark', fontsize=10, fontweight='bold')
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c, _ in arms] +
              [plt.Rectangle((0, 0), 1, 1, color=C['cobra_paper'], hatch='//')],
              [lab for lab, _, _, _ in arms] + ['CoBRA-M3 (original benchmark, paper)'],
              fontsize=6.8, loc='upper left', ncol=1)


    labels, means, errs, colors = [], [], [], []
    for label, d, color, _ in arms:
        m, e = avg_of(d)
        labels.append(label)
        means.append(m)
        errs.append(e)
        colors.append(color)
    labels.append('CoBRA-M3 (paper)')
    means.append(float(np.mean(list(PAPER['CoBRA-M3 (paper)'].values()))))
    errs.append(0.0)
    colors.append(C['cobra_paper'])
    y = np.arange(len(labels))[::-1]
    ax2.barh(y, means, 0.6, color=colors, edgecolor='white', xerr=errs, capsize=2,
             error_kw={'linewidth': 0.7})
    for yi, m, e in zip(y, means, errs):
        ax2.text(m + e + 0.014, yi, f'{m:.3f}', va='center', fontsize=7.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels(["seq+ss", "seq+ss+ss_alt", "MLP (RNA-FM)",
                         "CoBRA-M3 equiv", "CoBRA-M3 (paper)"][:len(labels)],
                        fontsize=8)
    ax2.set_xlabel('Average MCC (4 benchmarks)')
    ax2.set_xlim(0, 0.55)
    ax2.set_title('(B) Average', fontsize=10, fontweight='bold')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'Fig4_performance.{ext}'))
    print('[written] Fig4_performance updated (de-leaked benchmark; paper values annotated)')
    plt.close(fig)



def fig5():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))


    rows = []
    for tag, label, color in [('mlp', 'MLP (RNA-FM)', C['mlp']),
                              ('seq_ss', 'R-GCN seq+ss', C['seq_ss']),
                              ('seq_ss_alt', 'R-GCN seq+ss+ss_alt', C['ss_alt'])]:
        oc, cc = arm_from_results(tag, 'orig'), arm_from_results(tag, 'clean')
        if oc and cc:
            rows.append((label, avg_of(oc)[0], avg_of(cc)[0], color))
    m3o, m3c = m3_from_json('orig'), m3_from_json('clean')
    if m3o and m3c:
        rows.append(('CoBRA-M3 equiv', avg_of(m3o)[0], avg_of(m3c)[0], C['cobra_equiv']))
    x = np.arange(len(rows))
    w = 0.36
    ax1.bar(x - w / 2, [r[1] for r in rows], w, color=C['leak'], edgecolor='white',
            label='Original benchmark (train=345)')
    ax1.bar(x + w / 2, [r[2] for r in rows], w, color=C['clean'], edgecolor='white',
            label='De-leaked (train=314)')
    for i, r in enumerate(rows):
        ax1.annotate(f'−{r[1]-r[2]:.3f}', (x[i], max(r[1], r[2]) + 0.012),
                     ha='center', fontsize=7.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([r[0] for r in rows], fontsize=7, rotation=20, ha='right')
    ax1.set_ylabel('Average MCC (4 benchmarks)')
    ax1.set_ylim(0, 0.55)
    ax1.legend(fontsize=7, loc='upper right')
    ax1.set_title('(A) Leak inflation is model-wide', fontsize=9.5, fontweight='bold')


    hp = os.path.join(ROOT, 'results', 'homology_audit.json')
    if os.path.exists(hp):
        d = json.load(open(hp, encoding='utf-8'))
        bands = ['100%', '95-99%', '90-94%', '80-89%', '70-79%', '<70%']
        present = [b for b in bands if b in d['strata']]
        xs = np.arange(len(present))
        w2 = 0.36
        ss = [d['strata'][b].get('seq+ss_mean', np.nan) for b in present]
        ml = [d['strata'][b].get('MLP_mean', np.nan) for b in present]
        ns = [d['strata'][b]['n'] for b in present]
        ax2.bar(xs - w2 / 2, ss, w2, color=C['seq_ss'], edgecolor='white',
                label='R-GCN seq+ss')
        ax2.bar(xs + w2 / 2, ml, w2, color=C['mlp'], edgecolor='white',
                label='MLP (RNA-FM)')
        for xi, n_ in zip(xs, ns):
            ax2.text(xi, 0.02, f'n={n_}', ha='center', fontsize=6.5, color='#444')
        ax2.set_xticks(xs)
        ax2.set_xticklabels([f'≥{b}' if b != '<70%' else '<70%' for b in present],
                            fontsize=7.5)
        ax2.set_xlabel('Sequence identity to closest training chain')
        ax2.set_ylabel('MCC (per-chain mean, 5 seeds)')
        ax2.legend(fontsize=7, loc='upper left')
        ax2.set_title('(B) Performance tracks homology', fontsize=9.5,
                      fontweight='bold')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'Fig5_leak_sensitivity.{ext}'))
    print('[written] Fig5_leak_sensitivity updated (A: leak inflation; B: homology strata)')
    plt.close(fig)



def figs1():
    arms = [('ss_w1', 'ss edges, unit weights (impl. control)'),
            ('ss_pw', 'ss edges weighted by P'),
            ('ss_pwshuf', 'ss weights shuffled'),
            ('seq_ss', 'seq+ss (reference)'),
            ('ssalt_p10', 'ss_alt binary P≥0.1'),
            ('ssalt_p10rand', 'ss_alt random edges (control)'),
            ('bp10w', 'merged bp weighted P≥0.1'),
            ('bplow_w', 'soft structure graph P≥0.01')]
    rows = []
    for tag, label in arms:
        d = arm_from_results(tag, 'clean')
        if d:
            m, e = avg_of(d)
            rows.append((label, m, e))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(rows))
    base = [r for r in rows if 'reference' in r[0]]
    ref = base[0][1] if base else 0.343
    ax.axvline(ref, color='#c44e52', ls='--', lw=1)
    ax.text(ref + 0.004, 0.1, f'reference seq+ss = {ref:.3f}', color='#c44e52',
            fontsize=7, rotation=90, va='bottom')
    for i, (label, m, e) in enumerate(rows):
        col = C['seq_ss'] if 'reference' in label else (
            '#9e9e9e' if 'control' in label or 'shuffled' in label else '#4c72b0')
        ax.barh(i, m, 0.6, color=col, edgecolor='white', xerr=e, capsize=2,
                error_kw={'linewidth': 0.7})
        ax.text(m + e + 0.006, i, f'{m:.3f}', va='center', fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlim(0, 0.42)
    ax.set_xlabel('Average MCC (4 benchmarks, 5 seeds ± s.d.)')
    ax.set_title('Probability-encoding variants (de-leaked, controlled)',
                 fontsize=9.5, fontweight='bold')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'FigS1_ablation.{ext}'))
    print('[written] FigS1_ablation updated (probability-encoding negative results, '
          'source: results/controlled)')
    plt.close(fig)


if __name__ == '__main__':
    fig4()
    fig5()
    figs1()
