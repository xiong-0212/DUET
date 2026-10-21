from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import ttest_rel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)

R = os.path.join(ROOT, 'results')
T = ['TE18', 'TL12', 'JL10', 'RB9']
DEV_SEEDS = [42, 0, 1, 2, 3]
METRICS = ['MCC', 'AUC', 'AUPRC', 'F1', 'Precision', 'Sensitivity', 'Specificity', 'GMean']

HEAD = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}   % auto-width, left aligned
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\usepackage{graphicx}
\usepackage[hidelinks]{hyperref}
\newcommand{\duet}{DUET}
% Supplementary tables are numbered S1..S5, matching the way the main text cites them
% ("Supplementary Table S1"), instead of restarting at 1 in a separate document.
\renewcommand{\thetable}{S\arabic{table}}
% the two tables moved here from the main text use this conditional resize
\newcommand{\fittable}[1]{\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{#1}}
\renewcommand{\thefigure}{S\arabic{figure}}
\begin{document}
"""

FOOT = r"""
\end{document}
"""


def load(name):
    return json.load(open(os.path.join(R, name), encoding='utf-8'))


def f3(x):
    return f'{x:.3f}'


def table(caption, label, head, rows, align, font=r'\footnotesize'):
    
    out = [r'\begin{table}[htbp]', r'\centering', font,
           r'\setlength{\tabcolsep}{4pt}',
           f'\\caption{{{caption}}}', f'\\label{{{label}}}',
           f'\\begin{{tabularx}}{{\\textwidth}}{{{align}}}', r'\toprule', head, r'\midrule']
    out += rows
    out += [r'\bottomrule', r'\end{tabularx}', r'\end{table}', '']
    return out


def controlled_mcc(tag, sets, seeds=DEV_SEEDS):
    
    vals = []
    for s in seeds:
        p = os.path.join(R, 'controlled', f'train_{tag}_seed{s}.json')
        if os.path.exists(p):
            j = json.load(open(p, encoding='utf-8'))
            vals.append(float(np.mean([j['test_metrics'][sp]['at_threshold']['MCC'] for sp in sets])))
    return np.array(vals) if vals else None


def s1_dataset():
    d = load('dataset_tables.json')
    rows = []
    name = {'train': 'Training', 'val': 'Validation', 'TE18': 'TE18', 'TL12': 'TL12',
            'JL10': 'JL10', 'RB9': 'RB9'}
    src = {'train': 'HARIBOSS + TR60', 'val': 'HARIBOSS + TR60', 'TE18': 'RNAsite',
           'TL12': 'ZeSTa', 'JL10': 'ZeSTa', 'RB9': 'RBind'}
    for phase, label in [('before', 'Before de-leaking'), ('after', 'After de-leaking')]:
        rows.append(f'\\multicolumn{{8}}{{l}}{{\\textit{{{label}}}}} \\\\')
        for k in ['train', 'val'] + T:
            v = d[phase][k]
            row = (f'{name[k]} & {src[k]} & {v["chains"]} & {v["nucleotides"]:,} & '
                   f'{v["positive"]:,} & {v["negative"]:,} & {v["positive_ratio"]:.3f} & '
                   f'{v["ratio"]:.2f} : 1 \\\\')
            rows.append(row.replace(',', '{,}'))
        rows.append(r'\addlinespace')
    return table(
        'Composition of every split before and after de-leaking. A nucleotide is labelled as '
        'binding (positive) if any of its heavy atoms lies within 4\\,\\AA{} of a ligand heavy '
        'atom. De-leaking removes the training and validation chains whose sequence is identical '
        'to a test-set sequence (31 training and 3 validation chains); the three chains shared by '
        'TE18 and RB9 are kept in TE18, which is why RB9 drops from nine chains to six. Generated '
        'by \\texttt{scripts/18\\_dataset\\_tables.py}.',
        'tab:s1',
        r'Dataset & Source & Chains & Nucleotides & Positive & Negative & Positive & Ratio \\'
        + '\n'
        + r'        &        &        &             & sites    & sites    & ratio    & (neg : pos) \\',
        rows, 'l l r r r r r r')


def s2_fixed_threshold():
    j = load('three_tower_tables.json')['fixed_threshold']
    rows = []
    order = ['CoBRA-M3 equiv. (retrained baseline)', 'DUET graph tower (R-GCN)',
             'DUET sequence tower A (TCN)', 'DUET sequence tower B (DenseNet)',
             'DUET (three towers)']
    label = {'CoBRA-M3 equiv. (retrained baseline)': 'CoBRA-M3 equiv. (baseline)',
             'DUET graph tower (R-GCN)': 'Graph tower (R-GCN)',
             'DUET sequence tower A (TCN)': 'Tower A (TCN)',
             'DUET sequence tower B (DenseNet)': 'Tower B (DenseNet)',
             'DUET (three towers)': r'\textbf{DUET (three towers)}'}
    for k in order:
        cells = ' & '.join(f'{j[k][sp]["MCC"]["mean"]:.3f}' for sp in T + ['average'])
        rows.append(f'{label[k]} & {cells} \\\\')
    return table(
        'Robustness check: every model thresholded at a fixed 0.5 instead of at its own '
        'validation-selected threshold. Ten seeds per row; four-benchmark average MCC. The '
        'ranking is unchanged, and the ensemble keeps its margin; the absolute values differ '
        'from Table~2 of the main text because the operating point is fixed rather than tuned.',
        'tab:s2', r'Model & TE18 & TL12 & JL10 & RB9 & Average \\', rows, 'l r r r r r')


def s3_all_metrics():
    v = load('per_set_metrics.json')
    keys = list(v['per_set'].keys())
    duet = next(k for k in keys if k.startswith('DUET'))
    base = next(k for k in keys if not k.startswith('DUET'))
    rows = []
    metric_label = {'Sensitivity': 'Sensitivity (recall)'}
    for m in METRICS:
        rows.append(r'\addlinespace')
        for who, lab in [(duet, 'DUET'), (base, 'baseline')]:
            cells = ' & '.join(f'{v["per_set"][who][m][sp]["mean"]:.3f} $\\pm$ '
                               f'{v["per_set"][who][m][sp]["sd"]:.3f}'
                               for sp in T + ['average'])
            rows.append(f'{metric_label.get(m, m)} & {lab} & {cells} \\\\')
        cells = []
        for sp in T + ['average']:
            d = v['significance'][m][sp]
            p = d['p']
            ps = '$<$0.001' if p < 0.001 else f'{p:.3f}'
            cells.append(f'{d["delta"]:+.3f} ({ps})')
        rows.append(r'  & $\Delta$ ($p$) & ' + ' & '.join(cells) + r' \\')
    return table(
        'Full eight-metric comparison of DUET (three towers) with the retrained '
        'CoBRA-M3-equivalent baseline, per benchmark. Each of the ten seeds was evaluated at its '
        'own validation-selected threshold; means and standard deviations are over seeds, '
        '$\\Delta$ is the same-seed paired difference (DUET $-$ baseline) with the paired '
        '$t$-test $p$-value in parentheses. The last block repeats the four-benchmark average '
        'that the main text cites. Generated by \\texttt{scripts/17\\_per\\_set\\_metrics.py}.',
        'tab:s3', r'Metric & Model & TE18 & TL12 & JL10 & RB9 & Average \\', rows,
        'l l R R R R R', font=r'\scriptsize')


def s4_cobra_reproduction():

    rows = [
        r'Reproduction (hand-written inference, fp32) & 0.212 & 0.568 & 0.329 & 0.671 & 0.445 \\',
        r'Official code, end to end (new download) & 0.212 & 0.568 & 0.329 & 0.671 & 0.445 \\',
        r'Values printed in the CoBRA paper (M3) & 0.190 & 0.546 & 0.293 & 0.557 & 0.396 \\',
    ]
    return table(
        'Reproduction of the released CoBRA model (CoBRA-M3: RiNALMo with an MLP head and a '
        'binary cross-entropy loss). The released checkpoint and the values printed in the '
        'original publication could not be reconciled: the checkpoint scores higher on every '
        'benchmark than the published values, and no single evaluation convention (threshold '
        'sweep, removal of the overlapping test chains, per-chain instead of pooled averaging, '
        'the authors\' own inference code) reproduces them. Our baseline is therefore a model we '
        'retrained ourselves, and the comparison in the main text is made against that retrained '
        'model, not against the published numbers. Numbers transcribed from '
        '\\texttt{baselines/cobra\\_official/} (2026-10-11).',
        'tab:s4', r'CoBRA-M3 (MCC) & TE18 & TL12 & JL10 & RB9 & Average \\', rows,
        'l r r r r r')


def s5_backbones():
    sets3 = ['TE18', 'TL12', 'JL10']
    rows = []
    for lab, tag in [('RNA-FM (640-d)', 'clean_mlp'), ('RiNALMo-giga (1280-d)', 'clean_dual2_mlp'),
                     ('ERNIE-RNA (768-d)', 'clean_ernie_mlp')]:
        m = controlled_mcc(tag, sets3)
        if m is None:
            continue
        per = {sp: controlled_mcc(tag, [sp]) for sp in sets3}
        cells = ' & '.join(f'{per[sp].mean():.3f}' for sp in sets3)
        rows.append(f'{lab} & {cells} & {m.mean():.3f} $\\pm$ {m.std(ddof=1):.3f} \\\\')
    return table(
        'Backbone comparison at fixed tower architecture (a per-residue MLP) and fixed training '
        'protocol, five development seeds, three benchmarks (RB9 has no official preprocessed '
        'ERNIE-RNA features). The three backbones are within noise of one another; the ordering '
        'flips between benchmarks, so the choice of language model is not the bottleneck at this '
        'data scale. Generated from \\texttt{results/controlled/train\\_clean\\_*\\_seed*.json}.',
        'tab:s5', r'Backbone (per-residue MLP tower) & TE18 & TL12 & JL10 & Three-set average \\',
        rows, 'l r r r r')




GRID = [
    ('clean_seq_ss', r'\textbf{Binary MFE edges (\texttt{seq}+\texttt{ss}), reference}', '---'),
    ('clean_ssalt_p10', 'Extra relation from pairs at $P \\ge 0.1$', 'threshold sweep'),
    ('clean_ssalt_p20', 'Extra relation from pairs at $P \\ge 0.2$', 'threshold sweep'),
    ('clean_ssalt_p30w', 'Extra relation from pairs at $P \\ge 0.3$', 'threshold sweep'),
    ('clean_ssalt_p50', 'Extra relation from pairs at $P \\ge 0.5$', 'threshold sweep'),
    ('clean_ss_w1', 'All edges carry weight 1 (control for weighting)', 'control'),
    ('clean_ss_pw', 'MFE edges weighted by $P$', 'weighted MFE edges'),
    ('clean_ss_pwshuf', 'MFE edge weights shuffled within each chain', 'control'),
    ('clean_bp10', 'Merged relation (MFE $\\cup$ $P\\ge0.1$), binary', 'merged relation'),
    ('clean_bp10w', 'Merged relation (MFE $\\cup$ $P\\ge0.1$), weighted', 'merged relation'),
    ('clean_bplow_bin', 'Soft graph $P\\ge0.01$, binary', 'soft graph'),
    ('clean_bplow_w', 'Soft graph $P\\ge0.01$, weighted', 'soft graph'),
    ('clean_bplow_shuf', 'Soft graph, weights shuffled', 'control'),
    ('clean_pf_ss', 'Five probability-derived node features', 'node features'),
    ('clean_pf_ss_shuf', 'The same five features, shuffled', 'control'),
    ('clean_ss_prune90', 'MFE edges pruned at $P \\ge 0.9$', 'confidence pruning'),
    ('clean_ss_prune70', 'MFE edges pruned at $P \\ge 0.7$', 'confidence pruning'),
    ('clean_ssalt_p10w', 'Extra relation weighted by $P \\ge 0.1$', 'weighted extra edges'),
    ('clean_ssalt_p10shuf', 'The same edges, weights shuffled', 'control'),
    ('clean_ssalt_p10rand', 'The same number of random edges', 'control'),
]


def s6_probability_grid():
    ref = controlled_mcc('clean_seq_ss', T)
    rows = []
    for tag, label, kind in GRID:
        m = controlled_mcc(tag, T)
        if m is None:
            continue
        if tag == 'clean_seq_ss':
            rows.append(f'{label} & {kind} & {m.mean():.3f} $\\pm$ {m.std(ddof=1):.3f} & '
                        f'--- & --- \\\\')
            continue
        d = m - ref
        p = float(ttest_rel(m, ref).pvalue)
        ps = '$<$0.001' if p < 0.001 else f'{p:.3f}'
        mark = r'\textbf{' + f'{d.mean():+.3f}' + '}' if p < 0.05 else f'{d.mean():+.3f}'
        rows.append(f'{label} & {kind} & {m.mean():.3f} $\\pm$ {m.std(ddof=1):.3f} & {mark} & {ps} \\\\')
    head = (r'Variant & Family & MCC & $\Delta$ vs reference & $p$ \\')
    return table(
        'Probability-encoded edges: the complete grid of controlled experiments. Each arm was '
        'trained five times (seeds 42, 0, 1, 2, 3) on the de-leaked split with the same loss, '
        'optimiser and schedule as the graph tower, and evaluated at its own '
        'validation-selected threshold; MCC is the four-benchmark average. $\\Delta$ is the '
        'same-seed paired difference against the binary reference, and bold marks '
        '$p < 0.05$. Control arms replace real probabilities with shuffled weights, shuffled '
        'features or random edges of the same density. No variant improves on the binary '
        '\\texttt{seq}+\\texttt{ss} graph, and the extra relation built at $P \\ge 0.3$ is '
        'significantly worse. Generated from \\texttt{results/controlled/train\\_*.json}.',
        'tab:s6', head, rows, 'l l r r r', font=r'\scriptsize')


def s6_configs():
    
    ROW = ' \\\\'
    groups = [
        ('DUET, three towers (one configuration for all of them)', [
            ('Architecture',
             'graph tower: two R-GCN layers, hidden width 128. Sequence tower A (TCN): four '
             'residual dilated blocks, 160 channels, dilations 1-2-4-8. Sequence tower B '
             '(DenseNet): three dense blocks of four layers, growth rate 40, stem 160, '
             '1x1 transitions.'),
            ('Input', 'graph tower: RNA-FM 640-d, frozen. Sequence towers: RiNALMo-giga 1280-d, '
                      'frozen.'),
            ('Loss', 'TCL-focal: class-balanced focal ($\\gamma = 3$, $\\beta = 0.999$) plus '
                     'triplet centre loss ($\\lambda = 0.2$, margin 4) on the 64-d head '
                     'representation.'),
            ('Optimiser', 'AdamW, learning rate $5\\times10^{-4}$, weight decay $10^{-4}$, batch '
                          'size 8, cosine annealing with warm restarts ($T_0 = 10$, '
                          '$T_{\\mathrm{mult}} = 2$).'),
            ('Early stopping', 'validation MCC, patience 15 epochs, 120-epoch cap.'),
            ('Decision threshold',
             'searched on the validation split over 0.05 to 0.95 in steps of 0.01, maximising '
             'validation MCC; each seed uses its own.'),
            ('Tuning budget',
             'no per-split hyper-parameter search: the configuration was fixed after the '
             'development batch and reused unchanged on every split.'),
        ]),
        ('CoBRA-M3-equivalent baseline (retrained by us)', [
            ('Architecture',
             "the authors' predictor: 1280-d RiNALMo embeddings into an MLP 1024-256-128-64, "
             'dropout 0.1, two-class output.'),
            ('Loss', 'binary cross-entropy, the loss of the CoBRA-M3 configuration.'),
            ('Optimiser',
             'the same as the towers: AdamW, learning rate $5\\times10^{-4}$, batch size 8, cosine '
             'warm restarts, same early stopping, same threshold search.'),
            ('Tuning budget', 'same seeds, same splits and same protocol as DUET; no per-split '
                              'tuning.'),
        ]),
        ('GRASSP (retrained by us)', [
            ('Architecture',
             "the authors' model: ERNIE-RNA features with secondary-structure descriptors over "
             'the graph, two GATConv layers with adaptive gating.'),
            ('Search space',
             'the search space of the authors: learning rate {1e-4, 1e-3, 1e-2}, batch size '
             '{1, 16}, hidden {86, 128}, dropout {0.3, 0.6}.'),
            ('Selection', 'configuration chosen on the validation split of the de-leaked data, '
                          'exactly as the authors do; 100 epochs, batch 1, weight decay 1e-4.'),
            ('Tuning budget', 'the full grid was searched per split, so GRASSP received a larger '
                              'tuning budget than DUET.'),
        ]),
    ]
    rows = []
    for title, items in groups:
        rows.append('\\multicolumn{2}{l}{\\textit{' + title + '}}' + ROW)
        for k, v in items:
            rows.append(k + ' & ' + v + ROW)
        rows.append('\\addlinespace')
    return table(
        'Training configuration, selection rule and tuning budget of every model retrained for '
        'this study. All models were trained on the same splits with the same ten seeds and the '
        'same evaluation protocol; they differ in architecture and, for GRASSP, in the '
        'hyper-parameter search space its authors specify. No language model is fine-tuned at any '
        'point.',
        'tab:s6', 'Property & Setting' + ROW, rows, 'l L')


def s7_robustness():
    
    BS = chr(92)
    ROW = ' ' + BS * 2
    TT = BS + 'texttt'
    PCT = BS + '%'
    CMD = BS
    rig = load('rigour_analyses.json')
    extra = load('reporting_extras.json')
    d, b = rig['chain_level']['duet'], rig['chain_level']['baseline']
    h = rig['hierarchical_delta']['MCC']
    loo = rig['leave_one_out']
    th, cal, dv, fu = rig['threshold'], rig['calibration'], rig['diversity'], rig['fusion']
    rows = [
        BS + 'multicolumn{3}{l}{' + BS + 'textit{Chain-level inference (unit of replication)}}'
        + ROW,
        'DUET, four-benchmark average MCC & ' +
        '{:.3f} (95' .format(d['mean']) + PCT + ' CI {:.3f} to {:.3f}) & chain bootstrap'.format(
            d['ci95'][0], d['ci95'][1]) + ROW,
        'Baseline, four-benchmark average MCC & ' +
        '{:.3f} (95'.format(b['mean']) + PCT + ' CI {:.3f} to {:.3f}) & chain bootstrap'.format(
            b['ci95'][0], b['ci95'][1]) + ROW,
        'Paired difference & ' +
        '{:+.3f} (95'.format(h['delta']) + PCT + ' CI {:+.3f} to {:+.3f}, $p = {:.3f}$) & '
        'benchmarks, chains and seeds resampled'.format(
            h['ci95'][0], h['ci95'][1], h['p_boot']) + ROW,
        BS + 'addlinespace',
        BS + 'multicolumn{3}{l}{' + BS + 'textit{Aggregation and thresholds}}' + ROW,
        'Macro-average (benchmarks equally weighted) & {:.3f} vs {:.3f} & DUET vs baseline'.format(
            rig['macro_micro']['duet']['macro'], rig['macro_micro']['baseline']['macro']) + ROW,
        'Micro-average (residues pooled) & {:.3f} vs {:.3f} & DUET vs baseline'.format(
            rig['macro_micro']['duet']['micro'], rig['macro_micro']['baseline']['micro']) + ROW,
        'Leave-one-benchmark-out difference & ' +
        ', '.join('{} {:+.3f}'.format(k, loo['duet']['loo_benchmark'][k]
                                      - loo['baseline']['loo_benchmark'][k]) for k in T)
        + ' & MCC' + ROW,
        'Fixed threshold 0.5 & {:.3f} vs {:.3f} & DUET vs baseline'.format(
            th['duet']['fixed_0.5'], th['baseline']['fixed_0.5']) + ROW,
        'Oracle threshold, selected on the test data & {:.3f} vs {:.3f} & DUET vs baseline'.format(
            th['duet']['oracle_threshold'], th['baseline']['oracle_threshold']) + ROW,
        BS + 'addlinespace',
        BS + 'multicolumn{3}{l}{' + BS + 'textit{Multiplicity}}' + ROW,
        'Benjamini-Hochberg over eight metrics on four benchmarks and the average & ' +
        '{} of {} survive at $q < 0.05$ & sensitivity, per-benchmark AUC and several secondary '
        'contrasts do not'.format(
            sum(1 for v in extra['multiplicity']['all'].values() if v['survives']),
            len(extra['multiplicity']['all'])) + ROW,
        'Within the eight-metric average & {} of {} survive & only sensitivity does not'.format(
            sum(1 for v in extra['multiplicity']['average_only'].values() if v['survives']),
            len(extra['multiplicity']['average_only'])) + ROW,
        BS + 'addlinespace',
        BS + 'multicolumn{3}{l}{' + BS + 'textit{Calibration}}' + ROW,
        'Brier score & ' + ', '.join('{} {:.3f}'.format(k, cal[k]['brier'])
                                     for k in ['graph', 'tcn', 'dense', 'duet', 'baseline'])
        + ' & towers, ensemble, baseline' + ROW,
        'Expected calibration error, 10 bins & ' +
        ', '.join('{} {:.3f}'.format(k, cal[k]['ece_10bin'])
                  for k in ['graph', 'tcn', 'dense', 'duet', 'baseline'])
        + ' & towers, ensemble, baseline' + ROW,
        BS + 'addlinespace',
        BS + 'multicolumn{3}{l}{' + BS + 'textit{Member diversity (per-residue probabilities)}}'
        + ROW,
    ]
    for key, lab in [('graph|tcn', 'graph tower vs TCN'), ('graph|dense', 'graph tower vs DenseNet'),
                     ('tcn|dense', 'TCN vs DenseNet')]:
        v = dv[key]
        rows.append('{} & correlation $r={:.2f}$, $Q={:.2f}$, disagreement {:.3f}, '
                    'double fault {:.3f} & '.format(lab, v['prob_correlation'],
                                                    v['q_statistic'], v['disagreement'],
                                                    v['double_fault']) + ROW)
    rows.append('At least one member correct & {:.3f} of residues & oracle bound'.format(
        dv['oracle_any_member_correct']) + ROW)
    rows.append(BS + 'addlinespace')
    rows.append(BS + 'multicolumn{3}{l}{' + BS +
                'textit{Fusion rules, four-benchmark average MCC}}' + ROW)
    for key, lab in [('plain', 'equal-weight probabilities'), ('rank', 'rank averaging'),
                     ('weighted_val', 'validation-weighted averaging'),
                     ('logit', 'logit averaging'), ('majority', 'majority vote')]:
        rows.append('{} & {:.3f} & '.format(lab, fu[key]) + ROW)
    rows.append(BS + 'addlinespace')
    rows.append(BS + 'multicolumn{3}{l}{' + BS + 'textit{Runtime}}' + ROW)
    rows.append('Training time per tower & one to two minutes & one RTX 3060, 314-chain split'
                + ROW)
    rows.append('Inference, single-threaded CPU & ' +
                ', '.join('{} {:.0f} ms per chain'.format(k, extra['inference_cost_cpu'][k]['seconds_per_chain'] * 1000)
                          for k in ['graph', 'tcn', 'dense'])
                + ' & ensemble below 0.1 s per chain' + ROW)
    caption = ('Robustness, calibration, diversity and runtime analyses. The chain-level bootstrap '
               'resamples benchmarks, then chains within a benchmark, then seeds, so its interval '
               'is wider than the seed-level paired $t$-test and reflects the limited number of '
               'test chains. Values come from the released checkpoints via ' + TT +
               '{scripts/27' + BS + '_rigour' + BS + '_analyses.py} and ' + TT +
               '{scripts/29' + BS + '_reporting' + BS + '_extras.py}.')
    return table(caption, 'tab:s7', 'Quantity & Value & Notes' + ROW, rows, 'L L L', font=r'\scriptsize')


def s8_strict():
    
    BS = chr(92)
    ROW = ' ' + BS * 2
    st = load('strict_summary.json')['thresholds']
    rows = []
    label = {'p80': '80', 'p70': '70', 'p50': '50'}
    for thr in ['p80', 'p70', 'p50']:
        if thr not in st:
            continue
        e = st[thr]
        rows.append(BS + 'multicolumn{6}{l}{' + BS + 'textit{every chain at ' + label[thr] + BS + '% identity or above removed: ' + str(e['n_train']) +
                    ' training and ' + str(e['n_val']) + ' validation chains remain}}' + ROW)
        rows.append('Benchmark & DUET & Baseline & $' + BS + 'Delta$ & $p$ & (n chains)' + ROW)
        for sp in T:
            b = e['per_benchmark'][sp]
            rows.append('{} & {:.3f} & {:.3f} & {:+.3f} & {:.3f} & {}'.format(
                sp, b['duet'], b['baseline'], b['delta'], b['p'],
                {'TE18': 18, 'TL12': 12, 'JL10': 10, 'RB9': 6}[sp]) + ROW)
        rows.append('Four-benchmark average & {:.3f} & {:.3f} & {:+.3f} & {} & 46'.format(
            e['duet_mean'], e['baseline_mean'], e['delta'],
            'below 0.001' if e['p_seed_level'] < 0.001 else '{:.3f}'.format(e['p_seed_level']))
            + ROW)
        if 'grassp' in e:
            g = e['grassp']
            rows.append('GRASSP, three shared benchmarks & {:.3f} (DUET) & {:.3f} & {:+.3f} & '
                        '{:.3f} & {} seeds'.format(g['duet_3set'], g['mean'], g['delta'], g['p'],
                                                   g['n']) + ROW)
        rows.append(BS + 'addlinespace')
    return table(
        'Stricter de-leaking, per benchmark. In each block a training or validation chain is '
        'removed as soon as its maximum identity to a test chain reaches the threshold in the '
        'block heading, and DUET, the CoBRA-M3-equivalent baseline and GRASSP are retrained. '
        'Each seed is evaluated at its own validation-selected threshold and the p-value is the '
        'same-seed paired $t$-test. The last row of a block compares DUET and GRASSP on the three '
        'benchmarks the two studies share. Generated by '
        + chr(92) + 'texttt{scripts/30' + chr(92) + '_strict' + chr(92) + '_summary.py}.',
        'tab:s8', 'Group & DUET & Baseline & $' + chr(92) + 'Delta$ & $p$ & Notes' + ROW,
        rows, 'l r r r r l')


def s9_external():
    
    BS = chr(92)
    ROW = ' ' + BS * 2
    aud = load('external_data_audit.json')
    ev = load('external_eval.json')['per_set']
    ebase = load('external_eval_baseline.json')['per_set']
    rows = [BS + 'multicolumn{6}{l}{' + BS + 'textit{How the external splits were de-leaked}}'
            + ROW,
            'Split & listed & built & after internal de-duplication & removed (80' + BS +
            '% vs train/val) & final' + ROW]
    nice = {'hariboss_test': 'HARIBOSS test', 'external_jiang': 'Jiang external',
            'time_dependent': 'Time dependent'}
    for name, d in aud['splits'].items():
        rows.append('{} & {} & {} & {} & {} & {}'.format(
            nice.get(name, name), d['n_listed'], d['n_built'], d['n_after_internal_dedup'],
            len(d['vs_train_val']['removed']), d['n_final']) + ROW)
    rows.append(BS + 'addlinespace')
    rows.append(BS + 'multicolumn{6}{l}{' + BS +
                'textit{Performance on the surviving chains, ten seeds}}' + ROW)
    rows.append('Split & chains & residues & DUET MCC & baseline MCC & $' + BS + 'Delta$ ($p$)'
                + ROW)
    for name, d in ev.items():
        b = ebase.get(name, {})
        pv = b.get('paired_vs_duet', {})
        base_txt = ('{:.3f} $'.format(b['MCC']['mean']) + BS + 'pm$ {:.3f}'.format(b['MCC']['sd'])
                    if b else '---')
        pv_txt = ('{:+.3f} ({})'.format(pv['delta'],
                                        'below 0.001' if pv['p'] < 0.001 else '{:.3f}'.format(pv['p']))
                  if pv else '---')
        rows.append('{} & {} & {} & {:.3f} $'.format(
            nice.get(name, name), d['n_chains'], d['n_residues'], d['MCC']['mean']) + BS +
            'pm$ {:.3f} & {} & {}'.format(d['MCC']['sd'], base_txt, pv_txt) + ROW)
    caption = ('External validation on the SMARTPocket release. The upper block reports the '
               'de-leaking cascade: chains listed by the release, chains that could be rebuilt '
               'from the structures, chains surviving internal de-duplication at 80 ' + BS + '% '
               'identity, chains removed because they match our training or validation split, and '
               'the final count. The lower block reports both models on the surviving chains, each '
               'seed at the threshold it selected on its own validation split. The label '
               'convention of that release marks fewer residues as binding than our benchmarks do, '
               'so its absolute MCC values are comparable only between the two models. Generated '
               'by ' + BS + 'texttt{scripts/32' + BS + '_external' + BS + '_data.py}, ' + BS +
               'texttt{33' + BS + '_external' + BS + '_eval.py} and ' + BS + 'texttt{34' + BS +
               '_external' + BS + '_baseline.py}.')
    return table(caption, 'tab:s9',
                 'Group & Split & ' + BS + 'multicolumn{4}{l}{Value}' + ROW, rows, 'l l r r r r',
                 font=r'\scriptsize')


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    p1 = os.path.join(ROOT, 'paper', 'supplementary.tex')
    p2 = os.path.join(ROOT, 'paper', 'supplementary2.tex')

    lines = [HEAD,
             r'\section*{Additional file 1: Supplementary tables}', '',
             r'This file contains the supplementary tables of the manuscript '
             r'\emph{DUET: a sequence-only RNA language-model ensemble for RNA--small-molecule '
             r'binding-site prediction on de-leaked benchmarks}. Every table is '
             r'generated from the released checkpoints and result files; the script that '
             r'produces it is given in each caption.', '']
    lines += [r'\section*{S1. Dataset composition}', '']
    lines += s1_dataset()
    lines += [r'\section*{S2. Robustness at a fixed threshold}', '']
    lines += s2_fixed_threshold()
    lines += [r'\section*{S3. All eight metrics, per benchmark}', '']
    lines += s3_all_metrics()
    lines += [r'\section*{S4. Reproduction of the CoBRA release}', '']
    lines += s4_cobra_reproduction()
    lines += [r'\section*{S5. Backbone comparison}', '']
    lines += s5_backbones()
    lines += [r'\section*{S6. Training configuration and budget of every retrained model}', '']
    lines += s6_configs()
    lines += [r'\section*{S7. Robustness, calibration, diversity and runtime}', '']
    lines += s7_robustness()
    lines += [r'\section*{S8. Stricter de-leaking, per benchmark}', '']
    lines += s8_strict()
    lines += [r'\section*{S9. External validation on an independent release}', '']
    lines += s9_external()
    lines += [r'\section*{S9. External validation on an independent release}', '']
    lines += s9_external()
    lines += [r'\section*{S9. External validation on an independent release}', '']
    lines += s9_external()
    lines += [r'\section*{S10 and S11. Full combination and ablation grids}', '',
              r'\input{si_extra_tables}', '']
    lines += [r'\begin{figure}[htbp]', r'\centering',
              r'\includegraphics[width=\textwidth]{figures/FigS1_calibration.pdf}',
              (r'\caption{Probability calibration and selective prediction. '
               r'(a)~Reliability diagram for a single uncalibrated model (gray) and the DUET '
               r'ensemble (green), pooled across the four reference-matched benchmarks. Marker '
               r'area is proportional to the number of residues in each probability bin; ECE '
               r'values are reported in Table~S7. (b)~MCC, precision and sensitivity (recall) '
               r'as the retained fraction of the highest-confidence residues decreases from '
               r'1.0 to 0.2.}'),
              r'\label{fig:s1}', r'\end{figure}', '']
    lines += [FOOT]
    open(p1, 'w', encoding='utf-8').write('\n'.join(lines))
    print(f'[written] {p1}')

    lines2 = [HEAD,
              r'\section*{Additional file 2: probability-encoded edges}', '',
              r'The main text reports that no way of using RNAfold base-pairing probabilities '
              r'instead of binary minimum-free-energy edges improved on the simple graph. This '
              r'file gives the complete grid behind that statement.', '']
    lines2 += s6_probability_grid()
    lines2 += [FOOT]
    open(p2, 'w', encoding='utf-8').write('\n'.join(lines2))
    print(f'[written] {p2}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
