from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
T = ['TE18', 'TL12', 'JL10', 'RB9']
SEEDS = [42, 0, 1, 2, 3]

ARMS = {

    'mlp':            ('Reference', 'MLP only (CoBRA-M2 equivalent)'),
    'seq_ss':         ('Reference', 'R-GCN seq+ss (current best, de-leaked)'),
    'seq_ss_alt':     ('Reference', 'R-GCN seq+ss+ss_alt (binary, P≥0.3)'),

    'ssalt_p10':      ('1) Threshold', 'seq+ss+ss_alt binary P≥0.1'),
    'ssalt_p20':      ('1) Threshold', 'seq+ss+ss_alt binary P≥0.2'),
    'ssalt_p50':      ('1) Threshold', 'seq+ss+ss_alt binary P≥0.5'),

    'ssalt_p10w':     ('2) Weighted alt', 'seq+ss+ss_alt weighted P≥0.1'),
    'ssalt_p10shuf':  ('2) Control', 'seq+ss+ss_alt shuffled weights P≥0.1'),
    'ssalt_p10rand':  ('2) Control', 'seq+ss+ss_alt random edges P≥0.1 (same count)'),
    'ssalt_p30w':     ('2) Weighted alt', 'seq+ss+ss_alt weighted P≥0.3'),

    'ss_pw':          ('3) Weighted ss', 'seq+ss weighted (ss edge weight = pair probability)'),
    'ss_pwshuf':      ('3) Control', 'seq+ss shuffled weights'),
    'ss_w1':          ('3) Control', 'seq+ss all-ones weights (implementation control)'),

    'bp10':           ('4) Merged', 'seq+bp binary (merged MFE∪sub≥0.1 relation)'),
    'bp10w':          ('4) Merged', 'seq+bp weighted (merged relation, edge weight = probability)'),

    'bplow_bin':      ('5) Soft structure', 'seq+bp binary P≥0.01 (dense control)'),
    'bplow_w':        ('5) Soft structure', 'seq+bp weighted P≥0.01 (soft-structure graph)'),
    'bplow_shuf':     ('5) Control', 'seq+bp shuffled weights P≥0.01'),
    'ssalt_p01w':     ('5) Soft structure', 'seq+ss+ss_alt weighted P≥0.01'),

    'pf_ss':          ('6) Node features', 'seq+ss + probability node features (5-d standardized)'),
    'pf_ss_shuf':     ('6) Control', 'seq+ss + probability node features (shuffled within row)'),

    'ss_prune90':     ('7) Pruning', 'seq+ss (keep only MFE pairs with P≥0.9)'),
    'ss_prune70':     ('7) Pruning', 'seq+ss (keep only MFE pairs with P≥0.7)'),
}


PAIRS = [
    ('ssalt_p10', 'seq_ss',        'ss_alt threshold lowered to 0.1 (binary)'),
    ('ssalt_p10', 'ssalt_p10rand', 'real suboptimal edges vs random edges (same count)'),
    ('ssalt_p10w', 'ssalt_p10shuf', 'weighted ss_alt: real weights vs shuffled weights'),
    ('ss_pw', 'seq_ss',            'weighted MFE (same edge set) vs binary MFE'),
    ('ss_pw', 'ss_pwshuf',         'weighted MFE: real weights vs shuffled weights'),
    ('ss_pw', 'ss_w1',             'weighted MFE vs all-ones weights (implementation control)'),
    ('bp10w', 'bp10',              'merged bp relation: weighted vs binary'),
    ('bp10w', 'ss_pw',             'merged bp weighted vs MFE-only weighted'),
    ('bp10w', 'seq_ss',            'merged bp weighted vs seq+ss baseline'),
    ('bplow_w', 'bplow_bin',       'soft-structure graph: weighted vs binary (same edge set)'),
    ('bplow_w', 'bplow_shuf',      'soft-structure graph: real weights vs shuffled weights'),
    ('bplow_w', 'seq_ss',          'soft-structure graph vs seq+ss baseline'),
    ('ssalt_p01w', 'seq_ss',       'weighted ss_alt (P≥0.01) vs seq+ss baseline'),
    ('pf_ss', 'seq_ss',            'probability node features vs seq+ss baseline'),
    ('pf_ss', 'pf_ss_shuf',        'probability node features: real positions vs shuffled positions'),
    ('ss_prune90', 'seq_ss',       'confidence pruning P≥0.9 vs full MFE'),
    ('ss_prune70', 'seq_ss',       'confidence pruning P≥0.7 vs full MFE'),
]


def load_arm(tag):
    
    out = {}
    for f in glob.glob(os.path.join(ROOT, 'results', 'controlled',
                                    f'train_clean_{tag}_seed*.json')):
        base = os.path.basename(f)[len('train_'):-len('.json')]
        seed = int(base.rpartition('_seed')[2])
        r = json.load(open(f, encoding='utf-8'))
        tm = r.get('test_metrics', {})
        if not all(s in tm for s in T):
            continue
        per = {s: tm[s]['at_threshold']['MCC'] for s in T}
        out[seed] = {'avg': float(np.mean(list(per.values()))), 'per': per,
                     'auc': float(np.mean([tm[s]['at_threshold']['AUC'] for s in T]))}
    return out


def paired(a, b, seeds):
    
    va = np.array([a[s]['avg'] for s in seeds])
    vb = np.array([b[s]['avg'] for s in seeds])
    d = va - vb
    if len(d) < 2 or np.allclose(d, d[0]):
        p = 1.0 if np.allclose(d, 0) else 0.0
    else:
        p = float(stats.ttest_rel(va, vb).pvalue)
    pos = int((d > 0).sum())
    return float(d.mean()), p, pos, len(d)


def fmt(vals):
    if not vals:
        return '—'
    return f'{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}' if len(vals) > 1 \
        else f'{vals[0]:.3f}'


def main():
    data = {tag: load_arm(tag) for tag in ARMS}
    seeds = [s for s in SEEDS if s in data.get('seq_ss', {})]
    lines = ['# Scheme B: probability-weighted edges / threshold scan / random controls (auto-generated)', '',
             f'> Generated 2026-10-10 · source `results/controlled/train_clean_*_seed*.json`',
             f'> Protocol: TCL-Focal + `--drop_val_leaked` + epochs=120, de-leaked training set (train=314), '
             f'seeds {SEEDS}, metric = mean MCC over the 4 CoBRA test sets (`at_threshold`)', '']


    lines += ['## 1. Per-arm results (5 seeds, MCC ± std)', '',
              '| Category | Configuration | n | Mean MCC | Mean AUC | ' + ' | '.join(T) + ' |',
              '|:---|:---|---:|---:|---:|' + '---:|' * 4]
    for tag, (cat, label) in ARMS.items():
        d = data.get(tag, {})
        if not d:
            continue
        avgs = [d[s]['avg'] for s in sorted(d)]
        auc = [d[s]['auc'] for s in sorted(d)]
        cols = [fmt([d[s]['per'][t] for s in sorted(d)]) for t in T]
        lines.append(f'| {cat} | {label} | {len(d)} | **{fmt(avgs)}** | {fmt(auc)} | '
                     + ' | '.join(cols) + ' |')
    lines.append('')


    ref = data.get('seq_ss', {})
    lines += ['## 2. Same-seed paired tests vs the seq+ss baseline (0.343) (n=5)', '',
              '| Configuration | Mean MCC | Δ vs seq+ss | p | Same-direction seeds |', '|:---|---:|---:|---:|---:|']
    for tag, (cat, label) in ARMS.items():
        d = data.get(tag, {})
        common = [s for s in seeds if s in d and s in ref]
        if not common or tag in ('seq_ss',):
            continue
        md, p, pos, n = paired(d, ref, common)
        lines.append(f'| {label} | {fmt([d[s]["avg"] for s in common])} | '
                     f'{md:+.3f} | {p:.3f} | {pos}/{n} |')
    lines.append('')


    lines += ['## 3. Key control pairs (same seed)', '', '| Comparison | Δ | p | Same direction |', '|:---|---:|---:|---:|']
    for a_tag, b_tag, desc in PAIRS:
        a, b = data.get(a_tag, {}), data.get(b_tag, {})
        common = [s for s in seeds if s in a and s in b]
        if not common:
            lines.append(f'| {desc} | — | — | — |')
            continue
        md, p, pos, n = paired(a, b, common)
        star = '**' if p < 0.05 else ''
        lines.append(f'| {desc} | {star}{md:+.3f}{star} | {p:.3f} | {pos}/{n} |')
    lines.append('')


    lines += ['## 4. Interpretation (auto-generated, for manual review)', '']
    verdicts = []
    for a_tag, b_tag, desc in PAIRS[:8]:
        a, b = data.get(a_tag, {}), data.get(b_tag, {})
        common = [s for s in seeds if s in a and s in b]
        if not common:
            continue
        md, p, pos, n = paired(a, b, common)
        mark = 'sig+' if (p < 0.05 and md > 0) else ('sig-' if (p < 0.05 and md < 0) else 'n.s.')
        verdicts.append(f'- {mark} {desc}: Δ={md:+.3f} (p={p:.3f}, {pos}/{n} same direction)')
    lines += verdicts + ['']

    best_tag, best_val = None, -9
    for tag in ARMS:
        d = data.get(tag, {})
        if d:
            v = float(np.mean([d[s]['avg'] for s in d]))
            if v > best_val:
                best_tag, best_val = tag, v
    if best_tag:
        lines.append(f'Best in this batch: **{ARMS[best_tag][1]}** = {best_val:.3f}')

    md = '\n'.join(lines) + '\n'
    out = os.path.join(ROOT, 'docs', 'analysis', 'phaseB_weighted_results.md')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(md)
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    print(md)
    print(f'Wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
