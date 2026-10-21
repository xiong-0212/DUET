from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
T = ['TE18', 'TL12', 'JL10', 'RB9']
ARMS = [
    ('clean_seq_ss', 'Graph tower alone (R-GCN, RNA-FM)'),
    ('clean_dual2_mlp', 'MLP tower alone (RiNALMo, TCL-Focal from this pipeline)'),
    ('clean_dual2', '**PRISM-Ens: gated fusion (graph tower + MLP tower)**'),
    ('clean_dual2_avg', 'Equal-weight fusion (gating control)'),
    ('clean_dual2_logit', 'Logit averaging (single-model variant of probability ensembling)'),
]


def arm(tag):
    out = {}
    for f in glob.glob(os.path.join(ROOT, 'results', 'controlled',
                                    f'train_{tag}_seed*.json')):
        seed = int(os.path.basename(f)[:-5].rpartition('_seed')[2])
        r = json.load(open(f, encoding='utf-8'))
        tm = r.get('test_metrics', {})
        if all(s in tm for s in T):
            out[seed] = dict(
                v=float(np.mean([tm[s]['at_threshold']['MCC'] for s in T])),
                f=float(np.mean([tm[s]['at_0.5']['MCC'] for s in T])),
                per=[tm[s]['at_threshold']['MCC'] for s in T],
                auc=float(np.mean([tm[s]['at_threshold']['AUC'] for s in T])))
    return out


def main():
    data = {tag: arm(tag) for tag, _ in ARMS}
    ref = data['clean_seq_ss']
    lines = ['# Dual-backbone fusion (PRISM-Ens) results (2026-10-11)', '',
             '> Data: `results/controlled/train_clean_dual2*_seed*.json` (5 seeds, cleaned training set)',
             '> Protocol: per-seed validation-selected threshold plus fixed 0.5; graph = seq+ss; node features = RNA-FM(640)|RiNALMo(1280)', '']
    lines += ['| Model | n | val-selected threshold | fixed 0.5 | mean AUC | TE18 | TL12 | JL10 | RB9 |',
              '|:---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for tag, label in ARMS:
        d = data.get(tag, {})
        if not d:
            continue
        v = [x['v'] for x in d.values()]
        f = [x['f'] for x in d.values()]
        a = [x['auc'] for x in d.values()]
        per = np.mean([x['per'] for x in d.values()], axis=0)
        lines.append(f'| {label} | {len(d)} | **{np.mean(v):.4f} ± {np.std(v, ddof=1):.3f}** | '
                     f'{np.mean(f):.4f} | {np.mean(a):.3f} | '
                     + ' | '.join(f'{x:.3f}' for x in per) + ' |')
    lines += ['', '## Paired tests (same seeds, n=5, val-threshold protocol)', '',
              '| Comparison | Δ | p | Same-direction seeds |', '|:---|---:|---:|---:|']
    for tag, label in ARMS[1:]:
        d = data.get(tag, {})
        c = sorted(set(d) & set(ref))
        if len(c) < 3:
            continue
        x = np.array([d[s]['v'] for s in c])
        y = np.array([ref[s]['v'] for s in c])
        dd = x - y
        lines.append(f'| {label} − graph tower | {dd.mean():+.4f} | '
                     f'{float(stats.ttest_rel(x, y).pvalue):.3f} | '
                     f'{int((dd > 0).sum())}/{len(c)} |')
    md = '\n'.join(lines) + '\n'
    out = os.path.join(ROOT, 'docs', 'analysis', 'dual_backbone_fusion_results.md')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(md)
    print(md)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
