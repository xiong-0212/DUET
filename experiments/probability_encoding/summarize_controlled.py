from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
T = ['TE18', 'TL12', 'JL10', 'RB9']

MODEL_LABEL = {
    'mlp': 'MLP only (CoBRA-M2 equivalent)',
    'seq_ss': 'R-GCN seq+ss',
    'seq_ss_alt': 'R-GCN seq+ss+ss_alt (PRISM main model)',
    'seq_ss_sim': 'R-GCN seq+ss+sim',
    'seq_ss_alt_sim': 'R-GCN seq+ss+ss_alt+sim',
}
DATASET_LABEL = {'clean': 'De-leaked (train=314, val=40)', 'orig': 'Original (train=345, val=40)'}


def collect():
    rows = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(ROOT, 'results', 'controlled', 'train_*.json'))):
        base = os.path.basename(f)[len('train_'):-len('.json')]
        if base.startswith('smoke'):
            continue
        model, _, seed = base.rpartition('_seed')
        ds = model.split('_', 1)[0]
        m = model.split('_', 1)[1]
        r = json.load(open(f, encoding="utf-8"))
        tm = r.get('test_metrics', {})
        vals = [tm[s]['at_threshold']['MCC'] for s in T if s in tm]
        aucs = [tm[s]['at_threshold']['AUC'] for s in T if s in tm]
        if len(vals) < 4:
            continue
        rows[(ds, m)]['mcc'].append(float(np.mean(vals)))
        rows[(ds, m)]['auc'].append(float(np.mean(aucs)))
        for s, v in zip([s for s in T if s in tm], vals):
            rows[(ds, m)][s].append(v)
    return rows


def fmt(vals):
    if not vals:
        return '—'
    if len(vals) == 1:
        return f'{vals[0]:.3f}'
    return f'{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}'


def main():
    rows = collect()
    lines = []
    csv = ['dataset,model,n,avg_mcc,std_mcc,avg_auc,' + ','.join(T)]
    for ds in ['clean', 'orig']:
        keys = [k for k in rows if k[0] == ds]
        if not keys:
            continue
        lines.append(f'\n### {DATASET_LABEL.get(ds, ds)}\n')
        lines.append('| Model | n | Mean MCC | Mean AUC | ' + ' | '.join(T) + ' |')
        lines.append('|:---|:---|---:|---:|' + '---:|' * 4)

        ordered = sorted(keys, key=lambda k: -np.mean(rows[k]['mcc']))
        for k in ordered:
            d = rows[k]
            n = len(d['mcc'])
            lines.append('| %s | %d | %s | %s | %s |' % (
                MODEL_LABEL.get(k[1], k[1]), n,
                fmt(d['mcc']), fmt(d['auc']),
                ' | '.join(fmt(d[s]) for s in T)))
            csv.append('%s,%s,%d,%.4f,%.4f,%.4f,%s' % (
                ds, k[1], n, np.mean(d['mcc']),
                np.std(d['mcc'], ddof=1) if n > 1 else 0.0, np.mean(d['auc']),
                ','.join('%.4f' % np.mean(d[s]) if d[s] else '' for s in T)))
    md = '# Controlled grid results (auto-generated, do not edit by hand)\n' + '\n'.join(lines) + '\n'
    out_md = os.path.join(ROOT, 'docs', 'records', 'controlled_results.md')
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, 'w', encoding='utf-8').write(md)
    open(os.path.join(ROOT, 'results', 'controlled', 'summary.csv'), 'w', encoding='utf-8').write('\n'.join(csv))
    print(md)
    print(f'Wrote {out_md}')


if __name__ == '__main__':
    main()
