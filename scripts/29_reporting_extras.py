from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from rgcn_model import build_model

T = ['TE18', 'TL12', 'JL10', 'RB9']
METRICS = ['MCC', 'AUC', 'AUPRC', 'F1', 'Precision', 'Sensitivity', 'Specificity', 'GMean']
OUT = os.path.join(ROOT, 'results', 'reporting_extras.json')


def bh(pvals):
    
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        k = n - rank + 1
        prev = min(prev, p[idx] * n / k)
        q[idx] = prev
    return q


def multiplicity():
    v = json.load(open(os.path.join(ROOT, 'results', 'per_set_metrics.json'), encoding='utf-8'))
    sig = v['significance']
    fam = {}
    keys, pv = [], []
    for m in METRICS:
        for sp in T + ['average']:
            keys.append(f'{m}/{sp}')
            pv.append(sig[m][sp]['p'])
    q = bh(pv)
    fam['all'] = {k: {'p': float(p), 'q': float(qq), 'survives': bool(qq < 0.05)}
                  for k, p, qq in zip(keys, pv, q)}
    keys2, pv2 = [], []
    for m in METRICS:
        keys2.append(m)
        pv2.append(sig[m]['average']['p'])
    q2 = bh(pv2)
    fam['average_only'] = {k: {'p': float(p), 'q': float(qq), 'survives': bool(qq < 0.05)}
                           for k, p, qq in zip(keys2, pv2, q2)}
    return fam


def training_times():
    
    import glob
    import re
    rows = []
    for f in glob.glob(os.path.join(ROOT, 'logs', 'strict', '*.log')) + \
             glob.glob(os.path.join(ROOT, 'logs', 'controlled_b', '*.log')):
        txt = open(f, encoding='utf-8', errors='ignore').read()
        m = re.search(r'\((\d{2}):(\d{2}):(\d{2})\)\s*$', txt, re.M)
        if not m:
            continue
        tag = os.path.basename(f).split('_seed')[0]
        rows.append({'tag': tag, 'end': f'{m.group(1)}:{m.group(2)}:{m.group(3)}',
                     'file': os.path.basename(f)})
    return rows


def inference_cost():
    
    torch.set_num_threads(1)
    gobj_ss = torch.load(os.path.join(ROOT, 'data', 'processed', 'graphs_seq_ss_clean.pt'),
                         map_location='cpu', weights_only=False)
    gobj_dual = torch.load(os.path.join(ROOT, 'data', 'processed', 'graphs_dual_clean.pt'),
                           map_location='cpu', weights_only=False)
    out = {}
    for name, tag, gf, gobj in [('graph', 'clean_seq_ss', 'graphs_seq_ss_clean.pt', gobj_ss),
                                ('tcn', 'slim_tcn', 'graphs_dual_clean.pt', gobj_dual),
                                ('dense', 'slim_dense', 'graphs_dual_clean.pt', gobj_dual)]:


        chains = gobj['graphs']['TE18'][:6]
        ck = torch.load(os.path.join(ROOT, 'checkpoints', 'controlled', f'{tag}_seed42.pt'),
                        map_location='cpu', weights_only=False)
        cfg = dict(ck['config']['model'])
        cfg['num_relations'] = ck['num_relations']
        cfg['in_dim'] = gobj['in_dim']
        model = build_model(cfg, ck['num_relations'])
        model.load_state_dict(ck['state_dict'])
        model.eval()
        n_res = sum(g.x.size(0) for g in chains)
        with torch.no_grad():
            for g in chains[:1]:
                z = model(g.x, g.edge_index, g.edge_type, torch.zeros(g.x.size(0), dtype=torch.long))
            t0 = time.time()
            for g in chains:
                z = model(g.x, g.edge_index, g.edge_type, torch.zeros(g.x.size(0), dtype=torch.long))
            dt = time.time() - t0
        out[name] = {'seconds_per_chain': dt / len(chains),
                     'seconds_per_1k_residues': dt / n_res * 1000}
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    print('[a] Benjamini-Hochberg correction')
    mult = multiplicity()
    n_surv = sum(1 for v in mult['all'].values() if v['survives'])
    print(f'  family of {len(mult["all"])} tests: {n_surv} survive at q < 0.05')
    not_surv = [k for k, v in mult['all'].items() if not v['survives']]
    print('  do not survive:', not_surv)
    print(f'  metric-average family: '
          f'{sum(1 for v in mult["average_only"].values() if v["survives"])}/{len(mult["average_only"])}'
          ' survive')
    print('  non-survivors:', [k for k, v in mult['average_only'].items() if not v['survives']])

    print('\n[b] inference cost (CPU, 6 TE18 chains)')
    cost = inference_cost()
    for k, v in cost.items():
        print(f'  {k:<6} {v["seconds_per_chain"]*1000:.1f} ms/chain  '
              f'{v["seconds_per_1k_residues"]*1000:.1f} ms/1k residues')

    print('\n[c] training-time logs found:', len(training_times()))
    gpu = 'unknown'
    try:
        import subprocess
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                              '--format=csv,noheader'], capture_output=True, text=True,
                             timeout=20).stdout.strip() or 'unknown'
    except Exception:
        pass
    print('  GPU:', gpu)

    json.dump({'multiplicity': mult, 'inference_cost_cpu': cost, 'gpu': gpu,
               'training_logs': training_times()},
              open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n[written] {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
