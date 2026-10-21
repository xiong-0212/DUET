from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from rgcn_model import build_model

T = ['TE18', 'TL12', 'JL10', 'RB9']
SEEDS = [42, 0, 1, 2, 3]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
M = ['MCC', 'AUC', 'AUPRC', 'F1', 'Precision', 'Recall', 'GMean', 'Accuracy']


def gm(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    se = tp / (tp + fn) if (tp + fn) else 0
    sp = tn / (tn + fp) if (tn + fp) else 0
    return float(np.sqrt(se * sp))


def met(y, p, thr):
    pr = (p >= thr).astype(int)
    return {'MCC': matthews_corrcoef(y, pr), 'AUC': roc_auc_score_(y, p),
            'AUPRC': average_precision_score(y, p), 'F1': f1_score(y, pr, zero_division=0),
            'Precision': precision_score(y, pr, zero_division=0),
            'Recall': recall_score(y, pr, zero_division=0), 'GMean': gm(y, pr),
            'Accuracy': accuracy_score(y, pr)}


def roc_auc_score_(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_temp(p, y):
    z = logit(p)

    def nll(logT):
        zz = z / np.exp(logT)
        return float(np.mean(np.logaddexp(0, zz) - y * zz))
    return float(np.exp(minimize_scalar(nll, bounds=(-2, 3), method='bounded').x))


def per_chain(tag, gf):
    gobj = torch.load(os.path.join(ROOT, 'data', 'processed', gf),
                      map_location='cpu', weights_only=False)
    rmv = set(json.load(open(os.path.join(ROOT, 'data', 'leaked_pairs.json'),
                             encoding='utf-8'))['remove_val_ids'])
    sp = {'val': [g for g in gobj['graphs']['val'] if g.rna_id not in rmv]}
    for s in T:
        sp[s] = gobj['graphs'][s]
    acc = {}
    for seed in SEEDS:
        ck = torch.load(os.path.join(ROOT, 'checkpoints', 'controlled',
                                     f'{tag}_seed{seed}.pt'),
                        map_location='cpu', weights_only=False)
        mc = dict(ck['config']['model'])
        mc['num_relations'] = ck['num_relations']
        mc['in_dim'] = gobj['in_dim']
        m = build_model(mc, ck['num_relations']).to(DEV)
        m.load_state_dict(ck['state_dict'])
        m.eval()
        with torch.no_grad():
            for s, gs in sp.items():
                for g in gs:
                    z = m(g.x.to(DEV), g.edge_index.to(DEV), g.edge_type.to(DEV),
                          torch.zeros(g.x.size(0), dtype=torch.long, device=DEV))
                    if isinstance(z, tuple):
                        z = z[0]
                    k = f'{s}/{g.rna_id}'
                    acc.setdefault(k, {'p': [], 'y': g.y.numpy().astype(int)})
                    acc[k]['p'].append(torch.sigmoid(z).float().cpu().numpy())
    return {k: {'p': np.mean(v['p'], axis=0), 'y': v['y']} for k, v in acc.items()}


def main():
    d = json.load(open(os.path.join(ROOT, 'results', 'ensemble_member_probs.json'),
                       encoding='utf-8'))
    G = {}
    for s in ['val'] + T:
        for rid in d['seq_ss']['splits'][s]:
            G[f'{s}/{rid}'] = {'p': np.asarray(d['seq_ss']['splits'][s][rid]['prob']),
                               'y': np.asarray(d['seq_ss']['splits'][s][rid]['label'])}
    S = per_chain('clean_dual2_mlp', 'graphs_dual_clean.pt')

    def cat(A, B, s):
        ks = sorted(k for k in A if k.startswith(s + '/'))
        return (np.concatenate([A[k]['y'] for k in ks]),
                np.concatenate([A[k]['p'] for k in ks]),
                np.concatenate([B[k]['p'] for k in ks]))

    vy, vg, vs = cat(G, S, 'val')
    T1, T2 = fit_temp(vg, vy), fit_temp(vs, vy)
    grid = np.linspace(0, 1, 21)
    w_best = float(grid[int(np.argmax([
        matthews_corrcoef(vy, ((w * vg + (1 - w) * vs) >= 0.5).astype(int)) for w in grid]))])
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(np.c_[vg, vs], vy)

    def rank_avg(a, b):
        def rk(x):
            return np.argsort(np.argsort(x)).astype(float) / max(len(x) - 1, 1)
        return 0.5 * (rk(a) + rk(b))

    METHODS = [
        ('Equal-weight averaging (adopted)', lambda a, b: 0.5 * (a + b)),
        ('Temperature-calibrated averaging',
         lambda a, b: 0.5 * (1 / (1 + np.exp(-logit(a) / T1)) + 1 / (1 + np.exp(-logit(b) / T2)))),
        (f'Validation-weighted averaging (w = {w_best:.2f})',
         lambda a, b: w_best * a + (1 - w_best) * b),
        ('Logit averaging', lambda a, b: 1 / (1 + np.exp(-0.5 * (logit(a) + logit(b))))),
        ('Rank averaging', rank_avg),
        ('Validation-stacked logistic combination',
         lambda a, b: lr.predict_proba(np.c_[a, b])[:, 1]),
    ]

    th_grid = np.linspace(0.05, 0.95, 91)
    lines = ['### Table S2 | Comparison of ensemble combination strategies', '',
             'Both members (graph tower and sequence tower) are fixed; only the way their '
             'per-nucleotide probabilities are combined differs. Values are means over the four '
             'benchmark sets (five-seed probability averaging). Validation-derived parameters '
             '(temperatures, weight, stack coefficients) were fitted on the validation split only.',
             '']
    for mode, lab in [('fixed', 'At a fixed threshold of 0.5'), ('val', 'At each strategy’s validation-selected threshold')]:
        lines += [f'**{lab}**', '',
                  '| Combination strategy | ' + ' | '.join(M) + ' |',
                  '|:---|' + '---:|' * len(M)]
        for name, fn in METHODS:
            if mode == 'val':
                vp = fn(vg, vs)
                thr = float(th_grid[int(np.argmax([matthews_corrcoef(vy, (vp >= t).astype(int))
                                                   for t in th_grid]))])
            else:
                thr = 0.5
            vals = {m: [] for m in M}
            for s in T:
                y, p1, p2 = cat(G, S, s)
                pm = fn(p1, p2)
                r = met(y, pm, thr)
                for m in M:
                    vals[m].append(r[m])
            lines.append(f'| {name} | ' + ' | '.join(f'{np.mean(vals[m]):.3f}' for m in M) + ' |')
        lines.append('')
    md = '\n'.join(lines) + '\n'
    out = os.path.join(ROOT, 'docs', 'tables', 'paper_tables_S2.md')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, 'w', encoding='utf-8').write(md)
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    print(md)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
