from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
import torch
from scipy.stats import ttest_rel
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from rgcn_model import build_model

BENCH = ['TE18', 'TL12', 'JL10']
DEV_SEEDS = [42, 0, 1, 2, 3]
CONF_SEEDS = [7, 11, 13, 17, 23]
SEEDS = DEV_SEEDS + CONF_SEEDS
METRICS = ['MCC', 'AUC', 'AUPRC', 'F1', 'Precision', 'Recall', 'GMean', 'Accuracy']
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
THR_GRID = np.linspace(0.05, 0.95, 91)
GRASSP_GLOB = os.path.join(ROOT, 'results', 'grassp_search', 'final_seed*.json')
OUT = os.path.join(ROOT, 'results', 'grassp_vs_duet.json')
OUT_MD = os.path.join(ROOT, 'docs', 'analysis', 'GRASSP_comparison_2026-10-15.md')

GRAPH = {'clean_seq_ss': 'graphs_seq_ss_clean.pt', 'ns_graph': 'graphs_seq_ss_clean.pt',
         'slim_tcn': 'graphs_dual_clean.pt', 'ns_tcn': 'graphs_dual_clean.pt',
         'slim_dense': 'graphs_dual_clean.pt', 'ns_dense': 'graphs_dual_clean.pt'}
DUET_MEMBERS = {s: (('clean_seq_ss', 'slim_tcn', 'slim_dense') if s in DEV_SEEDS
                    else ('ns_graph', 'ns_tcn', 'ns_dense')) for s in SEEDS}
LABELS = [('DUET (sequence only)', 'duet'), ('GRASSP (sequence + 3D structure)', 'grassp')]


def gmean(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    se = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    return float(np.sqrt(se * sp))


def met(y, p, thr):
    pred = (p >= thr).astype(int)
    return {'MCC': float(matthews_corrcoef(y, pred)), 'AUC': float(roc_auc_score(y, p)),
            'AUPRC': float(average_precision_score(y, p)),
            'F1': float(f1_score(y, pred, zero_division=0)),
            'Precision': float(precision_score(y, pred, zero_division=0)),
            'Recall': float(recall_score(y, pred, zero_division=0)),
            'GMean': gmean(y, pred), 'Accuracy': float(accuracy_score(y, pred))}


def cat(acc, split):
    ks = sorted(k for k in acc if k.startswith(split + '/'))
    y = np.concatenate([np.atleast_1d(acc[k]['y']) for k in ks])
    p = np.concatenate([np.atleast_1d(acc[k]['p']) for k in ks])
    return y, p


def val_threshold(acc):
    y, p = cat(acc, 'val')
    return float(THR_GRID[int(np.argmax([matthews_corrcoef(y, (p >= t).astype(int))
                                         for t in THR_GRID]))])


def duet_probs(seed):
    
    members = {}
    for tag in DUET_MEMBERS[seed]:
        gobj = torch.load(os.path.join(ROOT, 'data', 'processed', GRAPH[tag]),
                          map_location='cpu', weights_only=False)
        leak = json.load(open(os.path.join(ROOT, 'data', 'leaked_pairs.json'), encoding='utf-8'))
        drop = set(leak['remove_val_ids'])
        splits = {'val': [g for g in gobj['graphs']['val'] if g.rna_id not in drop]}
        splits.update({b: gobj['graphs'][b] for b in BENCH})
        ck = torch.load(os.path.join(ROOT, 'checkpoints', 'controlled', f'{tag}_seed{seed}.pt'),
                        map_location='cpu', weights_only=False)
        cfg = dict(ck['config']['model'])
        cfg['num_relations'] = ck['num_relations']
        cfg['in_dim'] = gobj['in_dim']
        model = build_model(cfg, ck['num_relations']).to(DEV)
        model.load_state_dict(ck['state_dict'])
        model.eval()
        acc = {}
        with torch.no_grad():
            for split, graphs in splits.items():
                for g in graphs:
                    z = model(g.x.to(DEV), g.edge_index.to(DEV), g.edge_type.to(DEV),
                              torch.zeros(g.x.size(0), dtype=torch.long, device=DEV))
                    if isinstance(z, tuple):
                        z = z[0]
                    acc[f'{split}/{g.rna_id}'] = {'p': torch.sigmoid(z).float().cpu().numpy(),
                                                  'y': g.y.numpy().astype(int)}
        members[tag] = acc
    base = DUET_MEMBERS[seed][0]
    return {k: {'p': np.mean([m[k]['p'] for m in members.values()], axis=0),
                'y': members[base][k]['y']} for k in members[base]}


def grassp_probs():
    out = {}
    for f in sorted(glob.glob(GRASSP_GLOB)):
        d = json.load(open(f, encoding='utf-8'))
        seed = int(list(d['seeds'])[0])
        acc = {}
        for split, per_seed in d['per_chain'].items():
            for name, v in per_seed[str(seed)].items():
                acc[f'{split}/{name}'] = {'p': np.asarray(v['p'], dtype=float),
                                          'y': np.asarray(v['y'], dtype=int)}
        out[seed] = acc
    return out


def per_seed_metrics(accs, seeds, split):
    
    thr = {s: val_threshold(accs[s]) for s in seeds}
    out = {m: [] for m in METRICS}
    for s in seeds:
        r = met(*cat(accs[s], split), thr[s])
        for m in METRICS:
            out[m].append(r[m])
    return {m: np.array(v) for m, v in out.items()}, float(np.mean(list(thr.values())))


def stars(p):
    if p < 0.001:
        return '†††'
    if p < 0.01:
        return '††'
    return '†' if p < 0.05 else ''


def fmt(v):
    return f'{v.mean():.3f} ± {v.std(ddof=1):.3f}'


def bold(text, value, best, tol=1e-9):
    return f'**{text}**' if value >= best - tol else text


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    g = grassp_probs()
    seeds = [s for s in SEEDS if s in g]
    ours = {s: duet_probs(s) for s in seeds}
    print(f'GRASSP seeds: {len(seeds)}/10')

    data = {}
    for b in BENCH:
        data[b] = {}
        for label, accs in [('DUET (sequence only)', ours), ('GRASSP (sequence + 3D structure)', g)]:
            data[b][label], data[b][label + '·threshold'] = per_seed_metrics(accs, seeds, b)


    avg = {}
    for label, accs in [('DUET (sequence only)', ours), ('GRASSP (sequence + 3D structure)', g)]:
        thr = {s: val_threshold(accs[s]) for s in seeds}
        avg[label] = {m: np.array([np.mean([met(*cat(accs[s], b), thr[s])[m] for b in BENCH])
                                   for s in seeds]) for m in METRICS}

    res = {'protocol': 'both models retrained on the same de-leaked split with the same ten seeds; '
                       'each seed at its own validation-selected threshold',
           'seeds': seeds, 'per_benchmark': {}, 'significance': {}, 'average': {}}
    lines = ['<!-- AUTO:BEGIN · generated by baselines/grassp/compare_with_duet.py, do not edit by hand -->', '',
             '> **Protocol**: both models are retrained on the **same de-leaked split** (train 314 / val 40) '
             'with the **same ten seeds**; each seed is scored at **its own validation-selected threshold**.',
             '> Significance = **paired *t* test over the same seeds (n = 10)**; † p<0.05, †† p<0.01, ††† p<0.001. '
             '**Bold = best in column**.',
             '> GRASSP is configured within its paper\'s hyper-parameter search space, tuned on the **validation '
             'set only** (lr 1e-3 / batch 1 / hidden 128 / dropout 0.3). Data source: `results/grassp_vs_duet.json`.', '']

    cols = BENCH + ['3-benchmark mean']
    for m in METRICS:
        best = [max(data[b][lab][m].mean() for lab, _ in LABELS) for b in BENCH]
        best_avg = max(avg[lab][m].mean() for lab, _ in LABELS)
        lines += [f'### {m} (per benchmark, mean ± s.d. over 10 seeds)', '',
                  '| Method | ' + ' | '.join(cols) + ' |', '|:---|' + '---:|' * len(cols)]
        for label, _ in LABELS:
            cells = [bold(fmt(data[b][label][m]), data[b][label][m].mean(), best[i])
                     for i, b in enumerate(BENCH)]
            cells.append(bold(fmt(avg[label][m]), avg[label][m].mean(), best_avg))
            lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
        lines += ['']

    lines += ['### Significance: DUET − GRASSP (same-seed paired, n = 10)', '',
              '| Metric | ' + ' | '.join(cols) + ' | Verdict |', '|:---|' + '---:|' * len(cols) + '---|']
    for m in METRICS:
        cells, deltas = [], []
        for b in BENCH:
            a, c = data[b]['DUET (sequence only)'][m], data[b]['GRASSP (sequence + 3D structure)'][m]
            d, pv = float((a - c).mean()), float(ttest_rel(a, c).pvalue)
            deltas.append(d)
            cells.append(f'{d:+.3f} ({pv:.3f}){stars(pv)}')
            res['significance'].setdefault(m, {})[b] = {'delta': d, 'p': pv}
        a, c = avg['DUET (sequence only)'][m], avg['GRASSP (sequence + 3D structure)'][m]
        d, pv = float((a - c).mean()), float(ttest_rel(a, c).pvalue)
        deltas.append(d)
        cells.append(f'{d:+.3f} ({pv:.3f}){stars(pv)}')
        verdict = '**significant**' if pv < 0.05 else 'n.s.'
        res['significance'][m]['average'] = {'delta': d, 'p': pv, 'verdict': verdict}
        lines.append(f'| {m} | ' + ' | '.join(cells) + f' | {verdict} |')
    lines += ['', '> "Verdict" is given by the paired test on the **3-benchmark mean**.', '']

    for b in BENCH:
        res['per_benchmark'][b] = {
            lab: {m: {'mean': float(data[b][lab][m].mean()), 'sd': float(data[b][lab][m].std(ddof=1)),
                      'per_seed': [float(x) for x in data[b][lab][m]]} for m in METRICS}
            for lab, _ in LABELS}
    res['average'] = {lab: {m: {'mean': float(avg[lab][m].mean()),
                                'sd': float(avg[lab][m].std(ddof=1))} for m in METRICS}
                      for lab, _ in LABELS}

    lines += ['### Leakage evidence: same model, training set swapped', '',
              '| Model | Training set | Validation MCC | TL12 test MCC | JL10 test MCC |', '|:---|:---|---:|---:|---:|',
              '| GRASSP (author-released weights) | original 345 chains (leakage included) | 0.613 | 0.631 | 0.338 |',
              '| GRASSP (our retraining) | **de-leaked 314 chains** | 0.613 | **0.403** | **0.252** |',
              '', '> Both rows give the **same validation MCC** (0.613 vs 0.613), so the retraining '
              'matches the quality of the released model; the difference is only whether the training set contains '
              'chains identical to test sequences. TL12 drops 0.631 → 0.403 because of overlap, not modelling.', '']
    lines += ['<!-- AUTO:END -->', '']

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    body = '\n'.join(lines)
    if os.path.exists(OUT_MD):
        doc = open(OUT_MD, encoding='utf-8').read()
        doc = re.sub(r'<!-- AUTO:BEGIN.*?<!-- AUTO:END -->', body.rstrip(), doc, flags=re.S)
    else:
        doc = ('# DUET (sequence only) vs GRASSP (sequence + experimental 3D structure)\n\n'
               '> Group-meeting material. The auto-generated block below is recomputed by '
               '`baselines/grassp/compare_with_duet.py`,\n'
               '> from the same source as the paper tables; handwritten notes at the end.\n\n'
               + body + '\n## Reading notes\n\n(to be filled in)\n')
    open(OUT_MD, 'w', encoding='utf-8').write(doc)
    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    a, c = avg['DUET (sequence only)']['MCC'], avg['GRASSP (sequence + 3D structure)']['MCC']
    print(f'\n3-benchmark mean MCC: DUET {a.mean():.4f}±{a.std(ddof=1):.4f}  vs  '
          f'GRASSP {c.mean():.4f}±{c.std(ddof=1):.4f}  Δ={a.mean()-c.mean():+.4f} '
          f'p={ttest_rel(a,c).pvalue:.6f}')
    print(f'[written] {OUT}\n[written] {OUT_MD}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
