from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import torch
from scipy.stats import ttest_rel
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE

while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from rgcn_model import build_model

T = ['TE18', 'TL12', 'JL10', 'RB9']
DEV_SEEDS = [42, 0, 1, 2, 3]
CONF_SEEDS = [7, 11, 13, 17, 23]
SEEDS = DEV_SEEDS + CONF_SEEDS
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
THR_GRID = np.linspace(0.05, 0.95, 91)
METRICS = ['MCC', 'AUC', 'AUPRC', 'F1', 'Precision', 'Sensitivity', 'Specificity', 'GMean']
OUT_JSON = os.path.join(ROOT, 'results', 'per_set_metrics.json')
OUT_MD = os.path.join(ROOT, 'docs', 'analysis', 'per_testset_comparison_2026-10-12.md')


GRAPH = {'clean_seq_ss': 'graphs_seq_ss_clean.pt', 'ns_graph': 'graphs_seq_ss_clean.pt',
         'slim_tcn': 'graphs_dual_clean.pt', 'ns_tcn': 'graphs_dual_clean.pt',
         'slim_dense': 'graphs_dual_clean.pt', 'ns_dense': 'graphs_dual_clean.pt'}
DUET_MEMBERS = {s: (('clean_seq_ss', 'slim_tcn', 'slim_dense') if s in DEV_SEEDS
                    else ('ns_graph', 'ns_tcn', 'ns_dense')) for s in SEEDS}
BASELINE_FILES = {
    **{s: 'cobra_m3_equiv.json' for s in DEV_SEEDS},
    **{s: 'cobra_m3_equiv_newseeds.json' for s in CONF_SEEDS},
}


def gmean(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    se = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    return float(np.sqrt(se * sp))


def specificity(y, pred):
    tn, fp, _, _ = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def met(y, p, thr):
    pred = (p >= thr).astype(int)
    return {'MCC': float(matthews_corrcoef(y, pred)), 'AUC': float(roc_auc_score(y, p)),
            'AUPRC': float(average_precision_score(y, p)),
            'F1': float(f1_score(y, pred, zero_division=0)),
            'Precision': float(precision_score(y, pred, zero_division=0)),
            'Sensitivity': float(recall_score(y, pred, zero_division=0)),
            'Specificity': specificity(y, pred), 'GMean': gmean(y, pred)}


def pack(tag, seed):
    
    gobj = torch.load(os.path.join(ROOT, 'data', 'processed', GRAPH[tag]),
                      map_location='cpu', weights_only=False)
    leak = json.load(open(os.path.join(ROOT, 'data', 'leaked_pairs.json'), encoding='utf-8'))
    drop = set(leak['remove_val_ids'])
    splits = {'val': [g for g in gobj['graphs']['val'] if g.rna_id not in drop]}
    splits.update({s: gobj['graphs'][s] for s in T})
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
    return {'acc': acc, 'thr': float(ck['threshold'])}


def duet_pack(seed):
    members = [pack(tag, seed) for tag in DUET_MEMBERS[seed]]
    acc = {k: {'p': np.mean([m['acc'][k]['p'] for m in members], axis=0),
               'y': members[0]['acc'][k]['y']} for k in members[0]['acc']}
    ks = sorted(k for k in acc if k.startswith('val/'))
    y = np.concatenate([acc[k]['y'] for k in ks])
    p = np.concatenate([acc[k]['p'] for k in ks])
    return {'acc': acc, 'thr': float(THR_GRID[int(np.argmax(
        [matthews_corrcoef(y, (p >= t).astype(int)) for t in THR_GRID]))])}


def baseline_packs():
    out = {}
    cache = {}
    for seed, fname in BASELINE_FILES.items():
        if fname not in cache:
            cache[fname] = json.load(open(os.path.join(ROOT, 'results', fname), encoding='utf-8'))
        rows = cache[fname]['clean']['bce']
        row = next(r for r in rows if r['seed'] == seed)
        acc = {}
        for split, chains in row['per_chain'].items():
            for rid, v in chains.items():
                acc[f'{split}/{rid}'] = {'p': np.asarray(v['prob'], dtype=float),
                                         'y': np.asarray(v['label'], dtype=int)}
        out[seed] = {'acc': acc, 'thr': float(row['threshold'])}
    return out


def cat(pack_, split):
    ks = sorted(k for k in pack_['acc'] if k.startswith(split + '/'))
    y = np.concatenate([pack_['acc'][k]['y'] for k in ks])
    p = np.concatenate([pack_['acc'][k]['p'] for k in ks])
    return y, p


def value(pack_, split, metric):
    return met(*cat(pack_, split), pack_['thr'])[metric]


def bold(text, value_, best, tol=1e-9):
    return f'**{text}**' if value_ >= best - tol else text


def fmt(v):
    return f'{v.mean():.3f} ± {v.std(ddof=1):.3f}'


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    duet = {s: duet_pack(s) for s in SEEDS}
    base = baseline_packs()

    def series(pk, seeds, split, metric):
        return np.array([value(pk[s], split, metric) for s in seeds])

    methods = [('CoBRA-M3 equivalent (retrained baseline)', base),
               ('DUET (three-tower ensemble)', duet)]
    res = {'protocol': {'threshold': 'per seed, selected on the validation split',
                        'seeds': SEEDS, 'development_seeds': DEV_SEEDS,
                        'confirmation_seeds': CONF_SEEDS, 'test': 'same-seed paired t-test'},
           'per_set': {}, 'significance': {}, 'replication': {}}

    lines = ['<!-- AUTO:BEGIN · generated by scripts/17_per_set_metrics.py, do not edit by hand -->', '',
             '> **Protocol**: each seed is trained independently and evaluated at **its own '
             'validation threshold**; DUET = **equal-weight average** of the predicted '
             'probabilities of the three towers (graph tower + TCN + DenseNet).',
             '> **10 seeds** in total: 42/0/1/2/3 form the development batch, 7/11/13/17/23 the '
             '**independent confirmation batch** (Section 5 reports the agreement of the two).',
             '> Significance = **same-seed paired *t* test (n = 10)**; † p < 0.05, †† p < 0.01, '
             '††† p < 0.001. The best value of the two methods is bold.',
             '> Test sets are deduplicated (RB9 = 6 chains); training/validation sets are '
             'de-leaked (314 / 40 chains). Source: `results/per_set_metrics.json`.', '']


    for metric in METRICS:
        vals = {name: [series(pk, SEEDS, sp, metric) for sp in T] + [series(pk, SEEDS, '', metric)
                if False else np.array([np.mean([value(pk[s], sp, metric) for sp in T])
                                        for s in SEEDS])] for name, pk in methods}
        best = [max(vals[n][i].mean() for n in vals) for i in range(len(T) + 1)]
        lines += [f'### {metric} (per test set, mean ± s.d. over 10 seeds; **bold = best in column**)', '',
                  '| Method | ' + ' | '.join(T) + ' | Four-set average |', '|:---|' + '---:|' * 5]
        for name, _ in methods:
            cells = [bold(fmt(vals[name][i]), vals[name][i].mean(), best[i])
                     for i in range(len(T))]
            avg_txt = bold(fmt(vals[name][-1]), vals[name][-1].mean(), best[-1])
            res['per_set'].setdefault(name, {})[metric] = {
                sp: {'mean': float(vals[name][i].mean()), 'sd': float(vals[name][i].std(ddof=1))}
                for i, sp in enumerate(T)}
            res['per_set'][name][metric]['average'] = {
                'mean': float(vals[name][-1].mean()), 'sd': float(vals[name][-1].std(ddof=1))}
            lines.append(f'| {name} | ' + ' | '.join(cells) + f' | {avg_txt} |')
        lines += ['']


    lines += ['### Significance: DUET vs baseline (same-seed paired, n = 10; '
              '**bold = largest Δ in column**)', '',
              '| Metric | ' + ' | '.join(T) + ' | Four-set average | Verdict |',
              '|:---|' + '---:|' * 5 + '---|']
    deltas, texts, avg_deltas = {}, {}, {}
    for metric in METRICS:
        deltas[metric], texts[metric] = [], []
        for sp in T:
            a = series(duet, SEEDS, sp, metric)
            b = series(base, SEEDS, sp, metric)
            d, pv = float((a - b).mean()), float(ttest_rel(a, b).pvalue)
            deltas[metric].append(d)
            texts[metric].append(f'{d:+.3f} ({pv:.3f}){stars(pv)}')
            res['significance'].setdefault(metric, {})[sp] = {'delta': d, 'p': pv}
        a = np.array([np.mean([value(duet[s], sp, metric) for sp in T]) for s in SEEDS])
        b = np.array([np.mean([value(base[s], sp, metric) for sp in T]) for s in SEEDS])
        d, pv = float((a - b).mean()), float(ttest_rel(a, b).pvalue)
        deltas[metric].append(d)
        texts[metric].append(f'{d:+.3f} ({pv:.3f}){stars(pv)}')
        avg_deltas[metric] = pv
        res['significance'][metric]['average'] = {'delta': d, 'p': pv}
    best = [max(deltas[m][i] for m in METRICS) for i in range(len(T) + 1)]
    for metric in METRICS:
        cells = [bold(texts[metric][i], deltas[metric][i], best[i]) for i in range(len(T) + 1)]
        verdict = '**significant**' if avg_deltas[metric] < 0.05 else 'not significant'
        res['significance'][metric]['average']['verdict'] = verdict
        lines.append(f'| {metric} | ' + ' | '.join(cells) + f' | {verdict} |')
    lines += ['', '> Values in parentheses are p values of the paired *t* test; † p<0.05, †† p<0.01, '
              '††† p<0.001. The "Verdict" column is based on the **four-set average**.', '']


    lines += ['### Agreement between the two seed batches (5 development vs 5 independent '
              'confirmation)', '',
              '| Batch | DUET | Baseline | Δ | p | Seeds in favour |',
              '|:---|---:|---:|---:|---:|---:|']
    for label, seeds in [('Development 42/0/1/2/3', DEV_SEEDS),
                         ('Confirmation 7/11/13/17/23', CONF_SEEDS),
                         ('Combined, 10 seeds', SEEDS)]:
        a = np.array([np.mean([value(duet[s], sp, 'MCC') for sp in T]) for s in seeds])
        b = np.array([np.mean([value(base[s], sp, 'MCC') for sp in T]) for s in seeds])
        pv = float(ttest_rel(a, b).pvalue)
        lines.append(f'| {label} | {a.mean():.4f} | {b.mean():.4f} | {a.mean()-b.mean():+.4f} '
                     f'| {pv:.4f} | {int((a > b).sum())}/{len(seeds)} |')
        res['replication'][label] = {'duet': float(a.mean()), 'baseline': float(b.mean()),
                                     'delta': float(a.mean() - b.mean()), 'p': pv,
                                     'seeds_in_favour': int((a > b).sum()), 'n': len(seeds)}
    lines += ['', '<!-- AUTO:END -->', '']

    json.dump(res, open(OUT_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    generated = '\n'.join(lines)
    if os.path.exists(OUT_MD):
        doc = open(OUT_MD, encoding='utf-8').read()
        doc = re.sub(r'<!-- AUTO:BEGIN.*?<!-- AUTO:END -->', generated.rstrip(), doc, flags=re.S)
    else:
        doc = '# Per-test-set comparison: DUET (three towers) vs CoBRA-M3 baseline\n\n' + generated
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, 'w', encoding='utf-8').write(doc)

    print(f'[written] {OUT_JSON}\n[written] {OUT_MD}')
    a = np.array([np.mean([value(duet[s], sp, 'MCC') for sp in T]) for s in SEEDS])
    b = np.array([np.mean([value(base[s], sp, 'MCC') for sp in T]) for s in SEEDS])
    print(f'\n10 seeds: DUET {a.mean():.4f} ± {a.std(ddof=1):.4f}  vs  baseline '
          f'{b.mean():.4f} ± {b.std(ddof=1):.4f}   Δ={a.mean()-b.mean():+.4f}  '
          f'p={ttest_rel(a,b).pvalue:.4f}')
    return 0


def stars(p):
    
    if p < 0.001:
        return '†††'
    if p < 0.01:
        return '††'
    return '†' if p < 0.05 else ''


if __name__ == '__main__':
    raise SystemExit(main())
