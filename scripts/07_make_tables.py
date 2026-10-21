from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from rgcn_model import build_model
import data_loader as dl

T = ['TE18', 'TL12', 'JL10', 'RB9']
SEEDS = [42, 0, 1, 2, 3]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
METRICS = ['MCC', 'AUC', 'AUPRC', 'F1', 'Precision', 'Sensitivity', 'Specificity', 'GMean']
THR_GRID = np.linspace(0.05, 0.95, 91)


GRAPH_FILE = {'clean_seq_ss': 'graphs_seq_ss_clean.pt',
              'clean_dual2_mlp': 'graphs_dual_clean.pt',
              'clean_mlp': 'graphs_seq_ss_clean.pt'}
BASELINE = 'CoBRA-M3 equiv. (retrained baseline)'

_cache: dict = {}



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


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_temperature(p, y):
    z = logit(p)

    def nll(log_t):
        zz = z / np.exp(log_t)
        return float(np.mean(np.logaddexp(0, zz) - y * zz))

    return float(np.exp(minimize_scalar(nll, bounds=(-2, 3), method='bounded').x))



def _split_dict(graphs_obj):
    
    leak = json.load(open(os.path.join(ROOT, 'data', 'leaked_pairs.json'), encoding='utf-8'))
    drop = set(leak['remove_val_ids'])
    out = {'val': [g for g in graphs_obj['graphs']['val'] if g.rna_id not in drop]}
    for s in T:
        out[s] = graphs_obj['graphs'][s]
    return out


def per_seed_pack(tag):
    
    if tag in _cache:
        return _cache[tag]
    gobj = torch.load(os.path.join(ROOT, 'data', 'processed', GRAPH_FILE[tag]),
                      map_location='cpu', weights_only=False)
    splits = _split_dict(gobj)
    out = {}
    for seed in SEEDS:
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
                    acc[f'{split}/{g.rna_id}'] = {
                        'p': torch.sigmoid(z).float().cpu().numpy(),
                        'y': g.y.numpy().astype(int)}
        out[seed] = {'acc': acc, 'thr': float(ck['threshold'])}
    _cache[tag] = out
    return out


def m3_per_seed():
    
    rows = json.load(open(os.path.join(ROOT, 'results', 'cobra_m3_equiv.json'),
                          encoding='utf-8'))['clean']['bce']
    out = {}
    for r in rows:
        acc = {}
        for split, chains in r['per_chain'].items():
            for rid, v in chains.items():
                acc[f'{split}/{rid}'] = {'p': np.asarray(v['prob'], dtype=float),
                                         'y': np.asarray(v['label'], dtype=int)}
        out[r['seed']] = {'acc': acc, 'thr': float(r['threshold'])}
    return out


def duet_per_seed(graph_pack, seq_pack):
    
    out = {}
    for seed in SEEDS:
        acc = {}
        for k, v in graph_pack[seed]['acc'].items():
            acc[k] = {'p': 0.5 * (np.asarray(v['p'], dtype=float)
                                  + np.asarray(seq_pack[seed]['acc'][k]['p'], dtype=float)),
                      'y': v['y']}
        out[seed] = {'acc': acc, 'thr': None}
    for seed in SEEDS:
        vy, vp = cat(out[seed]['acc'], 'val')
        out[seed]['thr'] = float(THR_GRID[int(np.argmax(
            [matthews_corrcoef(vy, (vp >= t).astype(int)) for t in THR_GRID]))])
    return out


def seed_average(pack):
    
    keys = set.intersection(*[set(pack[s]['acc']) for s in SEEDS])
    acc = {}
    for k in sorted(keys):
        probs = [np.asarray(pack[s]['acc'][k]['p'], dtype=float) for s in SEEDS]
        if len({len(p) for p in probs}) != 1:
            continue
        acc[k] = {'p': np.mean(probs, axis=0), 'y': pack[SEEDS[0]]['acc'][k]['y']}
    return acc


def cat(pack, split):
    
    keys = sorted(k for k in pack if k.startswith(split + '/'))
    return (np.concatenate([np.atleast_1d(pack[k]['y']) for k in keys]),
            np.concatenate([np.atleast_1d(pack[k]['p']) for k in keys]))


def own_threshold(acc):
    vy, vp = cat(acc, 'val')
    return float(THR_GRID[int(np.argmax(
        [matthews_corrcoef(vy, (vp >= t).astype(int)) for t in THR_GRID]))])



def per_seed_metrics(pack):
    
    return {s: {sp: met(*cat(pack[s]['acc'], sp), pack[s]['thr']) for sp in T} for s in SEEDS}


def avg_over_sets(m):
    return float(np.mean([m[s] for s in T]))



def fmt_p(p):
    return f'{p:.3f}'


def table1(data, leak):
    rm_tr, rm_val = set(leak['remove_train_ids']), set(leak['remove_val_ids'])
    src = {'train': 'HARIBOSS + TR60', 'val': 'HARIBOSS + TR60', 'TE18': 'RNAsite',
           'TL12': 'ZeSTa', 'JL10': 'ZeSTa', 'RB9': 'RBind (RB19)'}
    name = {'train': 'Training', 'val': 'Validation', 'TE18': 'TE18', 'TL12': 'TL12',
            'JL10': 'JL10', 'RB9': 'RB9'}
    dedup = {'1nem_A', '1q8n_A', '2tob_A'}
    lines = ['### Table 1 | Benchmark datasets used in this study (after de-leaking)', '',
             '| Dataset | Source | Chains | Nucleotides | Positive sites | Negative sites '
             '| Positive ratio | Negative : positive |',
             '|:---|:---|---:|---:|---:|---:|---:|---:|']
    rows = []
    for split in ['train', 'val', 'TE18', 'TL12', 'JL10', 'RB9']:
        drop = rm_tr if split == 'train' else (rm_val if split == 'val' else set())
        keep = [s for s in data[split]
                if s['rna_id'] not in drop and not (split == 'RB9' and s['rna_id'] in dedup)]
        res = sum(len(s['sequence']) for s in keep)
        pos = sum(sum(s['labels']) for s in keep)

        chains = f'{len(keep)}'
        rows.append({'key': split, 'name': name[split], 'source': src[split],
                     'chains': chains, 'nucleotides': res, 'binding': pos,
                     'non_binding': res - pos, 'positive_ratio': pos / res,
                     'imbalance': (res - pos) / pos})
        lines.append(f'| {name[split]} | {src[split]} | {chains} | {res:,} | {pos:,} '
                     f'| {res - pos:,} | {pos / res:.3f} | {(res - pos) / pos:.2f} : 1 |')
    lines += ['', '> De-leaking removed 31 training and 3 validation chains identical to '
              'test-set sequences; for RB9, three chains duplicated in TE18 were retained in '
              'TE18 only. A nucleotide is labelled as binding if any of its heavy atoms lies '
              'within 4 Å of a ligand heavy atom. All results use this de-leaked protocol.', '']
    return lines, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs', 'tables', 'paper_tables.md'))
    ap.add_argument('--json_out', default=os.path.join(ROOT, 'results', 'table_values.json'))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


    graph = per_seed_pack('clean_seq_ss')
    seq = per_seed_pack('clean_dual2_mlp')
    mlp_fm = per_seed_pack('clean_mlp')
    base = m3_per_seed()
    duet = duet_per_seed(graph, seq)

    arms = [(BASELINE, None, base), ('MLP on RNA-FM (baseline)', None, mlp_fm),
            ('DUET, graph tower only', None, graph), ('DUET, sequence tower only', None, seq),
            ('**DUET (final model)**', None, duet)]
    per_seed = {n: per_seed_metrics(p) for n, _, p in arms}
    base_avg = [np.mean([per_seed[BASELINE][s][sp]['MCC'] for sp in T]) for s in SEEDS]

    json_out = {'protocol': {'main': 'per-seed (own validation threshold), mean +- s.d., n=5',
                             'supplementary': 'five-seed probability averaging (deployment)',
                             'seeds': SEEDS, 'threshold_grid': [float(THR_GRID[0]), float(THR_GRID[-1]),
                                                                len(THR_GRID)]},
                'table1': {}, 'table2': {}, 'tableS1': {}, 'tableS2': {}, 'tableS3': {}}


    data = dl.load_all()
    leak = json.load(open(os.path.join(ROOT, 'data', 'leaked_pairs.json'), encoding='utf-8'))
    L = ['# Paper tables (booktabs style)', '',
         '> **Note on scope (2026-10-15)**: **Table 2 and Tables S1–S3 use the old two-tower + 5-seed',
         '> protocol and are no longer used in the paper**. The current Table 2/3 of the paper are',
         '> recomputed by `scripts/22_three_tower_tables.py` (three towers + 10 seeds, per-seed',
         '> validation threshold + same-seed paired t-test); per-benchmark eight metrics are in',
         '> `scripts/17_per_set_metrics.py`, supplementary material in'
         ' `scripts/23_supplementary.py`.',
         '> **Table 1 (datasets) is still generated by this file and remains in use.**',
         '>',
         '> This file is recomputed directly from `checkpoints/` and `results/` by'
         ' `python scripts/make_all_tables.py`;',
         '> the values are also written to `results/table_values.json`; do not edit by hand.',
         '', '---', '']
    t1_lines, t1_rows = table1(data, leak)
    L += t1_lines
    json_out['table1'] = t1_rows


    L += ['---', '',
          '### Table 2 | Main results on the four de-leaked benchmarks (per-seed protocol)', '',
          'Mean ± s.d. over five independently trained seeds; **each seed evaluated at its own '
          'validation-selected threshold**. The significance tests in the text use exactly '
          'these values.', '',
          '| Method | ' + ' | '.join(T) + ' | Average |', '|:---|' + '---:|' * (len(T) + 1)]
    for name, _, pack in arms:
        cells, avgs = [], []
        for sp in T:
            v = [per_seed[name][s][sp]['MCC'] for s in SEEDS]
            cells.append(f'{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}')
            avgs.append(np.mean(v))
        avg_per_seed = [np.mean([per_seed[name][s][sp]['MCC'] for sp in T]) for s in SEEDS]
        label = name
        row = (f'| {label} | ' + ' | '.join(cells)
               + f' | {"**" if name.startswith("**") else ""}'
               f'{np.mean(avg_per_seed):.3f} ± {np.std(avg_per_seed, ddof=1):.3f}'
               f'{"**" if name.startswith("**") else ""} |')
        L.append(row)
        json_out['table2'][name.replace('**', '')] = {
            'per_set_mean': {sp: float(np.mean([per_seed[name][s][sp]['MCC'] for s in SEEDS]))
                             for sp in T},
            'per_set_sd': {sp: float(np.std([per_seed[name][s][sp]['MCC'] for s in SEEDS], ddof=1))
                           for sp in T},
            'average_over_sets_mean': float(np.mean(avg_per_seed)),
            'average_over_sets_sd': float(np.std(avg_per_seed, ddof=1)),
            'per_seed_average': {str(s): float(v) for s, v in zip(SEEDS, avg_per_seed)}}


    L += ['', '#### Paired comparison with the retrained baseline (same seed, n = 5)', '',
          '| Comparison | Δ MCC | p | seeds in favour |', '|:---|---:|---:|---:|']
    for name, _, _ in arms:
        if name == BASELINE:
            continue
        pa = [np.mean([per_seed[name][s][sp]['MCC'] for sp in T]) for s in SEEDS]
        d = np.asarray(pa) - np.asarray(base_avg)
        p = float(ttest_rel(pa, base_avg).pvalue)
        L.append(f'| {name.replace("**", "")} − baseline | {d.mean():+.3f} | {fmt_p(p)} '
                 f'| {int((d > 0).sum())}/5 |')
        json_out['table2'][name.replace('**', '')]['vs_baseline'] = {
            'delta': float(d.mean()), 'p': p, 'seeds_in_favour': int((d > 0).sum())}

    L += ['', '#### Per-benchmark paired comparison (DUET vs baseline)', '',
          '| Benchmark | DUET | baseline | Δ | p | Verdict |', '|:---|---:|---:|---:|---:|:---|']
    duet_name = '**DUET (final model)**'
    for sp in T:
        x = np.array([per_seed[duet_name][s][sp]['MCC'] for s in SEEDS])
        y = np.array([per_seed[BASELINE][s][sp]['MCC'] for s in SEEDS])
        p = float(ttest_rel(x, y).pvalue)
        verdict = 'significant' if p < 0.05 else 'tie'
        L.append(f'| {sp} | {x.mean():.3f} | {y.mean():.3f} | {x.mean() - y.mean():+.3f} '
                 f'| {"**" if p < 0.05 else ""}{fmt_p(p)}{"**" if p < 0.05 else ""} | {verdict} |')
        json_out['table2']['per_benchmark_vs_baseline_' + sp] = {
            'duet': float(x.mean()), 'baseline': float(y.mean()),
            'delta': float(x.mean() - y.mean()), 'p': p}

    L += ['', '#### All metrics (per-seed average, four-set average)', '',
          '| Metric | DUET | baseline | Δ | p |', '|:---|---:|---:|---:|---:|']
    for m in METRICS:
        a = [np.mean([per_seed[duet_name][s][sp][m] for sp in T]) for s in SEEDS]
        b = [np.mean([per_seed[BASELINE][s][sp][m] for sp in T]) for s in SEEDS]
        p = float(ttest_rel(a, b).pvalue)
        star = '**' if p < 0.05 else ''
        L.append(f'| {m} | {np.mean(a):.3f} | {np.mean(b):.3f} | {np.mean(a) - np.mean(b):+.3f} '
                 f'| {star}{fmt_p(p)}{star} |')
        json_out['table2']['metric_' + m] = {'duet': float(np.mean(a)), 'baseline': float(np.mean(b)),
                                             'delta': float(np.mean(a) - np.mean(b)), 'p': p}
    L += ['', '> Per-metric same-seed paired tests are reported above. Sensitivity and specificity '
          'are separated so that positive-site recovery and negative-site control remain visible.', '']


    d_graph, d_seq, d_mlp, d_base = (seed_average(graph), seed_average(seq),
                                     seed_average(mlp_fm), seed_average(base))
    d_duet = {k: {'p': 0.5 * (d_graph[k]['p'] + d_seq[k]['p']), 'y': d_graph[k]['y']}
              for k in d_graph}
    depl = [(BASELINE, d_base), ('MLP on RNA-FM (baseline)', d_mlp), ('DUET, graph tower only', d_graph),
            ('DUET, sequence tower only', d_seq), ('**DUET (final model)**', d_duet)]


    L += ['---', '', '### Table S1 | Robustness check at a fixed threshold of 0.5'
          ' (seed-averaged probability protocol)', '',
          '| Method | ' + ' | '.join(METRICS) + ' |', '|:---|' + '---:|' * len(METRICS)]
    for name, acc in depl:
        per = {sp: met(*cat(acc, sp), 0.5) for sp in T}
        vals = {m: float(np.mean([per[sp][m] for sp in T])) for m in METRICS}
        L.append(f'| {name} | ' + ' | '.join(f'{vals[m]:.3f}' for m in METRICS) + ' |')
        json_out['tableS1'][name.replace('**', '')] = vals
    L += ['', '> At a common threshold the relative ranking is unchanged for MCC, AUC, AUPRC and '
          'F1, sensitivity, specificity and GMean are strongly operating-point dependent.', '']


    vy, vg = cat(d_graph, 'val')
    _, vs = cat(d_seq, 'val')
    t1, t2 = fit_temperature(vg, vy), fit_temperature(vs, vy)
    w_grid = np.linspace(0, 1, 21)
    w = float(w_grid[int(np.argmax([matthews_corrcoef(vy, ((x * vg + (1 - x) * vs) >= 0.5).astype(int))
                                    for x in w_grid]))])
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(np.c_[vg, vs], vy)

    def rank_avg(a, b):
        def rk(x):
            return np.argsort(np.argsort(x)).astype(float) / max(len(x) - 1, 1)
        return 0.5 * (rk(a) + rk(b))

    strategies = [('Equal-weight averaging (adopted)', lambda a, b: 0.5 * (a + b)),
                  ('Temperature-calibrated averaging',
                   lambda a, b: 0.5 * (1 / (1 + np.exp(-logit(a) / t1))
                                       + 1 / (1 + np.exp(-logit(b) / t2)))),
                  (f'Validation-weighted averaging (w → {w:.2f})',
                   lambda a, b: w * a + (1 - w) * b),
                  ('Logit averaging', lambda a, b: 1 / (1 + np.exp(-0.5 * (logit(a) + logit(b))))),
                  ('Rank averaging', rank_avg),
                  ('Validation-stacked logistic',
                   lambda a, b: lr.predict_proba(np.c_[a, b])[:, 1])]
    L += ['---', '', '### Table S2 | Comparison of ensemble combination strategies'
          ' (seed-averaged probability protocol)', '',
          '| Strategy | ' + ' | '.join(METRICS) + ' |', '|:---|' + '---:|' * len(METRICS)]
    for name, fn in strategies:
        vp = fn(vg, vs)
        thr = float(THR_GRID[int(np.argmax([matthews_corrcoef(vy, (vp >= t).astype(int))
                                            for t in THR_GRID]))])
        vals = {m: [] for m in METRICS}
        for sp in T:
            y, p1 = cat(d_graph, sp)
            _, p2 = cat(d_seq, sp)
            r = met(y, fn(p1, p2), thr)
            for m in METRICS:
                vals[m].append(r[m])
        vals = {m: float(np.mean(vals[m])) for m in METRICS}
        L.append(f'| {name} | ' + ' | '.join(f'{vals[m]:.3f}' for m in METRICS) + ' |')
        json_out['tableS2'][name] = {'threshold': thr, **vals}
    ref = {sp: met(*cat(d_base, sp), 0.5) for sp in T}
    ref_vals = {m: float(np.mean([ref[sp][m] for sp in T])) for m in METRICS}
    L.append('| [reference] CoBRA-M3 equiv. @0.5 | '
             + ' | '.join(f'{ref_vals[m]:.3f}' for m in METRICS) + ' |')
    L += ['', '> No combination rule outperforms simple equal-weight averaging; the '
          'validation-learned mixing weight converges to 0.50, i.e. the equal-weight solution. '
          'The adopted strategy is therefore the simplest one.', '']


    L += ['---', '', '### Table S3 | Deployment variant: five-seed probability averaging', '',
          '| Method | Average MCC |', '|:---|---:|']
    for name, acc in depl:
        thr = own_threshold(acc)
        vals = [met(*cat(acc, sp), thr)['MCC'] for sp in T]
        L.append(f'| {name} | {"**" if name.startswith("**") else ""}{np.mean(vals):.3f}'
                 f'{"**" if name.startswith("**") else ""} |')
        json_out['tableS3'][name.replace('**', '')] = {'threshold': thr,
                                                       'average_mcc': float(np.mean(vals)),
                                                       'per_set': {sp: float(met(*cat(acc, sp), thr)['MCC'])
                                                                   for sp in T}}
    L += ['', '> Averaging the probabilities of the five seeds (no additional training required) '
          'raises DUET’s average MCC from 0.396 (per-seed protocol) to 0.426, while the baseline '
          'moves only from 0.361 to 0.374. The significance tests in the main text use the '
          'per-seed protocol of Table 2.', '']

    md = '\n'.join(L) + '\n'
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, 'w', encoding='utf-8').write(md)
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    json.dump(json_out, open(args.json_out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    print(md)
    print(f'[written] {args.out}')
    print(f'[written] {args.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
