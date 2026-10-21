from __future__ import annotations

import glob
import json
import os
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
T3 = ['TE18', 'TL12', 'JL10']
SEEDS = [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]
METRICS = ['MCC', 'AUC', 'AUPRC', 'F1', 'Precision', 'Sensitivity', 'Specificity', 'GMean']
THRESHOLDS = [80, 70, 50]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
THR_GRID = np.linspace(0.05, 0.95, 91)
OUT = os.path.join(ROOT, 'results', 'strict_summary.json')
OUT_MD = os.path.join(ROOT, 'docs', 'tables', 'strict_deleaking_2026-10-16.md')
RNG = np.random.default_rng(20261016)


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


def tower_pack(tag, seed, graphs_file):
    gobj = torch.load(os.path.join(ROOT, 'data', 'processed', graphs_file),
                      map_location='cpu', weights_only=False)
    splits = {'val': gobj['graphs']['val']}
    for s in T:
        splits[s] = gobj['graphs'][s]
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
    return acc


def cat(acc, split):
    ks = sorted(k for k in acc if k.startswith(split + '/'))
    return (np.concatenate([acc[k]['y'] for k in ks]),
            np.concatenate([acc[k]['p'] for k in ks]))


def duet_for(thr, seed):
    tags = [f'p{thr}_graph', f'p{thr}_tcn', f'p{thr}_dense']
    gfiles = [f'graphs_seq_ss_clean_p{thr}.pt', f'graphs_dual_clean_p{thr}.pt',
              f'graphs_dual_clean_p{thr}.pt']
    for t in tags:
        if not os.path.exists(os.path.join(ROOT, 'checkpoints', 'controlled', f'{t}_seed{seed}.pt')):
            return None
    members = [tower_pack(t, seed, g) for t, g in zip(tags, gfiles)]
    acc = {k: {'p': np.mean([m[k]['p'] for m in members], axis=0), 'y': members[0][k]['y']}
           for k in members[0]}
    vy, vp = cat(acc, 'val')
    thr_sel = float(THR_GRID[int(np.argmax([matthews_corrcoef(vy, (vp >= t).astype(int))
                                            for t in THR_GRID]))])
    return {'acc': acc, 'thr': thr_sel}


def baseline_for(thr, seed):
    f = os.path.join(ROOT, 'results', f'cobra_m3_equiv_p{thr}.json')
    if not os.path.exists(f):
        return None
    rows = json.load(open(f, encoding='utf-8'))['clean']['bce']
    row = next((r for r in rows if r['seed'] == seed), None)
    if row is None:
        return None
    acc = {}
    for split, chains in row['per_chain'].items():
        for rid, v in chains.items():
            acc[f'{split}/{rid}'] = {'p': np.asarray(v['prob'], dtype=float),
                                     'y': np.asarray(v['label'], dtype=int)}
    return {'acc': acc, 'thr': float(row['threshold'])}


def grassp_for(thr, seed):
    f = os.path.join(ROOT, 'results', f'grassp_deleaked_p{thr}.json')
    if not os.path.exists(f):
        return None
    d = json.load(open(f, encoding='utf-8'))
    if str(seed) not in d.get('seeds', {}):
        return None
    acc = {}
    for split, per_seed in d['per_chain'].items():
        for name, v in per_seed[str(seed)].items():
            acc[f'{split}/{name}'] = {'p': np.asarray(v['p'], dtype=float),
                                      'y': np.asarray(v['y'], dtype=int)}
    vy, vp = cat(acc, 'val')
    thr_sel = float(THR_GRID[int(np.argmax([matthews_corrcoef(vy, (vp >= t).astype(int))
                                            for t in THR_GRID]))])
    return {'acc': acc, 'thr': thr_sel}


def mcc_from_counts(c):
    tp, fp, fn, tn = c
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp * tn - fp * fn) / den if den else 0.0


def counts(pk, seed):
    d = {}
    for k, v in pk['acc'].items():
        pred = (v['p'] >= pk['thr']).astype(int)
        y = v['y']
        d[k] = (int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()),
                int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum()))
    return d


def chain_delta_ci(pk_a, pk_b, seeds, benches, n_boot=1000):
    ca = {s: counts(pk_a[s], s) for s in seeds}
    cb = {s: counts(pk_b[s], s) for s in seeds}
    ch = {sp: sorted(k.split('/', 1)[1] for k in ca[seeds[0]] if k.startswith(sp + '/'))
          for sp in benches}

    def avg(cc, seeds_used, sampled):
        per_seed = []
        for s in seeds_used:
            vals = []
            for sp, cs in sampled.items():
                vals.append(mcc_from_counts(np.sum([cc[s][f'{sp}/{c}'] for c in cs], axis=0)))
            per_seed.append(np.mean(vals))
        return float(np.mean(per_seed))

    obs = avg(ca, seeds, ch) - avg(cb, seeds, ch)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        bms = RNG.integers(0, len(benches), len(benches))
        sampled = {}
        for bi in bms:
            sp = benches[bi]
            sampled[sp] = [ch[sp][i] for i in RNG.integers(0, len(ch[sp]), len(ch[sp]))]
        ss = [seeds[i] for i in RNG.integers(0, len(seeds), len(seeds))]
        boot[b] = avg(ca, ss, sampled) - avg(cb, ss, sampled)
    return {'delta': float(obs),
            'ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            'p_boot': float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))}


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    res = {'thresholds': {}}
    lines = ['# Strict de-leaking: retrained results under 80 / 70 / 50 % identity de-leaking (2026-10-16)', '',
             '> At every threshold the three DUET towers, the CoBRA-M3 equivalent baseline and GRASSP are all **retrained** (10 seeds);',
             '> the protocol matches the main table: per-seed validation-selected thresholds + paired t-test on the same seeds; the 95 % CI of Δ uses',
             '> a chain-level hierarchical bootstrap (benchmark → chain → seed, 1000 resamples).', '',
             '| Identity threshold | Training chains | Validation chains | DUET (10 seeds) | Baseline | Δ ± CI | p |',
             '|:--|--:|--:|--:|--:|:--|--:|']

    for thr in THRESHOLDS:
        leak = json.load(open(os.path.join(ROOT, 'data', f'leaked_pairs_p{thr}.json'),
                              encoding='utf-8'))
        n_tr = leak['cleaned_train_size']
        n_va = len([1 for _ in range(0)]) or (43 - len(leak['remove_val_ids']))
        duet = {s: duet_for(thr, s) for s in SEEDS}
        base = {s: baseline_for(thr, s) for s in SEEDS}
        grassp = {s: grassp_for(thr, s) for s in SEEDS}
        ok = [s for s in SEEDS if duet[s] and base[s]]
        print(f'p{thr}: duet={sum(1 for s in SEEDS if duet[s])}/10 '
              f'baseline={sum(1 for s in SEEDS if base[s])}/10 '
              f'grassp={sum(1 for s in SEEDS if grassp[s])}/10')
        if len(ok) < 2:
            continue
        a = np.array([np.mean([met(*cat(duet[s]['acc'], x), duet[s]['thr'])['MCC'] for x in T])
                      for s in ok])
        b = np.array([np.mean([met(*cat(base[s]['acc'], x), base[s]['thr'])['MCC'] for x in T])
                      for s in ok])
        pv = float(ttest_rel(a, b).pvalue)
        ci = chain_delta_ci({s: duet[s] for s in ok}, {s: base[s] for s in ok}, ok, T)
        entry = {'n_train': n_tr, 'n_val': 43 - len(leak['remove_val_ids']), 'seeds': ok,
                 'duet_mean': float(a.mean()), 'duet_sd': float(a.std(ddof=1)),
                 'baseline_mean': float(b.mean()), 'baseline_sd': float(b.std(ddof=1)),
                 'delta': float((a - b).mean()), 'p_seed_level': pv,
                 'chain_ci': ci, 'per_benchmark': {}}
        for x in T:
            av = np.array([met(*cat(duet[s]['acc'], x), duet[s]['thr'])['MCC'] for s in ok])
            bv = np.array([met(*cat(base[s]['acc'], x), base[s]['thr'])['MCC'] for s in ok])
            entry['per_benchmark'][x] = {'duet': float(av.mean()), 'baseline': float(bv.mean()),
                                         'delta': float((av - bv).mean()),
                                         'p': float(ttest_rel(av, bv).pvalue)}
        okg = [s for s in ok if grassp[s]]
        if okg:
            g = np.array([np.mean([met(*cat(grassp[s]['acc'], x), grassp[s]['thr'])['MCC']
                                   for x in T3]) for s in okg])
            d = np.array([np.mean([met(*cat(duet[s]['acc'], x), duet[s]['thr'])['MCC']
                                   for x in T3]) for s in okg])
            entry['grassp'] = {'n': len(okg), 'mean': float(g.mean()), 'sd': float(g.std(ddof=1)),
                               'duet_3set': float(d.mean()), 'delta': float((d - g).mean()),
                               'p': float(ttest_rel(d, g).pvalue)}
            print(f'   GRASSP {g.mean():.4f} vs DUET(3 sets) {d.mean():.4f} '
                  f'Δ={(d - g).mean():+.4f} p={ttest_rel(d, g).pvalue:.4g} (n={len(okg)})')
        res['thresholds'][f'p{thr}'] = entry
        print(f'   DUET {a.mean():.4f} ± {a.std(ddof=1):.4f} | baseline {b.mean():.4f} ± '
              f'{b.std(ddof=1):.4f} | Δ={(a - b).mean():+.4f} p={pv:.4g} | '
              f'chain CI [{ci["ci95"][0]:+.3f}, {ci["ci95"][1]:+.3f}]')
        lines.append(f'| ≥{thr}% | {n_tr} | {n_va} | {a.mean():.3f} ± {a.std(ddof=1):.3f} | '
                     f'{b.mean():.3f} ± {b.std(ddof=1):.3f} | '
                     f'{ (a - b).mean():+.3f} [{ci["ci95"][0]:+.3f}, {ci["ci95"][1]:+.3f}] | '
                     f'{pv:.4f} |')

    lines += ['', '> Training chains fall from 314 (identical sequences removed only) → 255 (≥80%) → 220 (≥70%) → 56 (≥50%).',
              '> At 50 % the validation set keeps only 4 chains, so threshold selection itself is no longer reliable and that row is indicative of the trend only.', '']
    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, 'w', encoding='utf-8').write('\n'.join(lines))
    print(f'\n[written] {OUT}\n[written] {OUT_MD}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
