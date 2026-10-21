from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from scipy.stats import pearsonr, ttest_rel
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

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
DEV = 'cpu'
THR_GRID = np.linspace(0.05, 0.95, 91)
METRICS = ['MCC', 'AUC', 'AUPRC', 'F1', 'Precision', 'Sensitivity', 'Specificity', 'GMean']
OUT = os.path.join(ROOT, 'results', 'rigour_analyses.json')
OUT_MD = os.path.join(ROOT, 'docs', 'analysis', 'review_supplementary_analyses_2026-10-16.md')
RNG = np.random.default_rng(20261016)

MODELS = {
    'graph':    ('clean_seq_ss', 'ns_graph',       'graphs_seq_ss_clean.pt'),
    'tcn':      ('slim_tcn',     'ns_tcn',         'graphs_dual_clean.pt'),
    'dense':    ('slim_dense',   'ns_dense',       'graphs_dual_clean.pt'),
    'mlp_rnafm': ('clean_mlp',   'ns_mlp_rnafm',   'graphs_seq_ss_clean.pt'),
    'cap_match': ('capm_m1275',  'ns_cap_m1275',   'graphs_dual_clean.pt'),
    'cap_large': ('capm_m2048',  'ns_cap_m2048',   'graphs_dual_clean.pt'),
    'cap_union': ('capm_m1920',  'ns_cap_m1920',   'graphs_dual_clean.pt'),
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


def best_thr(y, p):
    return float(THR_GRID[int(np.argmax([matthews_corrcoef(y, (p >= t).astype(int))
                                         for t in THR_GRID]))])


def pack_tag(tag, seed, graphs_file):
    gobj = torch.load(os.path.join(ROOT, 'data', 'processed', graphs_file),
                      map_location='cpu', weights_only=False)
    leak = json.load(open(os.path.join(ROOT, 'data', 'leaked_pairs.json'), encoding='utf-8'))
    drop = set(leak['remove_val_ids'])
    splits = {'val': [g for g in gobj['graphs']['val'] if g.rna_id not in drop]}
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
                acc[f'{split}/{g.rna_id}'] = {'p': torch.sigmoid(z).float().numpy(),
                                              'y': g.y.numpy().astype(int)}
    return {'acc': acc, 'thr': float(ck['threshold'])}


def baseline_pack(seed):
    fname = 'cobra_m3_equiv.json' if seed in DEV_SEEDS else 'cobra_m3_equiv_newseeds.json'
    rows = json.load(open(os.path.join(ROOT, 'results', fname), encoding='utf-8'))['clean']['bce']
    row = next(r for r in rows if r['seed'] == seed)
    acc = {}
    for split, chains in row['per_chain'].items():
        for rid, v in chains.items():
            acc[f'{split}/{rid}'] = {'p': np.asarray(v['prob'], dtype=float),
                                     'y': np.asarray(v['label'], dtype=int)}
    return {'acc': acc, 'thr': float(row['threshold'])}


def combine(packs):
    keys = set(packs[0]['acc'])
    for p in packs[1:]:
        keys &= set(p['acc'])
    acc = {k: {'p': np.mean([p['acc'][k]['p'] for p in packs], axis=0),
               'y': packs[0]['acc'][k]['y']} for k in keys}
    vy, vp = cat(acc, 'val')
    return {'acc': acc, 'thr': best_thr(vy, vp)}


def cat(acc, split):
    ks = sorted(k for k in acc if k.startswith(split + '/'))
    return (np.concatenate([acc[k]['y'] for k in ks]),
            np.concatenate([acc[k]['p'] for k in ks]))


def per_chain_mcc(pk, seed_key=None):
    
    return {k.split('/', 1)[1]: float(matthews_corrcoef(v['y'], (v['p'] >= pk['thr']).astype(int)))
            for k, v in pk['acc'].items() if k.startswith(tuple(s + '/' for s in T))}


def build_all():
    packs = {}
    for name, (dev, conf, gf) in MODELS.items():
        packs[name] = {}
        for s in SEEDS:
            tag = dev if s in DEV_SEEDS else conf
            if os.path.exists(os.path.join(ROOT, 'checkpoints', 'controlled',
                                           f'{tag}_seed{s}.pt')):
                packs[name][s] = pack_tag(tag, s, gf)
    packs['baseline'] = {s: baseline_pack(s) for s in SEEDS}
    packs['duet'] = {s: combine([packs['graph'][s], packs['tcn'][s], packs['dense'][s]])
                     for s in SEEDS}
    return packs



def chain_counts(pk):
    
    out = {}
    for s in SEEDS:
        d = {}
        for k, v in pk[s]['acc'].items():
            pred = (v['p'] >= pk[s]['thr']).astype(int)
            y = v['y']
            d[k] = (int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()),
                    int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum()))
        out[s] = d
    return out


def mcc_from_counts(c):
    tp, fp, fn, tn = c
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp * tn - fp * fn) / den if den else 0.0


def avg_mcc(pk_counts, seeds, chains_by_split):
    
    per_seed = []
    for s in seeds:
        vals = []
        for sp, cs in chains_by_split.items():
            c = np.sum([pk_counts[s][f'{sp}/{c_}'] for c_ in cs], axis=0)
            vals.append(mcc_from_counts(c))
        per_seed.append(np.mean(vals))
    return float(np.mean(per_seed))


def chain_keys(pk):
    return {sp: sorted(k.split('/', 1)[1] for k in pk[SEEDS[0]]['acc']
                       if k.startswith(sp + '/')) for sp in T}


def chain_bootstrap_ci(pk, n_boot=1000):
    
    counts = chain_counts(pk)
    ch = chain_keys(pk)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sampled = {sp: [ch[sp][i] for i in RNG.integers(0, len(ch[sp]), len(ch[sp]))]
                   for sp in T}
        boot[b] = avg_mcc(counts, SEEDS, sampled)
    return {'mean': avg_mcc(counts, SEEDS, ch),
            'ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def hierarchical_bootstrap(pk_a, pk_b, n_boot=1000):
    
    ca, cb = chain_counts(pk_a), chain_counts(pk_b)
    ch = chain_keys(pk_a)
    d_boot = np.empty(n_boot)
    for b in range(n_boot):
        bms = RNG.integers(0, len(T), len(T))
        sampled = {}
        for bi in bms:
            sp = T[bi]
            sampled[sp] = [ch[sp][i] for i in RNG.integers(0, len(ch[sp]), len(ch[sp]))]
        seeds = [SEEDS[i] for i in RNG.integers(0, len(SEEDS), len(SEEDS))]
        d_boot[b] = avg_mcc(ca, seeds, sampled) - avg_mcc(cb, seeds, sampled)
    obs = avg_mcc(ca, SEEDS, ch) - avg_mcc(cb, SEEDS, ch)
    return {'delta': float(obs),
            'ci95': [float(np.percentile(d_boot, 2.5)), float(np.percentile(d_boot, 97.5))],
            'p_boot': float(2 * min((d_boot <= 0).mean(), (d_boot >= 0).mean()))}



def micro_macro(pk):
    macro, micro = [], []
    for s in SEEDS:
        per_set, ys, ps = [], [], []
        for sp in T:
            y, p = cat(pk[s]['acc'], sp)
            per_set.append(met(y, p, pk[s]['thr'])['MCC'])
            ys.append(y); ps.append(p)
        macro.append(np.mean(per_set))
        micro.append(met(np.concatenate(ys), np.concatenate(ps), pk[s]['thr'])['MCC'])
    return {'macro': float(np.mean(macro)), 'micro': float(np.mean(micro)),
            'per_seed_macro': [float(x) for x in macro],
            'per_seed_micro': [float(x) for x in micro]}


def leave_one_out(pk):
    
    loo_b = {}
    for sp in T:
        vals = []
        for s in SEEDS:
            vals.append(np.mean([met(*cat(pk[s]['acc'], x), pk[s]['thr'])['MCC']
                                 for x in T if x != sp]))
        loo_b[sp] = float(np.mean(vals))
    base = float(np.mean([np.mean([met(*cat(pk[s]['acc'], x), pk[s]['thr'])['MCC'] for x in T])
                          for s in SEEDS]))
    chain_effect = {}
    for sp in T:
        chains = sorted(k.split('/', 1)[1] for k in pk[SEEDS[0]]['acc'] if k.startswith(sp + '/'))
        per_chain = {}
        for c in chains:
            vals = []
            for s in SEEDS:
                others = [x for x in chains if x != c]
                per_set = []
                for x in T:
                    cs = others if x == sp else sorted(
                        k.split('/', 1)[1] for k in pk[s]['acc'] if k.startswith(x + '/'))
                    y = np.concatenate([pk[s]['acc'][f'{x}/{cc}']['y'] for cc in cs])
                    p = np.concatenate([pk[s]['acc'][f'{x}/{cc}']['p'] for cc in cs])
                    per_set.append(met(y, p, pk[s]['thr'])['MCC'])
                vals.append(np.mean(per_set))
            per_chain[c] = float(np.mean(vals))
        chain_effect[sp] = sorted(per_chain.items(), key=lambda kv: kv[1])
    return {'baseline_average': base, 'loo_benchmark': loo_b,
            'most_influential_chains': {sp: [{'chain': c, 'avg_without': v}
                                             for c, v in chain_effect[sp][:3]]
                                        for sp in T}}



def threshold_sensitivity(pk):
    out = {}
    for a in [0.5]:
        vals = [np.mean([met(*cat(pk[s]['acc'], x), a)['MCC'] for x in T]) for s in SEEDS]
        out[f'fixed_{a}'] = float(np.mean(vals))
    out['validation_selected'] = float(np.mean(
        [np.mean([met(*cat(pk[s]['acc'], x), pk[s]['thr'])['MCC'] for x in T]) for s in SEEDS]))
    oracle = []
    for s in SEEDS:
        per_set = []
        for x in T:
            y, p = cat(pk[s]['acc'], x)
            per_set.append(met(y, p, best_thr(y, p))['MCC'])
        oracle.append(np.mean(per_set))
    out['oracle_threshold'] = float(np.mean(oracle))
    out['selected_thresholds'] = [round(float(pk[s]['thr']), 3) for s in SEEDS]
    return out



def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    tot, n = 0.0, len(y)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            tot += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(tot)


def calibration(pk):
    ys = np.concatenate([cat(pk[s]['acc'], x)[0] for s in SEEDS for x in T])
    ps = np.concatenate([cat(pk[s]['acc'], x)[1] for s in SEEDS for x in T])
    return {'brier': float(brier_score_loss(ys, ps)), 'ece_10bin': ece(ys, ps),
            'mean_probability': float(ps.mean()), 'positive_rate': float(ys.mean())}



def q_statistic(ya, pa, yb, pb, thr_a, thr_b):
    
    ca = (pa >= thr_a).astype(int)
    cb = (pb >= thr_b).astype(int)
    n11 = int(((ca == ya) & (cb == ya)).sum())
    n00 = int(((ca != ya) & (cb != ya)).sum())
    n10 = int(((ca == ya) & (cb != ya)).sum())
    n01 = int(((ca != ya) & (cb == ya)).sum())
    den = n11 * n00 + n01 * n10
    return float((n11 * n00 - n01 * n10) / den) if den else float('nan')


def diversity(pk, members=('graph', 'tcn', 'dense')):
    out = {}
    npairs = [(members[i], members[j]) for i in range(len(members)) for j in range(i + 1, len(members))]
    for a, b in npairs:
        corrs, qs, dis, dflt = [], [], [], []
        for s in SEEDS:
            ya = np.concatenate([cat(pk[a][s]['acc'], x)[0] for x in T])
            pa = np.concatenate([cat(pk[a][s]['acc'], x)[1] for x in T])
            yb = np.concatenate([cat(pk[b][s]['acc'], x)[0] for x in T])
            pb = np.concatenate([cat(pk[b][s]['acc'], x)[1] for x in T])
            corrs.append(pearsonr(pa, pb)[0])
            qs.append(q_statistic(ya, pa, yb, pb, pk[a][s]['thr'], pk[b][s]['thr']))
            ca = (pa >= pk[a][s]['thr']).astype(int)
            cb = (pb >= pk[b][s]['thr']).astype(int)
            dis.append(float((ca != cb).mean()))
            dflt.append(float(((ca != ya) & (cb != yb)).mean()))
        out[f'{a}|{b}'] = {'prob_correlation': float(np.mean(corrs)),
                           'q_statistic': float(np.nanmean(qs)),
                           'disagreement': float(np.mean(dis)),
                           'double_fault': float(np.mean(dflt))}

    orc = []
    for s in SEEDS:
        y = np.concatenate([cat(pk['graph'][s]['acc'], x)[0] for x in T])
        ps = [np.concatenate([cat(pk[m][s]['acc'], x)[1] for x in T]) for m in members]
        right = np.stack([((p >= pk[m][s]['thr']).astype(int) == y) for m, p in zip(members, ps)])
        orc.append(float(right.any(axis=0).mean()))
    out['oracle_any_member_correct'] = float(np.mean(orc))
    return out


def fusion_rules(pk, members=('graph', 'tcn', 'dense')):
    
    from scipy.stats import rankdata

    def fuse(name, s, ps):
        if name == 'plain':
            return np.mean(ps, axis=0)
        if name == 'rank':
            return np.mean([rankdata(c) / len(c) for c in ps], axis=0)
        if name == 'logit':
            lg = [np.log(np.clip(c, 1e-6, 1 - 1e-6) / (1 - np.clip(c, 1e-6, 1 - 1e-6)))
                  for c in ps]
            return 1 / (1 + np.exp(-np.mean(lg, axis=0)))
        if name == 'majority':
            votes = np.stack([(c >= pk[m][s]['thr']).astype(float)
                              for m, c in zip(members, ps)])
            return (votes.mean(axis=0) > 0.5).astype(float)

        w = [max(met(*cat(pk[m][s]['acc'], 'val'), pk[m][s]['thr'])['MCC'], 1e-3)
             for m in members]
        w = np.array(w) / np.sum(w)
        return np.sum([wi * ci for wi, ci in zip(w, ps)], axis=0)

    rules = {}
    for name in ['plain', 'rank', 'weighted_val', 'logit', 'majority']:
        per_seed = []
        for s in SEEDS:
            vy, _ = cat(pk[members[0]][s]['acc'], 'val')
            vp = fuse(name, s, [pk[m][s]['acc'][k]['p'] for m in members
                                for k in sorted(pk[m][s]['acc']) if k.startswith('val/')]
                      if False else
                      [np.concatenate([pk[m][s]['acc'][k]['p'] for k in
                                       sorted(pk[m][s]['acc']) if k.startswith('val/')])
                       for m in members])
            thr = best_thr(vy, vp)
            per_set = []
            for x in T:
                chains = sorted(k.split('/', 1)[1] for k in pk[members[0]][s]['acc']
                                if k.startswith(x + '/'))
                ps = [np.concatenate([pk[m][s]['acc'][f'{x}/{c}']['p'] for c in chains])
                      for m in members]
                y = np.concatenate([pk[members[0]][s]['acc'][f'{x}/{c}']['y'] for c in chains])
                per_set.append(met(y, fuse(name, s, ps), thr)['MCC'])
            per_seed.append(np.mean(per_set))
        rules[name] = float(np.mean(per_seed))
    return rules


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    print('[build] evaluating checkpoints on CPU ...', flush=True)
    pk = build_all()
    res = {'protocol': 'ten seeds, per-seed validation-selected thresholds'}
    L = ['# Review supplementary analyses (2026-10-16)', '',
         '> Generated by `scripts/27_rigour_analyses.py`; based on published checkpoints, with no new training.', '']

    print('[A] chain-level / hierarchical bootstrap')
    res['chain_level'] = {}
    for name in ['duet', 'baseline']:
        res['chain_level'][name] = chain_bootstrap_ci(pk[name])
        print(f'  {name}: {res["chain_level"][name]["mean"]:.4f} '
              f'CI {res["chain_level"][name]["ci95"]}')
    res['hierarchical_delta'] = {'MCC': hierarchical_bootstrap(pk['duet'], pk['baseline'])}
    v = res['hierarchical_delta']['MCC']
    print(f'  DUET - baseline MCC: Δ={v["delta"]:+.4f} '
          f'CI {[round(x, 3) for x in v["ci95"]]} p_boot={v["p_boot"]:.4g}')

    print('[B] macro vs micro')
    res['macro_micro'] = {n: micro_macro(pk[n]) for n in ['duet', 'baseline']}
    res['leave_one_out'] = {n: leave_one_out(pk[n]) for n in ['duet', 'baseline']}
    for n in ['duet', 'baseline']:
        print(f'  {n}: macro={res["macro_micro"][n]["macro"]:.4f} '
              f'micro={res["macro_micro"][n]["micro"]:.4f}')
        print(f'     LOO benchmarks: '
              + ', '.join(f'{k} {v:.3f}' for k, v in res['leave_one_out'][n]['loo_benchmark'].items()))

    print('[C] threshold sensitivity')
    res['threshold'] = {n: threshold_sensitivity(pk[n]) for n in ['duet', 'baseline']}
    for n in ['duet', 'baseline']:
        t = res['threshold'][n]
        print(f'  {n}: fixed0.5={t["fixed_0.5"]:.4f} val={t["validation_selected"]:.4f} '
              f'oracle={t["oracle_threshold"]:.4f}')

    print('[D] calibration')
    res['calibration'] = {n: calibration(pk[n]) for n in ['graph', 'tcn', 'dense', 'duet', 'baseline']}
    for n, v in res['calibration'].items():
        print(f'  {n:<9} Brier={v["brier"]:.4f} ECE={v["ece_10bin"]:.4f}')

    print('[E] diversity + fusion')
    res['diversity'] = diversity(pk)
    res['fusion'] = fusion_rules(pk)
    for k, v in res['diversity'].items():
        if isinstance(v, dict):
            print(f'  {k}: r={v["prob_correlation"]:.3f} Q={v["q_statistic"]:.3f} '
                  f'disagree={v["disagreement"]:.3f} DF={v["double_fault"]:.3f}')
    print('  fusion:', {k: round(v, 4) for k, v in res['fusion'].items()})

    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n[written] {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
