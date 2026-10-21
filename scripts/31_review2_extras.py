from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, matthews_corrcoef

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
MEMBERS = {s: (('clean_seq_ss', 'slim_tcn', 'slim_dense') if s in DEV_SEEDS
               else ('ns_graph', 'ns_tcn', 'ns_dense')) for s in SEEDS}
GRAPH = {'clean_seq_ss': 'graphs_seq_ss_clean.pt', 'ns_graph': 'graphs_seq_ss_clean.pt',
         'slim_tcn': 'graphs_dual_clean.pt', 'ns_tcn': 'graphs_dual_clean.pt',
         'slim_dense': 'graphs_dual_clean.pt', 'ns_dense': 'graphs_dual_clean.pt'}
SHARED = ['1nem_A', '1q8n_A', '2tob_A']
OUT = os.path.join(ROOT, 'results', 'review2_analyses.json')
OUT_MD = os.path.join(ROOT, 'docs', 'analysis', 'second_review_supplementary_analyses_2026-10-16.md')
RNG = np.random.default_rng(20261016)


def met(y, p, thr):
    pred = (p >= thr).astype(int)
    return float(matthews_corrcoef(y, pred))


def pack(tag, seed, graphs_file):
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
                                              'y': g.y.numpy().astype(int),
                                              'len': int(g.x.size(0))}
    return acc


def build():
    duet, base = {}, {}
    for s in SEEDS:
        members = [pack(t, s, GRAPH[t]) for t in MEMBERS[s]]
        acc = {k: {'p': np.mean([m[k]['p'] for m in members], axis=0),
                   'y': members[0][k]['y'], 'len': members[0][k]['len']}
               for k in members[0]}
        vy = np.concatenate([acc[k]['y'] for k in sorted(acc) if k.startswith('val/')])
        vp = np.concatenate([acc[k]['p'] for k in sorted(acc) if k.startswith('val/')])
        duet[s] = {'acc': acc, 'thr': float(THR_GRID[int(np.argmax(
            [matthews_corrcoef(vy, (vp >= t).astype(int)) for t in THR_GRID]))])}
    for s in SEEDS:
        f = os.path.join(ROOT, 'results',
                         'cobra_m3_equiv.json' if s in DEV_SEEDS else 'cobra_m3_equiv_newseeds.json')
        row = next(r for r in json.load(open(f, encoding='utf-8'))['clean']['bce'] if r['seed'] == s)
        acc = {}
        for split, chains in row['per_chain'].items():
            for rid, v in chains.items():
                acc[f'{split}/{rid}'] = {'p': np.asarray(v['prob'], dtype=float),
                                         'y': np.asarray(v['label'], dtype=int), 'len': len(v['label'])}
        base[s] = {'acc': acc, 'thr': float(row['threshold'])}
    return duet, base


def cat(acc, split, keep=None, drop=None):
    ks = sorted(k for k in acc if k.startswith(split + '/'))
    if keep is not None:
        ks = [k for k in ks if k.split('/', 1)[1] in keep]
    if drop is not None:
        ks = [k for k in ks if k.split('/', 1)[1] not in drop]
    return (np.concatenate([acc[k]['y'] for k in ks]),
            np.concatenate([acc[k]['p'] for k in ks]))


def chain_boot(pk, split, n=2000, chains=None):
    
    ks = sorted(k.split('/', 1)[1] for k in pk[SEEDS[0]]['acc'] if k.startswith(split + '/'))
    if chains is not None:
        ks = [k for k in ks if k in chains]
    obs = np.mean([met(*cat(pk[s]['acc'], split), pk[s]['thr']) for s in SEEDS])
    boot = np.empty(n)
    for b in range(n):
        pick = [ks[i] for i in RNG.integers(0, len(ks), len(ks))]
        vals = []
        for s in SEEDS:
            y = np.concatenate([pk[s]['acc'][f'{split}/{c}']['y'] for c in pick])
            p = np.concatenate([pk[s]['acc'][f'{split}/{c}']['p'] for c in pick])
            vals.append(met(y, p, pk[s]['thr']))
        boot[b] = np.mean(vals)
    return {'mean': float(obs), 'ci95': [float(np.percentile(boot, 2.5)),
                                         float(np.percentile(boot, 97.5))]}


def per_chain_delta(da, db, split):
    
    ks = sorted(k.split('/', 1)[1] for k in da[SEEDS[0]]['acc'] if k.startswith(split + '/'))
    out = {}
    for c in ks:
        vals = []
        for s in SEEDS:
            a = da[s]['acc'][f'{split}/{c}']
            b = db[s]['acc'][f'{split}/{c}']
            vals.append(met(a['y'], a['p'], da[s]['thr']) - met(b['y'], b['p'], db[s]['thr']))
        out[c] = float(np.mean(vals))
    return out


def signflip_test(deltas, n=20000):
    
    d = np.asarray(list(deltas), dtype=float)
    obs = d.mean()
    signs = RNG.choice([-1.0, 1.0], size=(n, len(d)))
    null = (signs * d).mean(axis=1)
    return {'observed': float(obs), 'p': float((np.abs(null) >= abs(obs)).mean())}


def seed_permutation(da, db, n=20000):
    
    diffs = []
    for s in SEEDS:
        a = np.mean([met(*cat(da[s]['acc'], sp), da[s]['thr']) for sp in T])
        b = np.mean([met(*cat(db[s]['acc'], sp), db[s]['thr']) for sp in T])
        diffs.append(a - b)
    d = np.asarray(diffs)
    obs = float((d > 0).sum())

    signs = RNG.choice([-1.0, 1.0], size=(n, len(d)))
    null = (signs * d).mean(axis=1)
    return {'mean_delta': float(d.mean()), 'p_signflip': float((np.abs(null) >= abs(d.mean())).mean())}


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    print('[build] evaluating checkpoints ...', flush=True)
    duet, base = build()
    res = {'A_per_benchmark_ci': {}, 'B_tests': {}, 'C_shared_chains': {},
           'D_strata': {}, 'E_thresholds': {}}
    L = ['# Second-review supplementary analyses (2026-10-16)', '',
         '> Generated by `scripts/31_review2_extras.py`; entirely based on published checkpoints, with no new training.', '']

    print('[A] per-benchmark chain-level CIs')
    L += ['## A. Chain-level CI per benchmark (2000 resamples, chain as the unit)', '',
          '| Benchmark | chains | DUET (95% CI) | baseline (95% CI) |', '|:--|--:|:--|:--|']
    for sp in T:
        a = chain_boot(duet, sp)
        b = chain_boot(base, sp)
        n = len([k for k in duet[SEEDS[0]]['acc'] if k.startswith(sp + '/')])
        res['A_per_benchmark_ci'][sp] = {'duet': a, 'baseline': b, 'n_chains': n}
        L.append(f"| {sp} | {n} | {a['mean']:.3f} ({a['ci95'][0]:.3f}–{a['ci95'][1]:.3f}) | "
                 f"{b['mean']:.3f} ({b['ci95'][0]:.3f}–{b['ci95'][1]:.3f}) |")
        print(f'  {sp}: DUET {a["mean"]:.3f} [{a["ci95"][0]:.3f},{a["ci95"][1]:.3f}]  '
              f'base {b["mean"]:.3f} [{b["ci95"][0]:.3f},{b["ci95"][1]:.3f}]')
    L += ['']

    print('[B] permutation tests')
    all_d = {}
    for sp in T:
        all_d[sp] = per_chain_delta(duet, base, sp)
    pooled = [v for sp in T for v in all_d[sp].values()]
    sf = signflip_test(pooled)
    sp_ = seed_permutation(duet, base)
    res['B_tests'] = {'chain_signflip': sf, 'seed_signflip': sp_,
                      'per_benchmark': {sp: signflip_test(all_d[sp].values()) for sp in T}}
    print(f'  chain sign-flip: obs Δ={sf["observed"]:+.4f} p={sf["p"]:.4f}')
    print(f'  seed sign-flip: mean Δ={sp_["mean_delta"]:+.4f} p={sp_["p_signflip"]:.4f}')
    L += ['## B. Permutation tests', '',
          f"- per-chain sign flip (paired differences over 46 chains): observed Δ={sf['observed']:+.4f}, p={sf['p']:.4f}",
          f"- per-seed sign flip (10 seeds): mean Δ={sp_['mean_delta']:+.4f}, p={sp_['p_signflip']:.4f}", '']

    print('[C] shared-chain sensitivity')
    cur = np.mean([np.mean([met(*cat(duet[s]['acc'], x), duet[s]['thr']) for x in T]) for s in SEEDS])
    keep_rb9 = np.mean([np.mean([
        met(*cat(duet[s]['acc'], 'TE18', drop=SHARED), duet[s]['thr']),
        met(*cat(duet[s]['acc'], 'TL12'), duet[s]['thr']),
        met(*cat(duet[s]['acc'], 'JL10'), duet[s]['thr']),
        np.mean([met(duet[s]['acc'][f'RB9/{c}']['y'], duet[s]['acc'][f'RB9/{c}']['p'], duet[s]['thr'])
                 for c in sorted(k.split('/', 1)[1] for k in duet[s]['acc'] if k.startswith('RB9/'))
                 ] + [met(duet[s]['acc'][f'TE18/{c}']['y'], duet[s]['acc'][f'TE18/{c}']['p'],
                          duet[s]['thr']) for c in SHARED])
    ]) for s in SEEDS])
    res['C_shared_chains'] = {'current_te18_18_rb9_6': float(cur),
                              'alternative_te18_15_rb9_9': float(keep_rb9)}
    print(f'  current (TE18 18 / RB9 6): {cur:.4f} | alternative (TE18 15 / RB9 9): {keep_rb9:.4f}')
    L += ['## C. Sensitivity to the three duplicated TE18/RB9 chains', '',
          f"- current (kept in TE18, 6 in RB9): **{cur:.3f}**",
          f"- kept in RB9 instead (15 in TE18, 9 in RB9): **{keep_rb9:.3f}**", '']

    print('[D] length and homology strata')
    L += ['## D. Strata: chain length and homology to the training set', '',
          '| Stratum | Chains | DUET | baseline |', '|:--|--:|--:|--:|']
    lens = sorted(set(x['len'] for s in SEEDS for k, x in duet[s]['acc'].items()
                      if k.startswith(tuple(t + '/' for t in T))))
    med = float(np.median(lens))
    strata = {'short (<= median)': sorted({k for s in SEEDS for k, x in duet[s]['acc'].items()
                                            if k.startswith(tuple(t + '/' for t in T))
                                            and x['len'] <= med}),
              'long (> median)': sorted({k for s in SEEDS for k, x in duet[s]['acc'].items()
                                         if k.startswith(tuple(t + '/' for t in T))
                                         and x['len'] > med})}
    for name, keys in strata.items():
        keys = sorted(set(keys))
        a = np.mean([np.mean([met(duet[s]['acc'][k]['y'], duet[s]['acc'][k]['p'], duet[s]['thr'])
                              for k in keys if k.split('/', 1)[1] == k.split('/', 1)[1]])
                     for s in SEEDS])

        per_seed = []
        for s in SEEDS:
            y = np.concatenate([duet[s]['acc'][k]['y'] for k in keys])
            p = np.concatenate([duet[s]['acc'][k]['p'] for k in keys])
            per_seed.append(met(y, p, duet[s]['thr']))
        per_seed_b = []
        for s in SEEDS:
            y = np.concatenate([base[s]['acc'][k]['y'] for k in keys])
            p = np.concatenate([base[s]['acc'][k]['p'] for k in keys])
            per_seed_b.append(met(y, p, base[s]['thr']))
        res['D_strata'][name] = {'n_chains': len(set(k.split('/', 1)[1] for k in keys)),
                                 'duet': float(np.mean(per_seed)),
                                 'baseline': float(np.mean(per_seed_b))}
        L.append(f'| {name} (per-chain length threshold {med:.0f} nt) | '
                 f'{len(set(k.split("/", 1)[1] for k in keys))} | {np.mean(per_seed):.3f} | '
                 f'{np.mean(per_seed_b):.3f} |')
        print(f'  {name}: DUET {np.mean(per_seed):.3f} base {np.mean(per_seed_b):.3f}')



    audit = json.load(open(os.path.join(ROOT, 'results', 'homology_audit.json'), encoding='utf-8'))
    idc = {c: rec['pid'] for sp, per in audit['per_chain'].items() for c, rec in per.items()}
    hom = {'identical (100 %)': [], '80–99 %': [], 'below 80 %': []}
    for sp in T:
        for k in duet[SEEDS[0]]['acc']:
            if not k.startswith(sp + '/'):
                continue
            c = k.split('/', 1)[1]
            v = idc.get(c, 0.0)
            hom['identical (100 %)' if v >= 0.999 else
                '80–99 %' if v >= 0.8 else 'below 80 %'].append(k)
    for name, keys in hom.items():
        if not keys:
            continue
        keys = sorted(keys)
        per_seed, per_seed_b = [], []
        for s in SEEDS:
            y = np.concatenate([duet[s]['acc'][k]['y'] for k in keys])
            p = np.concatenate([duet[s]['acc'][k]['p'] for k in keys])
            per_seed.append(met(y, p, duet[s]['thr']))
            y2 = np.concatenate([base[s]['acc'][k]['y'] for k in keys])
            p2 = np.concatenate([base[s]['acc'][k]['p'] for k in keys])
            per_seed_b.append(met(y2, p2, base[s]['thr']))
        res['D_strata']['identity ' + name] = {'n_chains': len(set(k.split('/', 1)[1] for k in keys)),
                                               'duet': float(np.mean(per_seed)),
                                               'baseline': float(np.mean(per_seed_b))}
        L.append(f'| {name} vs training set | {len(set(k.split("/", 1)[1] for k in keys))} | '
                 f'{np.mean(per_seed):.3f} | {np.mean(per_seed_b):.3f} |')
        print(f'  identity {name}: DUET {np.mean(per_seed):.3f} base {np.mean(per_seed_b):.3f}')
    L += ['']

    print('[E] thresholds')
    thrs = [duet[s]['thr'] for s in SEEDS]
    res['E_thresholds'] = {'selected': [round(float(t), 3) for t in thrs],
                           'mean': float(np.mean(thrs)), 'sd': float(np.std(thrs, ddof=1))}
    opt = {}
    for sp in T:
        vals = []
        for s in SEEDS:
            y, p = cat(duet[s]['acc'], sp)
            vals.append(float(THR_GRID[int(np.argmax([matthews_corrcoef(y, (p >= t).astype(int))
                                                      for t in THR_GRID]))]))
        opt[sp] = float(np.mean(vals))
    res['E_thresholds']['per_benchmark_optimum'] = opt
    print(f'  selected thresholds: {[round(t, 2) for t in thrs]}')
    print(f'  per-benchmark test-set optimum: { {k: round(v, 2) for k, v in opt.items()} }')
    L += ['## E. Thresholds', '',
          f"- thresholds selected on each seed's validation set: {[round(t, 2) for t in thrs]} (mean {np.mean(thrs):.2f}, "
          f"s.d. {np.std(thrs, ddof=1):.3f})",
          f"- best threshold on the test data itself, per benchmark: { {k: round(v, 2) for k, v in opt.items()} }", '']

    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, 'w', encoding='utf-8').write('\n'.join(L))
    print(f'\n[written] {OUT}\n[written] {OUT_MD}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
