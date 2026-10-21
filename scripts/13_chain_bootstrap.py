from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from rgcn_model import build_model

T = ['TE18', 'TL12', 'JL10', 'RB9']
SEEDS = [42, 0, 1, 2, 3]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
THR_GRID = np.linspace(0.05, 0.95, 91)

MEMBERS = {'graph': ('clean_seq_ss', 'graphs_seq_ss_clean.pt'),
           'sequence': ('clean_dual2_mlp', 'graphs_dual_clean.pt')}
BASELINE_JSON = os.path.join(ROOT, 'results', 'cobra_m3_equiv.json')
OUT = os.path.join(ROOT, 'results', 'chain_bootstrap.json')


def val_filter(graphs_obj):
    leak = json.load(open(os.path.join(ROOT, 'data', 'leaked_pairs.json'), encoding='utf-8'))
    drop = set(leak['remove_val_ids'])
    return [g for g in graphs_obj['graphs']['val'] if g.rna_id not in drop]


def member_probs(tag, graphs_file):
    
    gobj = torch.load(os.path.join(ROOT, 'data', 'processed', graphs_file),
                      map_location='cpu', weights_only=False)
    splits = {'val': val_filter(gobj)}
    splits.update({s: gobj['graphs'][s] for s in T})
    acc = {}
    for seed in SEEDS:
        ck = torch.load(os.path.join(ROOT, 'checkpoints', 'controlled', f'{tag}_seed{seed}.pt'),
                        map_location='cpu', weights_only=False)
        cfg = dict(ck['config']['model'])
        cfg['num_relations'] = ck['num_relations']
        cfg['in_dim'] = gobj['in_dim']
        model = build_model(cfg, ck['num_relations']).to(DEV)
        model.load_state_dict(ck['state_dict'])
        model.eval()
        with torch.no_grad():
            for split, graphs in splits.items():
                for g in graphs:
                    z = model(g.x.to(DEV), g.edge_index.to(DEV), g.edge_type.to(DEV),
                              torch.zeros(g.x.size(0), dtype=torch.long, device=DEV))
                    if isinstance(z, tuple):
                        z = z[0]
                    k = f'{split}/{g.rna_id}'
                    acc.setdefault(k, {'p': [], 'y': g.y.numpy().astype(int)})
                    acc[k]['p'].append(torch.sigmoid(z).float().cpu().numpy())
    return {k: {'p': np.mean(v['p'], axis=0), 'y': v['y']} for k, v in acc.items()}


def baseline_probs():
    rows = json.load(open(BASELINE_JSON, encoding='utf-8'))['clean']['bce']
    acc = {}
    for r in rows:
        for split, chains in r['per_chain'].items():
            if split not in T + ['val']:
                continue
            for rid, v in chains.items():
                k = f'{split}/{rid}'
                acc.setdefault(k, {'p': [], 'y': np.asarray(v['label'], dtype=int)})
                acc[k]['p'].append(np.asarray(v['prob'], dtype=float))
    return {k: {'p': np.mean(v['p'], axis=0), 'y': v['y']} for k, v in acc.items()}


def cat(pack, keys):
    y = np.concatenate([pack[k]['y'] for k in keys])
    p = np.concatenate([pack[k]['p'] for k in keys])
    return y, p


def mcc(pack, keys, thr):
    y, p = cat(pack, keys)
    return float(matthews_corrcoef(y, (p >= thr).astype(int)))


def own_threshold(pack):
    keys = sorted(k for k in pack if k.startswith('val/'))
    y, p = cat(pack, keys)
    return float(THR_GRID[int(np.argmax(
        [matthews_corrcoef(y, (p >= t).astype(int)) for t in THR_GRID]))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_boot', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=11)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    graph = member_probs(*MEMBERS['graph'])
    seq = member_probs(*MEMBERS['sequence'])
    base = baseline_probs()
    duet = {k: {'p': 0.5 * (graph[k]['p'] + seq[k]['p']), 'y': graph[k]['y']} for k in graph}
    thr = {'DUET': own_threshold(duet), 'baseline': own_threshold(base)}
    keys = {s: sorted(k for k in duet if k.startswith(s + '/')) for s in T}

    def per_set(pack, s, kk):
        return mcc(pack, kk, thr['DUET' if pack is duet else 'baseline'])

    point = {s: per_set(duet, s, keys[s]) - per_set(base, s, keys[s]) for s in T}
    point_avg = float(np.mean([point[s] for s in T]))

    rng = np.random.default_rng(args.seed)
    reps = {s: [] for s in T}
    for _ in range(args.n_boot):
        for s in T:
            kk = [keys[s][i] for i in rng.integers(0, len(keys[s]), len(keys[s]))]
            reps[s].append(per_set(duet, s, kk) - per_set(base, s, kk))
    avg = np.mean([np.asarray(reps[s]) for s in T], axis=0)

    out = {'protocol': {'deployment': 'five-seed probability averaging',
                        'threshold_DUET': thr['DUET'], 'threshold_baseline': thr['baseline'],
                        'n_boot': args.n_boot, 'seed': args.seed,
                        'resampling': 'chains resampled within each benchmark'},
           'average': {'delta': point_avg, 'ci95': [float(np.percentile(avg, 2.5)),
                                                    float(np.percentile(avg, 97.5))],
                       'p_delta_gt_0': float((avg > 0).mean())}}
    print(f"DUET thr={thr['DUET']:.2f}  baseline thr={thr['baseline']:.2f}")
    print(f"average ΔMCC = {point_avg:+.4f}  95% CI "
          f"[{out['average']['ci95'][0]:+.4f}, {out['average']['ci95'][1]:+.4f}]  "
          f"P(Δ>0) = {out['average']['p_delta_gt_0']:.3f}")
    for s in T:
        d = np.asarray(reps[s])
        out[s] = {'delta': point[s], 'ci95': [float(np.percentile(d, 2.5)),
                                              float(np.percentile(d, 97.5))],
                  'p_delta_gt_0': float((d > 0).mean())}
        print(f"  {s:<6} Δ={point[s]:+.4f}  CI[{out[s]['ci95'][0]:+.3f}, "
              f"{out[s]['ci95'][1]:+.3f}]  P(Δ>0)={out[s]['p_delta_gt_0']:.3f}")

    json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'[written] {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
