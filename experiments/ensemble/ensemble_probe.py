from __future__ import annotations

import glob
import itertools
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from rgcn_model import build_model
from utils import compute_metrics, find_best_threshold

T = ['TE18', 'TL12', 'JL10', 'RB9']
SEEDS = [42, 0, 1, 2, 3]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


MEMBERS = {
    'seq_ss':   ('clean_seq_ss', 'graphs_seq_ss_clean.pt'),
    'mlp':      ('clean_mlp', 'graphs_seq_ss_clean.pt'),
    'rinalmo':  ('clean_rinalmo_ss', 'graphs_seq_ss_rinalmo_clean.pt'),
    'rinalmo512': ('clean_rinalmo_ss_h512', 'graphs_seq_ss_rinalmo_clean.pt'),
    'ss_w1':    ('clean_ss_w1', 'graphs_ss_w1_clean.pt'),
    'prune90':  ('clean_ss_prune90', 'graphs_ss_prune90_clean.pt'),
    'dualview': ('clean_dualview', 'graphs_seq_ss_clean.pt'),
    'fusion':   ('clean_fusion', 'graphs_seq_ss_clean.pt'),
}
CACHE = os.path.join(ROOT, 'results', 'ensemble_member_probs.json')


@torch.no_grad()
def member_probs(prefix, graph_file, gobj_cache={}):
    if graph_file not in gobj_cache:
        gobj_cache[graph_file] = torch.load(
            os.path.join(ROOT, 'data', 'processed', graph_file),
            map_location='cpu', weights_only=False)
    gobj = gobj_cache[graph_file]
    leaked = json.load(open(os.path.join(ROOT, 'data', 'leaked_pairs.json'),
                            encoding='utf-8'))
    rm_val = set(leaked['remove_val_ids'])
    splits = {'val': [g for g in gobj['graphs']['val'] if g.rna_id not in rm_val]}
    for s in T:
        splits[s] = gobj['graphs'][s]

    acc, ths = {}, []
    for seed in SEEDS:
        ck_path = os.path.join(ROOT, 'checkpoints', 'controlled',
                               f'{prefix}_seed{seed}.pt')
        if not os.path.exists(ck_path):
            return None
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        mcfg = dict(ck['config']['model'])
        mcfg['num_relations'] = ck['num_relations']
        mcfg['in_dim'] = gobj.get('in_dim') or mcfg.get('in_dim', 640)
        model = build_model(mcfg, ck['num_relations']).to(DEV)
        model.load_state_dict(ck['state_dict'])
        model.eval()
        ths.append(float(ck['threshold']))
        for split, gs in splits.items():
            for g in gs:
                ew = g.edge_weight.to(DEV) if getattr(g, 'edge_weight', None) is not None else None
                z = model(g.x.to(DEV), g.edge_index.to(DEV), g.edge_type.to(DEV),
                          torch.zeros(g.x.size(0), dtype=torch.long, device=DEV),
                          edge_weight=ew)
                p = torch.sigmoid(z).float().cpu().numpy()
                rec = acc.setdefault((split, g.rna_id), {'p': [], 'y': None})
                rec['p'].append(p)
                rec['y'] = g.y.numpy().astype(int)
    out = {}
    for (split, rid), rec in acc.items():
        out.setdefault(split, {})[rid] = dict(
            prob=np.mean(rec['p'], axis=0).tolist(), label=rec['y'].tolist())
    return {'threshold_mean': float(np.mean(ths)), 'splits': out}


def eval_combo(probs, keys, fixed_thr=None):
    
    if fixed_thr is None:
        val_p, val_y = [], []
        for rid, rec in probs[keys[0]]['splits']['val'].items():
            ps = [np.asarray(probs[k]['splits']['val'][rid]['prob']) for k in keys]
            val_p.append(np.mean(ps, axis=0))
            val_y.append(np.asarray(rec['label']))
        val_p, val_y = np.concatenate(val_p), np.concatenate(val_y)
        th, _ = find_best_threshold(val_y, val_p, 'mcc')
    else:
        th = float(fixed_thr)
    per = {}
    for split in T:
        P, Y = [], []
        for rid, rec in probs[keys[0]]['splits'][split].items():
            ps = [np.asarray(probs[k]['splits'][split][rid]['prob']) for k in keys]
            P.append(np.mean(ps, axis=0))
            Y.append(np.asarray(rec['label']))
        p, y = np.concatenate(P), np.concatenate(Y)
        per[split] = float(compute_metrics(y, p, th)['MCC'])
    return float(np.mean(list(per.values()))), per, float(th)


def member_from_m3(ds='clean', loss='bce'):
    
    p = os.path.join(ROOT, 'results', 'cobra_m3_equiv.json')
    if not os.path.exists(p):
        return None
    rows = json.load(open(p, encoding='utf-8'))[ds][loss]
    if not rows[0].get('per_chain') or 'val' not in rows[0]['per_chain']:
        return None
    acc = {}
    for r in rows:
        for split, chains in r['per_chain'].items():
            if split == 'hariofficial_test':
                continue
            for rid, v in chains.items():
                rec = acc.setdefault((split, rid), {'p': [], 'y': v['label']})
                rec['p'].append(np.asarray(v['prob']))
    out = {}
    for (split, rid), rec in acc.items():
        out.setdefault(split, {})[rid] = dict(
            prob=np.mean(rec['p'], axis=0).tolist(), label=rec['y'])
    return {'threshold_mean': float(np.mean([r['threshold'] for r in rows])),
            'splits': out}


def main():
    probs = {}
    for name, (prefix, gf) in MEMBERS.items():
        m = member_probs(prefix, gf)
        if m is None:
            print(f'  [skip] {name}: incomplete checkpoints', flush=True)
            continue
        probs[name] = m
        print(f'  [ok] {name}', flush=True)
    m3 = member_from_m3()
    if m3:
        probs['cobra_m3'] = m3
        print('  [ok] cobra_m3 (our retrained RiNALMo + official head)', flush=True)
    keys = list(probs)

    results = {}
    print(f'\n{"combination":<46}{"val thr":<9}{"TE18":<8}{"TL12":<8}{"JL10":<8}{"RB9":<8}{"mean"}')
    rows = []
    for r in range(1, min(len(keys), 4) + 1):
        for combo in itertools.combinations(keys, r):
            avg, per, th = eval_combo(probs, list(combo))
            rows.append((combo, avg, per, th))
    rows.sort(key=lambda x: -x[1])
    for combo, avg, per, th in rows[:18]:
        print(f'{"+".join(combo):<46}{th:<9.2f}'
              + ''.join(f'{per[s]:<8.3f}' for s in T) + f'{avg:.4f}')
    best = rows[0]
    print(f'\nbest combination: {"+".join(best[0])} → {best[1]:.4f}'
          f' (best single model {max(r[1] for r in rows if len(r[0])==1):.4f})')


    print(f'\n===== fixed threshold 0.5 (protocol-independent control) =====')
    print(f'{"combination":<46}{"TE18":<8}{"TL12":<8}{"JL10":<8}{"RB9":<8}{"mean"}')
    fixed_rows = []
    for combo in [k for k in [(k,) for k in keys]] + \
                 [c for c in itertools.combinations(keys, 2)] + \
                 ([tuple(keys)] if len(keys) > 2 else []):
        key = '+'.join(combo)
        avg, per, _ = eval_combo(probs, list(combo), fixed_thr=0.5)
        fixed_rows.append((key, avg, per))
    fixed_rows.sort(key=lambda x: -x[1])
    for key, avg, per in fixed_rows:
        print(f'{key:<46}' + ''.join(f'{per[s]:<8.3f}' for s in T) + f'{avg:.4f}')

    with open(os.path.join(ROOT, 'results', 'ensemble_probe.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'rows': [{'combo': list(c), 'avg': a, 'per': p, 'th': t}
                            for c, a, p, t in rows],
                   'fixed_0.5': [{'combo': k, 'avg': a, 'per': p}
                                 for k, a, p in fixed_rows]},
                  f, ensure_ascii=False, indent=1)

    with open(os.path.join(ROOT, 'results', 'ensemble_member_probs.json'), 'w',
              encoding='utf-8') as f:
        json.dump(probs, f, ensure_ascii=False)
    print('wrote results/ensemble_probe.json + ensemble_member_probs.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
