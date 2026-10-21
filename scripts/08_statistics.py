from __future__ import annotations

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
N_BOOT = 2000
N_PERM = 10000
RNG = np.random.default_rng(20261011)
CACHE = os.path.join(ROOT, 'results', 'stats_perchain.json')


GRAPH_ARMS = {
    'seq_ss':     ('clean_seq_ss', 'graphs_seq_ss_clean.pt'),
    'mlp':        ('clean_mlp', 'graphs_seq_ss_clean.pt'),
    'rinalmo_ss': ('clean_rinalmo_ss', 'graphs_seq_ss_rinalmo_clean.pt'),
    'rinalmo_ss_h256': ('clean_rinalmo_ss_h256', 'graphs_seq_ss_rinalmo_clean.pt'),
    'rinalmo_ss_h512': ('clean_rinalmo_ss_h512', 'graphs_seq_ss_rinalmo_clean.pt'),
    'rinalmo_ss_skip': ('clean_rinalmo_ss_skip', 'graphs_seq_ss_rinalmo_clean.pt'),
}
LABEL = {
    'seq_ss': 'R-GCN seq+ss (RNA-FM)',
    'mlp': 'MLP (RNA-FM)',
    'cobra_m3': 'CoBRA-M3 equiv (RiNALMo+MLP+BCE)',
    'rinalmo_ss': 'R-GCN seq+ss (RiNALMo, h128)',
    'rinalmo_ss_h256': 'R-GCN seq+ss (RiNALMo, h256)',
    'rinalmo_ss_h512': 'R-GCN seq+ss (RiNALMo, h512)',
    'rinalmo_ss_skip': 'R-GCN seq+ss (RiNALMo, input-skip)',
}


@torch.no_grad()
def collect_graph_arm(tag, prefix, graph_file):
    gobj = torch.load(os.path.join(ROOT, 'data', 'processed', graph_file),
                      map_location='cpu', weights_only=False)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ths, acc = [], {}
    for seed in SEEDS:
        ck_path = os.path.join(ROOT, 'checkpoints', 'controlled',
                               f'{prefix}_seed{seed}.pt')
        if not os.path.exists(ck_path):
            return None
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        mcfg = dict(ck['config']['model'])
        mcfg['num_relations'] = ck['num_relations']

        mcfg['in_dim'] = gobj.get('in_dim') or mcfg.get('in_dim', 640)
        model = build_model(mcfg, ck['num_relations']).to(dev)
        model.load_state_dict(ck['state_dict'])
        model.eval()
        ths.append(float(ck['threshold']))
        for split in T:
            for g in gobj['graphs'][split]:
                z = model(g.x.to(dev), g.edge_index.to(dev), g.edge_type.to(dev),
                          torch.zeros(g.x.size(0), dtype=torch.long, device=dev))
                p = torch.sigmoid(z).float().cpu().numpy()

                rec = acc.setdefault(f'{split}/{g.rna_id}',
                                     {'split': split, 'probs': [], 'label': None})
                rec['probs'].append(p.tolist())
                rec['label'] = g.y.numpy().astype(int).tolist()

    out = {k: dict(split=rec['split'], probs=rec['probs'], label=rec['label'])
           for k, rec in acc.items()}
    return {'thresholds': [float(t) for t in ths], 'chains': out}


def collect_from_json_model(mode='clean', loss='bce'):
    p = os.path.join(ROOT, 'results', 'cobra_m3_equiv.json')
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding='utf-8'))
    rows = d.get(mode, {}).get(loss, [])
    if not rows or not rows[0].get('per_chain'):
        return None
    ths = [float(r['threshold']) for r in rows]
    acc = {}
    for r in rows:
        for split, chains in r['per_chain'].items():
            for rid, v in chains.items():
                rec = acc.setdefault(f'{split}/{rid}',
                                     {'split': split, 'probs': [], 'label': v['label']})
                rec['probs'].append(list(v['prob']))
    out = {k: dict(split=rec['split'], probs=rec['probs'], label=rec['label'])
           for k, rec in acc.items()}
    return {'thresholds': ths, 'chains': out}


def load_all_models():
    if os.path.exists(CACHE):
        print(f'[cache] reusing {CACHE}', flush=True)
        return json.load(open(CACHE, encoding='utf-8'))
    models = {}
    for tag, (prefix, gf) in GRAPH_ARMS.items():
        m = collect_graph_arm(tag, prefix, gf)
        if m is None:
            print(f'  [skip] {tag}: checkpoint not found', flush=True)
            continue
        models[tag] = m
        print(f'  {tag}: {len(m["chains"])} chains thresholds={m["thresholds"]}', flush=True)
    m3 = collect_from_json_model()
    if m3:
        models['cobra_m3'] = m3
        print(f'  cobra_m3: {len(m3["chains"])} chains thresholds={m3["thresholds"]}', flush=True)
    json.dump(models, open(CACHE, 'w', encoding='utf-8'), ensure_ascii=False)
    return models



def pooled_metric(model, chains_by_split):
    
    per_seed = []
    for si, th in enumerate(model['thresholds']):
        mccs = []
        for split in T:
            P, Y = [], []
            for rid in chains_by_split[split]:
                rec = model['chains'][rid]
                P.append(np.asarray(rec['probs'][si]))
                Y.append(np.asarray(rec['label']))
            if not P:
                continue
            p, y = np.concatenate(P), np.concatenate(Y)
            mccs.append(matthews_corrcoef(y, (p >= th).astype(int)))
        per_seed.append(float(np.mean(mccs)))
    return float(np.mean(per_seed))


def chains_map(models, key):
    return {split: [rid for rid, rec in models[key]['chains'].items()
                    if rec['split'] == split] for split in T}


def bootstrap(models, pairs, n_boot=N_BOOT):
    ids = {k: chains_map(models, k) for k in models}
    allids = {split: sorted(set.intersection(*[set(ids[k][split]) for k in models]))
              for split in T}
    stats_out = {k: [] for k in models}
    delta = {p: [] for p in pairs}
    for b in range(n_boot):
        samp = {split: [allids[split][i] for i in
                        RNG.integers(0, len(allids[split]), len(allids[split]))]
                for split in T}
        vals = {}
        for k in models:
            vals[k] = pooled_metric(models[k], samp)
            stats_out[k].append(vals[k])
        for a, bb in pairs:
            delta[(a, bb)].append(vals[a] - vals[bb])
    res = {}
    for k, v in stats_out.items():
        v = np.array(v)
        res[k] = dict(mean=float(v.mean()), lo=float(np.percentile(v, 2.5)),
                      hi=float(np.percentile(v, 97.5)))
    res['_delta'] = {}
    for pr, v in delta.items():
        v = np.array(v)
        res['_delta'][f'{pr[0]} - {pr[1]}'] = dict(
            mean=float(v.mean()), lo=float(np.percentile(v, 2.5)),
            hi=float(np.percentile(v, 97.5)), p_gt0=float((v > 0).mean()))
    return res


def permutation_test(models, a, b, n_perm=N_PERM):
    
    d = []
    for rid, rec in models[a]['chains'].items():
        if rid not in models[b]['chains']:
            continue
        y = np.asarray(rec['label'])

        pa = np.mean([np.asarray(rec['probs'][i]) >= th
                      for i, th in enumerate(models[a]['thresholds'])], axis=0) >= 0.5
        recb = models[b]['chains'][rid]
        pb = np.mean([np.asarray(recb['probs'][i]) >= th
                      for i, th in enumerate(models[b]['thresholds'])], axis=0) >= 0.5
        if len(y) < 2 or len(np.unique(y)) < 2:
            continue
        d.append(matthews_corrcoef(y, pa.astype(int))
                 - matthews_corrcoef(y, pb.astype(int)))
    d = np.array(d)
    obs = float(d.mean())
    signs = RNG.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d).mean(axis=1)
    p = float((np.abs(null) >= abs(obs)).mean())
    return dict(n_chains=int(len(d)), delta=obs, p_perm=p)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    models = load_all_models()
    keys = list(models)
    print(f'\nModels compared: {keys}', flush=True)
    pairs = [('seq_ss', 'mlp'), ('seq_ss', 'cobra_m3')]
    for k in keys:
        if k.startswith('rinalmo'):
            pairs += [('seq_ss', k), (k, 'cobra_m3'), (k, 'mlp')]
    pairs = [p for p in pairs if p[0] in models and p[1] in models]
    bs = bootstrap(models, pairs)
    perm = {f'{a} - {b}': permutation_test(models, a, b) for a, b in pairs}

    print(f"\n{'Model':<36}{'4-set MCC':<12}{'95% CI (chain bootstrap)':<26}")
    for k in keys:
        v = bs[k]
        print(f'{LABEL.get(k, k):<36}{v["mean"]:<12.4f}[{v["lo"]:.3f}, {v["hi"]:.3f}]')
    print(f"\n{'Comparison':<40}{'Δ':<10}{'95% CI':<22}{'P(Δ>0)':<9}{'perm p'}")
    for a, b in pairs:
        d = bs['_delta'][f'{a} - {b}']
        pt = perm[f'{a} - {b}']
        print(f'{LABEL.get(a,a)+" vs "+LABEL.get(b,b):<44}{d["mean"]:<+10.4f}'
              f'[{d["lo"]:+.3f}, {d["hi"]:+.3f}]   {d["p_gt0"]:<9.3f}{pt["p_perm"]:.4f}')

    out = {'bootstrap': bs, 'permutation': perm,
           'labels': {k: LABEL.get(k, k) for k in keys},
           'n_chains': {k: len(models[k]['chains']) for k in keys}}
    with open(os.path.join(ROOT, 'results', 'stats_rigor.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print('\n[written] results/stats_rigor.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
