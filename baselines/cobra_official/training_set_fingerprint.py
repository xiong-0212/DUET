from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import ttest_ind
from sklearn.metrics import matthews_corrcoef

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

MAX_LEN = 161
OUT = os.path.join(ROOT, 'results', 'cobra_training_fingerprint.json')
COBRA_ROOT = os.environ.get('COBRA_ROOT', os.path.join(ROOT, 'third_party', 'CoBRA-main'))
RINALMO_ROOT = os.environ.get('RINALMO_ROOT', os.path.join(ROOT, 'third_party', 'RiNALMo-main'))


def group_ids():
    leak = json.load(open(os.path.join(ROOT, 'data', 'leaked_pairs.json'), encoding='utf-8'))
    drop = set(leak['remove_train_ids'])
    path = os.path.join(ROOT, 'data', 'raw', 'cobra', 'Hariboss_TR60', 'train.csv')
    ids = [f"{r['PDB_ID']}_{r['Chain']}"
           for r in csv.DictReader(open(path, encoding='utf-8-sig'))]
    return [i for i in ids if i in drop], [i for i in ids if i not in drop]


def _ours(tag, seed=42):
    
    from rgcn_model import build_model
    gobj = torch.load(os.path.join(ROOT, 'data', 'processed', 'graphs_seq_ss.pt'),
                      map_location='cpu', weights_only=False)
    ck = torch.load(os.path.join(ROOT, 'checkpoints', 'controlled', f'{tag}_seed{seed}.pt'),
                    map_location='cpu', weights_only=False)
    cfg = dict(ck['config']['model'])
    cfg['num_relations'] = ck['num_relations']
    cfg['in_dim'] = gobj['in_dim']
    model = build_model(cfg, ck['num_relations']).to('cuda')
    model.load_state_dict(ck['state_dict'])
    model.eval()
    out = {}
    with torch.no_grad():
        for g in gobj['graphs']['train']:
            z = model(g.x.cuda(), g.edge_index.cuda(), g.edge_type.cuda(),
                      torch.zeros(g.x.size(0), dtype=torch.long, device='cuda'))
            if isinstance(z, tuple):
                z = z[0]
            p = torch.sigmoid(z).float().cpu().numpy()
            out[g.rna_id] = float(matthews_corrcoef(g.y.numpy(), (p >= ck['threshold']).astype(int)))
    return out


def summarise(per_chain, A, B):
    a = np.array([per_chain[i] for i in A if i in per_chain])
    b = np.array([per_chain[i] for i in B if i in per_chain])
    return {'A_mean': float(a.mean()), 'A_sd': float(a.std(ddof=1)), 'nA': int(len(a)),
            'B_mean': float(b.mean()), 'B_sd': float(b.std(ddof=1)), 'nB': int(len(b)),
            'gap': float(a.mean() - b.mean()),
            'p': float(ttest_ind(a, b, equal_var=False).pvalue)}


def _released_measurement():
    
    path = os.path.join(ROOT, 'results', 'cobra_training_fingerprint_released.json')
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding='utf-8'))['released_checkpoint']


def _released_measurement():
    
    path = os.path.join(ROOT, 'results', 'cobra_training_fingerprint_released.json')
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding='utf-8'))['released_checkpoint']


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    A, B = group_ids()
    print(f'training set {len(A) + len(B)} chains = group A {len(A)} (sequence identical to a test chain) + group B {len(B)}')
    res = {'protocol': 'per-chain MCC; group A = chains identical to a test sequence '
                       '(removed by de-leaking), group B = the rest; threshold per model',
           'released_checkpoint': None, 'references': {}}
    for tag, label in [('clean_seq_ss', 'graph tower trained on 314 (A unseen)'),
                       ('orig_seq_ss', 'graph tower trained on 345 (A seen)')]:
        s = summarise(_ours(tag), A, B)
        res['references'][label] = s
        print(f'  {label:<38} A {s["A_mean"]:.3f}  B {s["B_mean"]:.3f}  gap {s["gap"]:+.3f} '
              f'(p={s["p"]:.4f})')
    rel = _released_measurement()
    if rel:
        res['released_checkpoint'] = rel
        print(f'  {"CoBRA released checkpoint":<38} A {rel["groupA_mean"]:.3f}  '
              f'B {rel["groupB_mean"]:.3f}  gap {rel["gap"]:+.3f} (p={rel["p"]:.4f})')
    rel = _released_measurement()
    if rel:
        res['released_checkpoint'] = rel
        print(f'  {"CoBRA released checkpoint":<38} A {rel["groupA_mean"]:.3f}  '
              f'B {rel["groupB_mean"]:.3f}  gap {rel["gap"]:+.3f} (p={rel["p"]:.4f})')
    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'[written] {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
