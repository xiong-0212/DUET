from __future__ import annotations

import glob
import json
import os
import shutil
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from rgcn_model import build_model
from utils import compute_metrics

T = ['TE18', 'TL12', 'JL10', 'RB9']
BACKUP = os.path.join(ROOT, 'archive', 'results_pre_overlap_fix')
VERSION = 'overlap_fix_2026-10-11'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def eval_ckpt(ck_path):
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    gpath = ck.get('graphs_file')
    if not gpath:
        return None
    if not os.path.isabs(gpath):
        gpath = os.path.join(ROOT, gpath)
    if not os.path.exists(gpath):
        return None
    gobj = torch.load(gpath, map_location='cpu', weights_only=False)
    mcfg = dict(ck['config']['model'])
    mcfg['num_relations'] = ck['num_relations']
    mcfg['in_dim'] = gobj.get('in_dim') or mcfg.get('in_dim', 640)
    model = build_model(mcfg, ck['num_relations']).to(DEV)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    th = ck['threshold']
    out = {}
    with torch.no_grad():
        for split in T:
            gs = gobj['graphs'].get(split, [])
            if not gs:
                continue
            P, Y = [], []
            for g in gs:
                z = model(g.x.to(DEV), g.edge_index.to(DEV), g.edge_type.to(DEV),
                          torch.zeros(g.x.size(0), dtype=torch.long, device=DEV))
                P.append(torch.sigmoid(z).float().cpu().numpy())
                Y.append(g.y.numpy().astype(int))
            p, y = np.concatenate(P), np.concatenate(Y)
            out[split] = {'at_threshold': compute_metrics(y, p, th),
                          'at_0.5': compute_metrics(y, p, 0.5),
                          'n_residues': int(len(y))}
    return out


def main():
    os.makedirs(BACKUP, exist_ok=True)
    jobs = []
    for ck in sorted(glob.glob(os.path.join(ROOT, 'checkpoints', 'controlled', '*.pt'))):
        jobs.append((ck, os.path.join(ROOT, 'results', 'controlled')))
    for ck in sorted(glob.glob(os.path.join(ROOT, 'checkpoints', '*.pt'))):
        jobs.append((ck, os.path.join(ROOT, 'results')))
    print(f'{len(jobs)} checkpoints to re-evaluate', flush=True)

    before_after = {}
    for i, (ck, resdir) in enumerate(jobs, 1):
        tag = os.path.basename(ck)[:-3]
        rp = os.path.join(resdir, f'train_{tag}.json')
        ts = eval_ckpt(ck)
        if ts is None or not os.path.exists(rp):
            continue
        shutil.copy2(rp, os.path.join(BACKUP, os.path.basename(rp)))
        r = json.load(open(rp, encoding='utf-8'))
        old = r.get('test_metrics', {})
        if old.get('RB9') and ts.get('RB9'):
            before_after[tag] = (float(np.mean([old[s]['at_threshold']['MCC'] for s in T if s in old])),
                                 float(np.mean([ts[s]['at_threshold']['MCC'] for s in T if s in ts])))
        r['test_metrics'] = ts
        r['eval_split_version'] = VERSION
        with open(rp, 'w', encoding='utf-8') as f:
            json.dump(r, f, ensure_ascii=False, indent=1, default=float)
        if i % 20 == 0:
            print(f'  {i}/{len(jobs)}', flush=True)


    m3p = os.path.join(ROOT, 'results', 'cobra_m3_equiv.json')
    if os.path.exists(m3p):
        shutil.copy2(m3p, os.path.join(BACKUP, 'cobra_m3_equiv.json'))
        d = json.load(open(m3p, encoding='utf-8'))
        DROP = {'1nem_A', '1q8n_A', '2tob_A'}
        for mode in d:
            for loss in d[mode]:
                for row in d[mode][loss]:
                    th = row['threshold']
                    avg, tst = [], {}
                    for split in T:
                        chains = {k: v for k, v in row['per_chain'][split].items()
                                  if not (split == 'RB9' and k in DROP)}
                        y = np.concatenate([np.array(v['label']) for v in chains.values()])
                        p = np.concatenate([np.array(v['prob']) for v in chains.values()])
                        tst[split] = {'at_threshold': compute_metrics(y, p, th),
                                      'at_0.5': compute_metrics(y, p, 0.5)}
                    row['test'] = tst
                    row['avg'] = float(np.mean([tst[s]['at_threshold']['MCC'] for s in T]))
        d['eval_split_version'] = VERSION
        with open(m3p, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=1, default=float)
        print('  cobra_m3_equiv.json recomputed', flush=True)


    print(f'\n{"checkpoint":<34}{"old 4-set mean":<12}{"new 4-set mean":<12}Δ')
    for tag in ['clean_seq_ss_seed42', 'clean_seq_ss_alt_seed42', 'clean_mlp_seed42',
                'orig_seq_ss_seed42', 'orig_seq_ss_alt_seed42', 'orig_mlp_seed42',
                'bp_prob_seed42']:
        if tag in before_after:
            a, b = before_after[tag]
            print(f'{tag:<34}{a:<12.4f}{b:<12.4f}{b-a:+.4f}')
    print(f'\nOld results backed up to {BACKUP}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
