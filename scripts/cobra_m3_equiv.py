from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from utils import compute_metrics, find_best_threshold, save_json, set_seed
import data_loader as dl

EMB = os.path.join(ROOT, 'data', 'processed', 'rinalmo_embeddings.pt')
LEAKED = os.path.join(ROOT, 'data', 'leaked_pairs.json')
T = ['TE18', 'TL12', 'JL10', 'RB9']
SEEDS = [int(x) for x in os.environ.get('C0BRA_SEEDS', '42,0,1,2,3').split(',')]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


class CoBRAHead(nn.Module):
    

    def __init__(self, in_dim=1280, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.LayerNorm(1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        return self.classifier(self.mlp(x))[:, 1]


def load_samples(remove_leaked):
    emb = torch.load(EMB, map_location='cpu', weights_only=False)['embeddings']
    data = dl.load_all()
    remove = set()
    if remove_leaked:
        remove = set(json.load(open(LEAKED, encoding='utf-8'))['remove_train_ids'])
    remove_val = set(json.load(open(LEAKED, encoding='utf-8'))['remove_val_ids'])

    out = {}
    for split, samples in data.items():
        recs = []
        for s in samples:
            rid = s['rna_id']
            if split == 'train' and rid in remove:
                continue
            if split == 'val' and rid in remove_val:
                continue
            e = emb.get(rid)
            if e is None or e.shape[0] != len(s['sequence']):
                continue
            recs.append(dict(rid=rid, x=e.float(),
                             y=torch.tensor([int(c) for c in s['labels']],
                                            dtype=torch.float)))
        out[split] = recs
    return out


def train_eval(data, seed, loss_name='bce', epochs=120, patience=15, bs=8, quiet=False):
    set_seed(seed)
    train, val = data['train'], data['val']
    model = CoBRAHead().to(DEV)
    n_pos = float(sum(r['y'].sum().item() for r in train))
    n_neg = float(sum(len(r['y']) - r['y'].sum().item() for r in train))
    pw = None
    if loss_name == 'bce_pw':
        pw = torch.tensor([max(n_neg / max(n_pos, 1.0), 3.0)], device=DEV)

    def crit(z, y):
        return F.binary_cross_entropy_with_logits(z, y, pos_weight=pw)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2,
                                                               eta_min=1e-6)
    best = {'mcc': -2.0, 'state': None, 'epoch': -1}
    bad = 0

    @torch.no_grad()
    def predict(recs):
        model.eval()
        P, Y = [], []
        for r in recs:
            z = model(r['x'].to(DEV))
            P.append(torch.sigmoid(z).float().cpu().numpy())
            Y.append(r['y'].numpy().astype(int))
        return np.concatenate(Y), np.concatenate(P)

    g = torch.Generator().manual_seed(seed)
    for ep in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=g).tolist()
        tot, nb = 0.0, 0
        for i in range(0, len(order), bs):
            batch = [train[j] for j in order[i:i + bs]]
            x = torch.cat([r['x'] for r in batch]).to(DEV)
            y = torch.cat([r['y'] for r in batch]).to(DEV)
            loss = crit(model(x), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss)
            nb += 1
        sch.step()
        yv, pv = predict(val)
        mcc = compute_metrics(yv, pv, 0.5)['MCC']
        if mcc > best['mcc']:
            best.update(mcc=mcc, epoch=ep,
                        state={k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best['state'])
    yv, pv = predict(val)
    th, _ = find_best_threshold(yv, pv, 'mcc')
    test, per_chain = {}, {'val': {}}
    model.eval()
    for r in val:
        with torch.no_grad():
            z = model(r['x'].to(DEV))
        per_chain['val'][r['rid']] = dict(
            prob=[float(x) for x in torch.sigmoid(z).float().cpu().numpy()],
            label=[int(x) for x in r['y'].numpy()], mcc=None)
    for split in T:
        if split not in data or not data[split]:
            continue
        yt, pt = predict(data[split])
        test[split] = {'at_threshold': compute_metrics(yt, pt, th),
                       'at_0.5': compute_metrics(yt, pt, 0.5)}

        per_chain[split] = {}
        model.eval()
        for r in data[split]:
            with torch.no_grad():
                z = model(r['x'].to(DEV))
            p = torch.sigmoid(z).float().cpu().numpy()
            per_chain[split][r['rid']] = dict(
                prob=[float(x) for x in p],
                label=[int(x) for x in r['y'].numpy()],
                mcc=float(compute_metrics(r['y'].numpy().astype(int), p, th)['MCC'])
                if len(r['y']) >= 2 and len(np.unique(r['y'].numpy())) >= 2 else None)
    avg = float(np.mean([test[s]['at_threshold']['MCC'] for s in T if s in test]))
    if not quiet:
        print(f'  seed {seed} [{loss_name}] best_ep={best["epoch"]} th={th:.2f} '
              f'4-set mean={avg:.4f}', flush=True)
    return dict(seed=seed, loss=loss_name, best_epoch=best['epoch'], threshold=th,
                avg=avg, test=test, per_chain=per_chain,
                n_train=len(train), n_val=len(val))


def main():
    only = sys.argv[1:]
    modes = [only[0]] if only else ['clean', 'orig']
    losses = [only[1]] if len(only) > 1 else ['bce', 'bce_pw']

    out = {}
    if os.path.exists(os.path.join(ROOT, 'results', os.environ.get('C0BRA_OUT', 'cobra_m3_equiv.json'))):
        old = json.load(open(os.path.join(ROOT, 'results', os.environ.get('C0BRA_OUT', 'cobra_m3_equiv.json')),
                             encoding='utf-8'))
        for m, per_loss in old.items():
            if not isinstance(per_loss, dict):
                continue
            for lname, rows in per_loss.items():
                if rows and rows[0].get("per_chain") and "val" in rows[0]["per_chain"]:
                    out.setdefault(m, {})[lname] = rows
    for mode, remove_leaked in [('clean', True), ('orig', False)]:
        if mode not in modes:
            continue
        data = load_samples(remove_leaked)
        print(f'== {mode}: train={len(data["train"])} val={len(data["val"])} ==',
              flush=True)
        out.setdefault(mode, {})
        for loss_name in losses:
            if out[mode].get(loss_name):
                print(f'  {mode}/{loss_name}: per_chain results already present, skipping', flush=True)
                continue
            rows = []
            for seed in SEEDS:
                rows.append(train_eval(data, seed, loss_name))
            out[mode][loss_name] = rows
            avg = float(np.mean([r['avg'] for r in rows]))
            std = float(np.std([r['avg'] for r in rows], ddof=1))
            print(f'{mode}/{loss_name}: mean MCC over 5 seeds = {avg:.4f} ± {std:.4f}',
                  flush=True)
    save_json(out, os.path.join(ROOT, 'results', os.environ.get('C0BRA_OUT', 'cobra_m3_equiv.json')))
    print('written results/cobra_m3_equiv.json')
    return 0


if __name__ == '__main__':
    t0 = time.time()
    raise SystemExit(main())
