from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from torch_geometric.data import Data

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)

GRASSP_ROOT = os.environ.get('GRASSP_ROOT', os.path.join(ROOT, 'third_party', 'GRASSP-main'))
sys.path.insert(0, os.path.join(GRASSP_ROOT, 'model'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

BENCH = ['TE18', 'TL12', 'JL10']
SPLITS = {'train': 'HARI60NEW_TRAIN', 'val': 'HARI60NEW_VAL'}
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = os.path.join(ROOT, 'results', os.environ.get('GRASSP_OUT', 'grassp_deleaked.json'))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_chains(subdir):
    

    for rel in (os.path.join('data', 'processed', 'CORBRA', subdir),
                os.path.join('data', 'processed', subdir)):
        path = os.path.join(GRASSP_ROOT, rel, 'processed', 'processed_data_hariboss.pt')
        if os.path.exists(path):
            break
    else:
        raise FileNotFoundError(f'no processed data for {subdir} under {GRASSP_ROOT}')
    data, slices = torch.load(path, map_location='cpu', weights_only=False)
    names = list(data.pdb_name)
    xs, es, ys, ss = slices['x'], slices['edge_index'], slices['y'], slices['ss_emb']
    out = {}
    for i, nm in enumerate(names):
        n0, n1 = int(xs[i]), int(xs[i + 1])
        e0, e1 = int(es[i]), int(es[i + 1])


        ei = data.edge_index[:, e0:e1]
        out[nm] = Data(x=data.x[n0:n1], edge_index=ei, y=data.y[n0:n1],
                       ss_emb=data.ss_emb[n0:n1])
    return out


def to_grassp_name(rna_id):
    
    pdb, _, chain = rna_id.partition('_')
    return (pdb + chain).upper()


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    se = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    return {'MCC': float(matthews_corrcoef(y, pred)), 'AUC': float(roc_auc_score(y, p)),
            'AUPRC': float(average_precision_score(y, p)),
            'F1': float(f1_score(y, pred, zero_division=0)),
            'Precision': float(precision_score(y, pred, zero_division=0)),
            'Recall': float(recall_score(y, pred, zero_division=0)),
            'GMean': float(np.sqrt(se * sp)), 'Accuracy': float(accuracy_score(y, pred))}


@torch.no_grad()
def predict(model, chains, device=DEV):
    
    model.eval()
    probs, labels = [], []
    for _name, d in chains:
        d = d.to(device)
        batch = torch.zeros(d.x.size(0), dtype=torch.long, device=device)
        logit = model(d.x, d.ss_emb, d.edge_index, batch)
        probs.append(torch.sigmoid(logit).float().cpu().numpy())
        labels.append(d.y.cpu().numpy().astype(int))
    return np.concatenate(probs), np.concatenate(labels)


def evaluate(model, chains_per_split, thr=0.5):
    out = {}
    for split, chains in chains_per_split.items():
        p, y = predict(model, chains)
        out[split] = metrics(y, p, thr)
    return out


def train_one(seed, train_chains, val_chains, test_chains, epochs=100, lr=1e-4, wd=1e-4,
              hidden=86, dropout=0.6, patience=100, quiet=False, batch_size=1):
    
    from RNABP import HybridRNABindingSiteModel
    from torch_geometric.data import Batch

    set_seed(seed)
    model = HybridRNABindingSiteModel(rna_dim=768, ss_dim=4, hidden=hidden,
                                      dropout=dropout).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n_pos = int(sum(int(d.y.sum()) for _, d in train_chains))
    n_neg = int(sum(d.y.numel() - int(d.y.sum()) for _, d in train_chains))
    pos_weight = torch.tensor([max(n_neg / max(1, n_pos), 3)], device=DEV)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)

    best_mcc, best_state, best_th = -1.0, None, 0.5
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_chains)))
        random.shuffle(order)
        running = 0.0
        for lo in range(0, len(order), batch_size):
            chunk = order[lo:lo + batch_size]
            if batch_size == 1:
                _, d = train_chains[chunk[0]]
                bd = d
                b = torch.zeros(d.x.size(0), dtype=torch.long, device=DEV)
                y = d.y
            else:
                bd = Batch.from_data_list([train_chains[j][1] for j in chunk])
                b, y = bd.batch, bd.y
            logit = model(bd.x, bd.ss_emb, bd.edge_index, b)
            loss = criterion(logit, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss)
        sched.step()
        vp, vy = predict(model, val_chains)
        for thr in (0.5,):
            m = matthews_corrcoef(vy, (vp >= thr).astype(int))
            if m > best_mcc:
                best_mcc, best_th = float(m), thr
                best_state = copy.deepcopy(model.state_dict())
        if not quiet and epoch % 10 == 0:
            print(f'    seed {seed} epoch {epoch:3d} loss {running/len(train_chains):.4f} '
                  f'val_MCC {best_mcc:.4f}', flush=True)
    model.load_state_dict(best_state)


    vp, vy = predict(model, val_chains)
    grid = np.linspace(0.05, 0.95, 91)
    val_thr = float(grid[int(np.argmax([matthews_corrcoef(vy, (vp >= t).astype(int))
                                        for t in grid]))])
    return model, best_th, best_mcc, val_thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 0, 1, 2, 3])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--hidden', type=int, default=86)
    ap.add_argument('--dropout', type=float, default=0.6)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out_path = args.out or OUT
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    leak = json.load(open(os.path.join(ROOT, 'data', os.environ.get(
        'GRASSP_LEAKED', 'leaked_pairs.json')), encoding='utf-8'))
    drop_train = {to_grassp_name(x) for x in leak['remove_train_ids']}
    drop_val = {to_grassp_name(x) for x in leak['remove_val_ids']}

    raw_train = load_chains(SPLITS['train'])
    raw_val = load_chains(SPLITS['val'])
    test = {b: load_chains(b) for b in BENCH}
    print(f'raw GRASSP data: train {len(raw_train)} / val {len(raw_val)} '
          f'/ test ' + ' '.join(f'{b} {len(test[b])}' for b in BENCH))


    train_chains = [(k, v.to(DEV)) for k, v in raw_train.items() if k not in drop_train]
    val_chains = [(k, v.to(DEV)) for k, v in raw_val.items() if k not in drop_val]
    test = {b: {k: v.to(DEV) for k, v in test[b].items()} for b in BENCH}
    print(f'after de-leaking: train {len(train_chains)} / val {len(val_chains)} '
          f'(removed {len(drop_train)} / {len(drop_val)} chains)')

    out = {'protocol': 'authors\' model and hyper-parameters (hidden 86, dropout 0.6, lr 1e-4, '
                       'wd 1e-4, 100 epochs, batch 1, threshold fixed at 0.5) on the de-leaked split',
           'n_train': len(train_chains), 'n_val': len(val_chains), 'seeds': {}, 'per_chain': {}}
    for seed in args.seeds:
        print(f'  seed {seed} ...', flush=True)
        model, thr, val_mcc, val_thr = train_one(seed, train_chains, val_chains, test,
                                                 epochs=args.epochs, lr=args.lr,
                                                 hidden=args.hidden, dropout=args.dropout,
                                                 batch_size=args.batch_size, quiet=True)
        entry = {'val_mcc': val_mcc, 'threshold_fixed': thr, 'threshold_from_val': val_thr,
                 'per_set': {}}


        vp, _ = predict(model, val_chains)
        lo = 0
        for name, d in val_chains:
            out['per_chain'].setdefault('val', {}).setdefault(str(seed), {})[name] = {
                'p': vp[lo:lo + d.y.numel()].tolist(), 'y': d.y.cpu().numpy().astype(int).tolist()}
            lo += d.y.numel()
        for b in BENCH:
            p, y = predict(model, list(test[b].items()))
            entry['per_set'][b] = {'at_0.5': metrics(y, p, 0.5),
                                   'val_threshold': metrics(y, p, val_thr)}
            out['per_chain'].setdefault(b, {})[str(seed)] = {
                name: {'p': p[lo:lo + d.y.numel()].tolist(), 'y': d.y.cpu().numpy().astype(int).tolist()}
                for (name, d), lo in _offsets(list(test[b].items()), p)}
        out['seeds'][str(seed)] = entry
        print(f'    val_MCC {val_mcc:.4f} @0.5 (val-searched thr {val_thr:.2f}) | ' + ' '.join(
            f'{b}@0.5 {entry["per_set"][b]["at_0.5"]["MCC"]:.3f} '
            f'{b}@val {entry["per_set"][b]["val_threshold"]["MCC"]:.3f}' for b in BENCH), flush=True)

    json.dump(out, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'[written] {out_path}')
    return 0


def _offsets(chains, probs):
    
    lo = 0
    for name, d in chains:
        yield (name, d), lo
        lo += d.y.numel()


if __name__ == '__main__':
    raise SystemExit(main())
