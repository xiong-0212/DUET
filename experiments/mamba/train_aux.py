from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from utils import (compute_metrics, drop_edge, find_best_threshold,
                   save_json, set_seed)
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader

import data_loader as dl
import losses as L
from rgcn_model import build_model

T = ['TE18', 'TL12', 'JL10', 'RB9']
W, STRIDE = 3, 3
BASE_IDX = {'A': 0, 'U': 1, 'G': 2, 'C': 3}


def chain_aux_tensors(batch):
    
    b = batch.batch
    N = int(b.numel())
    idx = torch.arange(N, device=b.device)

    starts = torch.zeros(int(b.max()) + 1, dtype=torch.long, device=b.device)
    first = torch.ones_like(b, dtype=torch.bool)
    first[1:] = b[1:] != b[:-1]
    starts[b[first]] = idx[first]
    pos = idx - starts[b]
    lens = torch.zeros_like(starts)
    lens.scatter_add_(0, b, torch.ones_like(b))
    has_left = pos > 0
    has_right = pos < (lens[b] - 1)
    prev_idx = torch.where(has_left, idx - 1, idx)
    next_idx = torch.where(has_right, idx + 1, idx)

    win_list, win_lab = [], []
    y = batch.y
    for ci in range(len(lens)):
        m = b == ci
        i0 = int(starts[ci])
        Lc = int(lens[ci])
        st = list(range(0, max(Lc - W + 1, 1), STRIDE))
        if Lc >= W and st and st[-1] + W < Lc:
            st.append(Lc - W)
        if not st:
            continue
        mem = torch.tensor([[i0 + s + k for k in range(W)] for s in st],
                           dtype=torch.long, device=b.device)
        lab = torch.tensor([1.0 if y[i0 + s:i0 + s + W].sum() > 0 else 0.0
                            for s in st], device=b.device)
        win_list.append(mem)
        win_lab.append(lab)
    win_mem = torch.cat(win_list, 0) if win_list else torch.zeros(0, W, dtype=torch.long,
                                                                 device=b.device)
    win_label = torch.cat(win_lab, 0) if win_lab else torch.zeros(0, device=b.device)
    return dict(win_mem=win_mem, win_label=win_label, has_left=has_left,
                has_right=has_right, prev_idx=prev_idx, next_idx=next_idx,
                base=batch.base)


@torch.no_grad()
def collect(model, loader, device, tau=0.5):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_type,
                    getattr(batch, 'batch', None))
        logit = out[0] if isinstance(out, tuple) else out
        probs.append(torch.sigmoid(logit).float().cpu().numpy())
        labels.append(batch.y.cpu().numpy())
    return np.concatenate(labels), np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(ROOT, 'configs', 'controlled.yaml'))
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--graphs', default='data/processed/graphs_seq_ss_clean.pt')
    ap.add_argument('--arch', default='dualview')
    ap.add_argument('--fine_loss', default='bce', choices=['bce', 'tcl_focal'])
    ap.add_argument('--lam_win', type=float, default=1.0)
    ap.add_argument('--lam_aux', type=float, default=0.3)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding='utf-8'))
    cfg['model']['arch'] = a.arch
    cfg['model']['use_aux'] = a.arch in ('dualview', 'fusion', 'fusion_avg',
                                         'fusion_concat')
    set_seed(a.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gp = a.graphs if os.path.isabs(a.graphs) else os.path.join(ROOT, a.graphs)
    gobj = torch.load(gp, map_location='cpu', weights_only=False)
    graphs = gobj['graphs']
    leaked = json.load(open(os.path.join(ROOT, 'data', 'leaked_pairs.json'),
                            encoding='utf-8'))
    rm_val = set(leaked['remove_val_ids'])
    val_g = [g for g in graphs['val'] if g.rna_id not in rm_val]


    seqs = {s['rna_id']: s['sequence']
            for recs in dl.load_all().values() for s in recs}
    for split in ['train', 'val'] + T:
        for g in graphs[split]:
            seq = seqs.get(g.rna_id, '')
            base = np.full(g.y.shape[0], -100, dtype=np.int64)
            for i, c in enumerate(seq[:g.y.shape[0]]):
                base[i] = BASE_IDX.get(c.upper(), -100)
            g.base = torch.from_numpy(base)

    mcfg = dict(cfg['model'])
    mcfg['in_dim'] = gobj.get('in_dim') or 640
    mcfg['num_relations'] = gobj['num_relations']
    model = build_model(mcfg, mcfg['num_relations']).to(dev)

    bs = cfg['train']['batch_size']
    gtr = torch.Generator().manual_seed(a.seed)
    train_loader = DataLoader(graphs['train'], batch_size=bs, shuffle=True,
                              generator=gtr)
    val_loader = DataLoader(val_g, batch_size=bs, shuffle=False)

    crit = None
    if a.fine_loss == 'tcl_focal':
        n_pos = sum(int((g.y == 1).sum()) for g in graphs['train'])
        n_neg = sum(int((g.y == 0).sum()) for g in graphs['train'])
        crit = L.build_loss('tcl_focal', counts=(n_neg, n_pos),
                            feat_dim=getattr(model, 'feat_dim', 64)).to(dev)
    else:
        n_pos = float(sum((g.y == 1).sum() for g in graphs['train']))
        n_neg = float(sum((g.y == 0).sum() for g in graphs['train']))
        pw = torch.tensor([max(n_neg / max(n_pos, 1.0), 3.0)], device=dev)

    params = list(model.parameters()) + (list(crit.parameters()) if crit else [])
    opt = torch.optim.AdamW(params, lr=cfg['train']['lr'],
                            weight_decay=cfg['train']['weight_decay'])
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=cfg['train']['T_0'], T_mult=cfg['train']['T_mult'],
        eta_min=cfg['train']['eta_min'])

    if not a.quiet:
        print(f'[{a.tag}/seed{a.seed}] arch={a.arch} fine={a.fine_loss} '
              f'λ_win={a.lam_win} λ_aux={a.lam_aux} params='
              f'{sum(p.numel() for p in model.parameters())}', flush=True)

    best = {'mcc': -2.0, 'state': None, 'epoch': -1}
    patience = 0
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed)
    hist = []
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = nb = 0
        for batch in train_loader:
            batch = batch.to(dev)
            ei, et = drop_edge(batch.edge_index, batch.edge_type,
                               cfg['train']['drop_edge_p'], generator=gen)
            logit, feat, aux_logits, h_bi = model(
                batch.x, ei, et, getattr(batch, 'batch', None), return_aux=True)
            y = batch.y.float()
            if crit is not None:
                l_fine, _ = crit(logit, y, feat)
            else:
                l_fine = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw)
            aux = chain_aux_tensors(batch)
            l_win = torch.tensor(0.0, device=dev)
            if a.lam_win > 0 and aux['win_mem'].numel():
                Zw = a.tau * torch.logsumexp(
                    logit[aux['win_mem']].view(-1, W) / a.tau, dim=1)
                l_win = F.binary_cross_entropy_with_logits(Zw, aux['win_label'])
            l_aux = torch.tensor(0.0, device=dev)
            if a.lam_aux > 0 and aux_logits is not None:

                valid = aux['has_left'] & aux['has_right'] & (aux['base'] >= 0)
                if valid.any():
                    l_aux = F.cross_entropy(aux_logits[valid], aux['base'][valid])
            loss = l_fine + a.lam_win * l_win + a.lam_aux * l_aux
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
            opt.step()
            tot += float(loss)
            nb += 1
        sch.step()
        yv, pv = collect(model, val_loader, dev)
        vmcc = compute_metrics(yv, pv, 0.5)['MCC']
        hist.append({'epoch': ep, 'loss': tot / max(nb, 1), 'val_MCC': vmcc})
        if vmcc > best['mcc']:
            best.update(mcc=vmcc, epoch=ep,
                        state={k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()})
            patience = 0
        else:
            patience += 1
        if not a.quiet and (ep % 10 == 0 or ep == 1):
            print(f'  ep{ep:3d} loss={tot/max(nb,1):.4f} val_MCC={vmcc:.4f} '
                  f'patience={patience}', flush=True)
        if patience >= cfg['train']['early_stop_patience']:
            break

    model.load_state_dict(best['state'])
    yv, pv = collect(model, val_loader, dev)
    th, _ = find_best_threshold(yv, pv, 'mcc')
    test = {}
    for split in T:
        yt, pt = collect(model, DataLoader(graphs[split], batch_size=bs,
                                           shuffle=False), dev)
        test[split] = {'at_threshold': compute_metrics(yt, pt, th),
                       'at_0.5': compute_metrics(yt, pt, 0.5),
                       'n_residues': int(len(yt))}
    avg = float(np.mean([test[s]['at_threshold']['MCC'] for s in T]))
    res = {'tag': a.tag, 'seed': a.seed, 'preset': gobj.get('preset'),
           'graphs_file': gp, 'arch': a.arch, 'fine_loss': a.fine_loss,
           'lam_win': a.lam_win, 'lam_aux': a.lam_aux, 'tau': a.tau,
           'config': cfg, 'best_epoch': best['epoch'], 'threshold': th,
           'val_metrics': compute_metrics(yv, pv, th), 'test_metrics': test,
           'history': hist}
    save_json(res, os.path.join(ROOT, 'results', 'controlled',
                                f'train_{a.tag}_seed{a.seed}.json'))
    print(f'[{a.tag}/seed{a.seed}] best_ep={best["epoch"]} th={th:.2f} '
          f'4-set avg={avg:.4f}  '
          + ' '.join(f'{s}:{test[s]["at_threshold"]["MCC"]:.3f}' for s in T))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
