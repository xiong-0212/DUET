from __future__ import annotations

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, HERE)

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from rgcn_model import build_model
from utils import compute_metrics, save_json

TEST_SPLITS = ['TE18', 'TL12', 'JL10', 'RB9', 'hariofficial_test']


def resolve_graphs_path(p):
    
    if os.path.isabs(p) and not os.path.exists(p):
        p = os.path.join('data', 'processed', os.path.basename(p))
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def load_graphs(cfg, preset, graphs_file=None):
    
    p = resolve_graphs_path(graphs_file) if graphs_file else os.path.join(
        ROOT, cfg['data']['graphs_dir'], f'graphs_{preset}.pt')
    return torch.load(p, weights_only=False)['graphs']


@torch.no_grad()
def predict(model, graphs, device, bs=8):
    model.eval()
    loader = DataLoader(graphs, batch_size=bs, shuffle=False)
    P, Y = [], []
    for batch in loader:
        batch = batch.to(device)

        ew = batch.edge_weight if hasattr(batch, 'edge_weight') else None
        logit = model(batch.x, batch.edge_index, batch.edge_type,
                      getattr(batch, 'batch', None), edge_weight=ew)
        P.append(torch.sigmoid(logit).float().cpu().numpy())
        Y.append(batch.y.cpu().numpy())
    return np.concatenate(Y), np.concatenate(P)


def build_from_ckpt(ckpt, device):
    cfg = ckpt['config']
    mcfg = dict(cfg['model'])
    mcfg['num_relations'] = ckpt['num_relations']
    model = build_model(mcfg, ckpt['num_relations']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    return model, cfg


def evaluate_ckpt(ckpt_path, device, verbose=True):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model, cfg = build_from_ckpt(ckpt, device)
    graphs = load_graphs(cfg, ckpt['preset'], ckpt.get('graphs_file'))
    th = ckpt['threshold']

    out = {'tag': ckpt['tag'], 'seed': ckpt['seed'], 'preset': ckpt['preset'],
           'threshold': th, 'val_metrics': ckpt.get('val_metrics'), 'test': {}}
    if verbose:
        print(f'\n=== {ckpt["tag"]} / seed {ckpt["seed"]} / preset {ckpt["preset"]} '
              f'(threshold {th:.2f})===')
        print(f'{"split":<20s}{"N":>6s}{"AUC":>8s}{"AUPRC":>8s}{"MCC":>8s}'
              f'{"F1":>7s}{"Prec":>7s}{"Rec":>7s}{"GMean":>7s}')
    for split in TEST_SPLITS:
        if split not in graphs:
            continue
        y, p = predict(model, graphs[split], device)
        m_t = compute_metrics(y, p, th)
        m_05 = compute_metrics(y, p, 0.5)
        out['test'][split] = {'at_threshold': m_t, 'at_0.5': m_05, 'n_residues': int(len(y))}
        if verbose:
            print(f'{split:<20s}{len(y):>6d}{m_t["AUC"]:>8.4f}{m_t["AUPRC"]:>8.4f}'
                  f'{m_t["MCC"]:>8.4f}{m_t["F1"]:>7.4f}{m_t["Precision"]:>7.4f}'
                  f'{m_t["Recall"]:>7.4f}{m_t["GMean"]:>7.4f}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--ckpt-dir', default=None)
    ap.add_argument('--tags', default=None, help='comma-separated; evaluate only these tags')
    ap.add_argument('--out', default=None)
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = []
    if a.ckpt:
        paths = [a.ckpt]
    else:
        d = a.ckpt_dir or os.path.join(ROOT, 'checkpoints')
        paths = sorted(glob.glob(os.path.join(d, '*.pt')))
        if a.tags:
            keep = set(a.tags.split(','))
            paths = [p for p in paths
                     if os.path.basename(p).rsplit('_seed', 1)[0] in keep]

    results = []
    for p in paths:
        results.append(evaluate_ckpt(p, device, verbose=not a.quiet))

    if a.out or len(results) > 1:
        out = a.out or os.path.join(ROOT, 'results', 'eval_all.json')
        save_json(results, out)
        print(f'\nsaved {len(results)} evaluation results -> {out}')


if __name__ == '__main__':
    main()
