from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch
from scipy import stats
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from utils import compute_metrics, find_best_threshold
from rgcn_model import build_model

T = ['TE18', 'TL12', 'JL10', 'RB9']
GRAPHS = os.path.join(ROOT, 'data', 'processed', 'graphs_seq_ss_clean.pt')
CKPT_DIR = os.path.join(ROOT, 'checkpoints', 'controlled')
LEAKED = os.path.join(ROOT, 'data', 'leaked_pairs.json')
SEEDS = [42, 0, 1, 2, 3]



def ece_binary(p, y, n_bins=15):
    
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece, rows = 0.0, []
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            rows.append(dict(lo=bins[b], hi=bins[b + 1], n=0, conf=float('nan'),
                             acc=float('nan')))
            continue
        conf, acc, cnt = float(p[m].mean()), float(y[m].mean()), int(m.sum())
        ece += cnt / n * abs(conf - acc)
        rows.append(dict(lo=bins[b], hi=bins[b + 1], n=cnt, conf=conf, acc=acc))
    return float(ece), rows


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def fit_temperature(logits, y):
    
    logits, y = np.asarray(logits, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def nll(logT):
        z = logits / np.exp(logT)

        return float(np.mean(np.logaddexp(0, z) - y * z))

    r = minimize_scalar(nll, bounds=(-2.0, 3.0), method='bounded')
    return float(np.exp(r.x))


def entropy_binary(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(p * np.log(p) + (1 - p) * np.log(1 - p)))



@torch.no_grad()
def collect_logits():
    
    from torch_geometric.loader import DataLoader
    gobj = torch.load(GRAPHS, map_location='cpu', weights_only=False)
    graphs = gobj['graphs']
    remove_val = set(json.load(open(LEAKED, encoding='utf-8'))['remove_val_ids'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    splits = {'val': [g for g in graphs['val'] if g.rna_id not in remove_val]}
    for s in T:
        splits[s] = graphs[s]

    out = {}
    for seed in SEEDS:
        ck = torch.load(os.path.join(CKPT_DIR, f'clean_seq_ss_seed{seed}.pt'),
                        map_location='cpu', weights_only=False)
        mcfg = dict(ck['config']['model'])
        mcfg['num_relations'] = ck['num_relations']
        model = build_model(mcfg, ck['num_relations']).to(dev)
        model.load_state_dict(ck['state_dict'])
        model.eval()
        for split, gs in splits.items():

            rec = {'ids': [], 'logits': [], 'labels': []}
            for g in gs:
                z = model(g.x.to(dev), g.edge_index.to(dev), g.edge_type.to(dev),
                          torch.zeros(g.x.size(0), dtype=torch.long, device=dev))
                rec['ids'].append(g.rna_id)
                rec['logits'].append(z.float().cpu().numpy())
                rec['labels'].append(g.y.numpy().astype(int))
            out[(seed, split)] = rec
        print(f'seed {seed} predictions done', flush=True)
    return out



def main():
    preds, logits = {}, {}
    raw = collect_logits()
    for (seed, split), rec in raw.items():
        logits[(seed, split)] = rec['logits']
        preds[(seed, split)] = rec

    seeds = SEEDS
    result = {'per_seed': {}, 'ensemble': {}, 'chain_level': {}, 'temperature': {}}


    per_split_metrics = {k: {'uncal': [], 'cal': []} for k in T}
    val_logits, val_y = [], []
    for seed in seeds:
        v = preds[(seed, 'val')]
        vl = np.concatenate(v['logits'])
        vy = np.concatenate(v['labels'])
        thr = torch.load(os.path.join(CKPT_DIR, f'clean_seq_ss_seed{seed}.pt'),
                         map_location='cpu', weights_only=False)['threshold']
        Tt = fit_temperature(vl, vy)
        result['temperature'][seed] = Tt
        for split in T:
            r = preds[(seed, split)]
            lg = np.concatenate(r['logits'])
            yy = np.concatenate(r['labels'])
            p0 = 1 / (1 + np.exp(-lg))
            p1 = 1 / (1 + np.exp(-lg / Tt))
            m0 = compute_metrics(yy, p0, thr)
            m1 = compute_metrics(yy, p1, thr)
            e0, _ = ece_binary(p0, yy)
            e1, _ = ece_binary(p1, yy)
            per_split_metrics[split]['uncal'].append(dict(MCC=m0['MCC'], AUC=m0['AUC'],
                                                          ECE=e0, Brier=brier(p0, yy)))
            per_split_metrics[split]['cal'].append(dict(MCC=m1['MCC'], AUC=m1['AUC'],
                                                        ECE=e1, Brier=brier(p1, yy)))

        val_logits.append(vl)
        val_y.append(vy)

    for split in T:
        for kind in ('uncal', 'cal'):
            rows = per_split_metrics[split][kind]
            result['per_seed'].setdefault(kind, {})[split] = {
                k: [float(r[k]) for r in rows] for k in rows[0]}


    ens_val_p = np.mean([1 / (1 + np.exp(-v)) for v in val_logits], axis=0)
    ens_val_y = val_y[0]
    ens_thr, _ = find_best_threshold(ens_val_y, ens_val_p, 'mcc')
    result['ensemble']['threshold'] = float(ens_thr)
    for split in T:
        ps = [1 / (1 + np.exp(-np.concatenate(preds[(s, split)]['logits'])))
              for s in seeds]
        yy = np.concatenate(preds[(seeds[0], split)]['labels'])
        p = np.mean(ps, axis=0)
        m = compute_metrics(yy, p, ens_thr)
        e, _ = ece_binary(p, yy)
        result['ensemble'].setdefault('per_split', {})[split] = dict(
            MCC=m['MCC'], AUC=m['AUC'], ECE=e, Brier=brier(p, yy), n=int(len(yy)))

    result['ensemble']['single_mean_MCC'] = {
        split: float(np.mean([r['MCC'] for r in per_split_metrics[split]['uncal']]))
        for split in T}


    leak_map = json.load(open(LEAKED, encoding='utf-8'))['leak_map']
    ens_p, ch = {}, {}
    for split in T:
        ys = preds[(seeds[0], split)]['labels']
        ids = preds[(seeds[0], split)]['ids']
        for i, rid in enumerate(ids):

            p = np.mean([1 / (1 + np.exp(-np.asarray(
                preds[(s, split)]['logits'][i]).ravel())) for s in seeds], axis=0)
            y = np.asarray(ys[i]).ravel()

            mcc = acc = float('nan')
            if len(y) >= 2 and len(np.unique(y)) >= 2:
                con = compute_metrics(y, p, ens_thr)
                mcc, acc = con['MCC'], con['Accuracy']
            ch[rid] = dict(split=split, mcc=mcc, acc=acc,
                           entropy=entropy_binary(p),
                           mean_conf=float(np.mean(np.maximum(p, 1 - p))),
                           n=int(len(y)), pos=int(y.sum()),
                           leaked=rid in leak_map)
    mccs = np.array([v['mcc'] for v in ch.values()])
    ents = np.array([v['entropy'] for v in ch.values()])
    ok = np.isfinite(mccs)
    rho, pval = stats.spearmanr(ents[ok], mccs[ok])
    result['chain_level']['overall'] = dict(
        n=int(ok.sum()), spearman_rho=float(rho), p=float(pval))
    for split in T:
        sub = [v for v in ch.values() if v['split'] == split]
        e = np.array([v['entropy'] for v in sub])
        m = np.array([v['mcc'] for v in sub])
        f = np.isfinite(m)
        rr, pp = (stats.spearmanr(e[f], m[f]) if f.sum() > 3 else (np.nan, np.nan))
        result['chain_level'][split] = dict(
            n=len(sub), mean_entropy=float(e.mean()),
            mean_mcc=float(m[f].mean()), rho=float(rr), p=float(pp))
    lk = [v for v in ch.values() if v['leaked']]
    nl = [v for v in ch.values() if not v['leaked']]
    result['chain_level']['leaked_vs_not'] = {
        'leaked': dict(n=len(lk), mean_entropy=float(np.mean([v['entropy'] for v in lk])),
                       mean_mcc=float(np.mean([v['mcc'] for v in lk]))),
        'clean': dict(n=len(nl), mean_entropy=float(np.mean([v['entropy'] for v in nl])),
                      mean_mcc=float(np.mean([v['mcc'] for v in nl])))}


    sel = {}
    for split in T:
        ps = [1 / (1 + np.exp(-np.concatenate(preds[(s, split)]['logits'])))
              for s in seeds]
        p = np.mean(ps, axis=0)
        y = np.concatenate(preds[(seeds[0], split)]['labels'])
        conf = np.maximum(p, 1 - p)
        order = np.argsort(-conf)
        rows = []
        for cov in (1.0, 0.8, 0.6, 0.4, 0.2):
            k = max(int(len(y) * cov), 12)
            idx = order[:k]
            if len(np.unique(y[idx])) < 2:
                continue
            m = compute_metrics(y[idx], p[idx], ens_thr)
            rows.append(dict(coverage=cov, n=int(k), MCC=m['MCC'], Precision=m['Precision'],
                             Recall=m['Recall'], F1=m['F1']))
        sel[split] = rows

    pall = np.concatenate([np.mean([1 / (1 + np.exp(-np.concatenate(
        preds[(s, sp)]['logits']))) for s in seeds], axis=0) for sp in T])
    yall = np.concatenate([np.concatenate(preds[(seeds[0], sp)]['labels']) for sp in T])
    order = np.argsort(-np.maximum(pall, 1 - pall))
    sel['pooled'] = []
    for cov in (1.0, 0.8, 0.6, 0.4, 0.2):
        k = int(len(yall) * cov)
        idx = order[:k]
        m = compute_metrics(yall[idx], pall[idx], ens_thr)
        sel['pooled'].append(dict(coverage=cov, n=int(k), MCC=m['MCC'],
                                  Precision=m['Precision'], Recall=m['Recall'],
                                  F1=m['F1']))
    result['selective'] = sel


    payload = {'result': result, 'chains': ch,
               'reliability': {}}

    allp, ally = [], []
    for split in T:
        ps = [1 / (1 + np.exp(-np.concatenate(preds[(s, split)]['logits'])))
              for s in seeds]
        allp.append(np.mean(ps, axis=0))
        ally.append(np.concatenate(preds[(seeds[0], split)]['labels']))
    allp, ally = np.concatenate(allp), np.concatenate(ally)
    e, rows = ece_binary(allp, ally)
    payload['reliability']['ensemble'] = dict(ece=e, bins=rows)

    s0 = [np.concatenate(preds[(42, sp)]['logits']) for sp in T]
    y0 = [np.concatenate(preds[(42, sp)]['labels']) for sp in T]
    p0 = 1 / (1 + np.exp(-np.concatenate(s0)))
    y0 = np.concatenate(y0)
    e1, rows1 = ece_binary(p0, y0)
    payload['reliability']['single_uncal'] = dict(ece=e1, bins=rows1)

    with open(os.path.join(ROOT, 'results', 'calibration.json'), 'w',
              encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=float)

    print(json.dumps({k: v for k, v in result.items() if k != 'per_seed'},
                     ensure_ascii=False, indent=1, default=float))
    print('\nwrote results/calibration.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
