from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from Bio import Align
from scipy.stats import ttest_rel
from sklearn.metrics import matthews_corrcoef

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from rgcn_model import build_model
import data_loader as dl

DEV_SEEDS = [42, 0, 1, 2, 3]
CONF_SEEDS = [7, 11, 13, 17, 23]
SEEDS = DEV_SEEDS + CONF_SEEDS
T = ['TE18', 'TL12', 'JL10', 'RB9']
EXT = 'hariofficial_test'
DEV = 'cpu'
THR_GRID = np.linspace(0.05, 0.95, 91)
MEMBERS = {s: (('clean_seq_ss', 'slim_tcn', 'slim_dense') if s in DEV_SEEDS
               else ('ns_graph', 'ns_tcn', 'ns_dense')) for s in SEEDS}
GRAPH = {'clean_seq_ss': 'graphs_seq_ss_clean.pt', 'ns_graph': 'graphs_seq_ss_clean.pt',
         'slim_tcn': 'graphs_dual_clean.pt', 'ns_tcn': 'graphs_dual_clean.pt',
         'slim_dense': 'graphs_dual_clean.pt', 'ns_dense': 'graphs_dual_clean.pt'}
OUT = os.path.join(ROOT, 'results', 'external_split.json')

aligner = Align.PairwiseAligner()
aligner.mode = 'global'
aligner.match_score = 1.0
aligner.mismatch_score = -1.0
aligner.open_gap_score = -4.0
aligner.extend_gap_score = -1.0


def identity(a, b):
    la, lb = len(a), len(b)
    if min(la, lb) / max(la, lb) < 0.8:
        return float('nan')
    try:
        aln = aligner.align(a, b)[0]
    except Exception:
        return float('nan')
    m = 0
    for (sa, ea), (sb, eb) in zip(aln.aligned[0], aln.aligned[1]):
        m += sum(1 for x, y in zip(a[sa:ea], b[sb:eb]) if x == y)
    return m / min(la, lb)


def homology_part():
    data = dl.load_all()
    ref = {s['rna_id']: s['sequence'] for s in data['train']} | \
          {s['rna_id']: s['sequence'] for s in data['val']}
    ext = {s['rna_id']: s['sequence'] for s in data[EXT]}
    bench = {s['rna_id']: s['sequence'] for sp in T for s in data[sp]}
    out = {'n_chains': len(ext), 'vs_train_val': {}, 'vs_benchmarks': {}}
    for label, other in [('vs_train_val', ref), ('vs_benchmarks', bench)]:
        hits = {90: 0, 80: 0, 70: 0, 50: 0}
        detail = {}
        for rid, seq in ext.items():
            best, best_to = 0.0, None
            for oid, oseq in other.items():
                v = identity(seq, oseq)
                if not np.isnan(v) and v > best:
                    best, best_to = v, oid
            detail[rid] = {'pid': round(best, 4), 'to': best_to}
            for thr in hits:
                if best >= thr / 100:
                    hits[thr] += 1
        out[label] = {'hits': {f'p{k}': v for k, v in hits.items()},
                      'n_exact': sum(1 for v in detail.values() if v['pid'] == 1.0),
                      'detail': detail}
        print(f'  {label}: exact={out[label]["n_exact"]} ' +
              ', '.join(f'>={k}%: {v}' for k, v in hits.items()), flush=True)
    return out


def pack(tag, seed):
    gf = GRAPH[tag]
    gobj = torch.load(os.path.join(ROOT, 'data', 'processed', gf), map_location='cpu',
                      weights_only=False)
    leak = json.load(open(os.path.join(ROOT, 'data', 'leaked_pairs.json'), encoding='utf-8'))
    drop = set(leak['remove_val_ids'])
    splits = {'val': [g for g in gobj['graphs']['val'] if g.rna_id not in drop],
              EXT: gobj['graphs'][EXT]}
    ck = torch.load(os.path.join(ROOT, 'checkpoints', 'controlled', f'{tag}_seed{seed}.pt'),
                    map_location='cpu', weights_only=False)
    cfg = dict(ck['config']['model'])
    cfg['num_relations'] = ck['num_relations']
    cfg['in_dim'] = gobj['in_dim']
    model = build_model(cfg, ck['num_relations']).to(DEV)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    acc = {}
    with torch.no_grad():
        for split, graphs in splits.items():
            for g in graphs:
                z = model(g.x.to(DEV), g.edge_index.to(DEV), g.edge_type.to(DEV),
                          torch.zeros(g.x.size(0), dtype=torch.long, device=DEV))
                if isinstance(z, tuple):
                    z = z[0]
                acc[f'{split}/{g.rna_id}'] = {'p': torch.sigmoid(z).float().numpy(),
                                              'y': g.y.numpy().astype(int)}
    return {'acc': acc}


def cat(acc, split):
    ks = sorted(k for k in acc if k.startswith(split + '/'))
    return (np.concatenate([acc[k]['y'] for k in ks]),
            np.concatenate([acc[k]['p'] for k in ks]))


def evaluate_part():
    res = {}
    for s in SEEDS:
        members = [pack(t, s) for t in MEMBERS[s]]
        acc = {k: {'p': np.mean([m['acc'][k]['p'] for m in members], axis=0),
                   'y': members[0]['acc'][k]['y']} for k in members[0]['acc']}
        vy, vp = cat(acc, 'val')
        thr = float(THR_GRID[int(np.argmax([matthews_corrcoef(vy, (vp >= t).astype(int))
                                            for t in THR_GRID]))])
        y, p = cat(acc, EXT)
        res[s] = {'MCC': float(matthews_corrcoef(y, (p >= thr).astype(int))),
                  'threshold': thr, 'n_chains': len(acc) - 40}
    vals = np.array([res[s]['MCC'] for s in SEEDS])
    out = {'duet': {'mean': float(vals.mean()), 'sd': float(vals.std(ddof=1)),
                    'per_seed': {str(s): res[s]['MCC'] for s in SEEDS},
                    'thresholds': {str(s): res[s]['threshold'] for s in SEEDS}}}

    f = os.path.join(ROOT, 'results', 'cobra_m3_equiv_external.json')
    if os.path.exists(f):
        rows = json.load(open(f, encoding='utf-8'))['clean']['bce']
        bvals = []
        for r in rows:
            y = np.concatenate([np.asarray(v['label'], dtype=int)
                                for v in r['per_chain'][EXT].values()])
            p = np.concatenate([np.asarray(v['prob'], dtype=float)
                                for v in r['per_chain'][EXT].values()])
            bvals.append(float(matthews_corrcoef(y, (p >= r['threshold']).astype(int))))
        bvals = np.array(bvals)
        out['baseline'] = {'mean': float(bvals.mean()), 'sd': float(bvals.std(ddof=1)),
                           'per_seed': {str(s): float(v) for s, v in zip(SEEDS, bvals)}}
        out['paired'] = {'delta': float(vals.mean() - bvals.mean()),
                         'p': float(ttest_rel(vals, bvals).pvalue),
                         'seeds_in_favour': int((vals > bvals).sum())}
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    print('[1] homology of', EXT)
    hom = homology_part()
    print('[2] evaluating DUET checkpoints on', EXT)
    ev = evaluate_part()
    print(f'  DUET {ev["duet"]["mean"]:.4f} ± {ev["duet"]["sd"]:.4f} '
          f'({ev["duet"]["per_seed"]})')
    if 'baseline' in ev:
        print(f'  baseline {ev["baseline"]["mean"]:.4f} ± {ev["baseline"]["sd"]:.4f}  '
              f'Δ={ev["paired"]["delta"]:+.4f} p={ev["paired"]["p"]:.4f} '
              f'{ev["paired"]["seeds_in_favour"]}/10')
    else:
        print('  (baseline not evaluated yet: run scripts/06_train_baselines.py with '
              'BASELINE_OUT=cobra_m3_equiv_external.json)')
    json.dump({'homology': hom, 'evaluation': ev}, open(OUT, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'[written] {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
