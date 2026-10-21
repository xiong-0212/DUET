from __future__ import annotations

import difflib
import json
import os
import sys
import time

import numpy as np
import torch
from Bio import Align

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import data_loader as dl
from rgcn_model import build_model
from utils import compute_metrics

CACHE = os.path.join(ROOT, 'results', 'homology_identity.json')
OUT_JSON = os.path.join(ROOT, 'results', 'homology_audit.json')
T = ['TE18', 'TL12', 'JL10', 'RB9']
SEEDS = [42, 0, 1, 2, 3]
MIN_LEN_RATIO = 0.8

aligner = Align.PairwiseAligner()
aligner.mode = 'global'
aligner.match_score = 1.0
aligner.mismatch_score = -1.0
aligner.open_gap_score = -4.0
aligner.extend_gap_score = -1.0


def identity(a: str, b: str) -> float:
    
    la, lb = len(a), len(b)
    if min(la, lb) / max(la, lb) < MIN_LEN_RATIO:
        return float('nan')
    try:
        aln = aligner.align(a, b)[0]
    except Exception:
        return float('nan')
    matches = 0
    for (sa, ea), (sb, eb) in zip(aln.aligned[0], aln.aligned[1]):
        sub_a, sub_b = a[sa:ea], b[sb:eb]
        matches += sum(1 for x, y in zip(sub_a, sub_b) if x == y)
    return matches / min(la, lb)


def build_matrix():
    if os.path.exists(CACHE):
        print(f'[cache] reusing {CACHE}', flush=True)
        out = json.load(open(CACHE, encoding='utf-8'))
        return _add_exact(out)
    data = dl.load_all()
    train = [(s['rna_id'], s['sequence']) for s in data['train']]
    q = {}
    for split in ['val'] + T:
        q[split] = [(s['rna_id'], s['sequence']) for s in data[split]]
    out = {'train_ids': [r for r, _ in train], 'queries': {}}
    t0 = time.time()
    n_done = 0
    for split, recs in q.items():
        out['queries'][split] = {}
        for rid, seq in recs:
            best, best_rid, n_valid = 0.0, None, 0
            for trid, tseq in train:
                v = identity(seq, tseq)
                if np.isnan(v):
                    continue
                n_valid += 1
                if v > best:
                    best, best_rid = v, trid
            pid = best if n_valid else float('nan')
            best_difflib = (difflib.SequenceMatcher(
                None, seq, dict(train)[best_rid]).ratio() if best_rid else float('nan'))
            out['queries'][split][rid] = dict(pid=round(pid, 4), to=best_rid,
                                              difflib=round(best_difflib, 4),
                                              len=len(seq))
            n_done += 1
            if n_done % 10 == 0:
                print(f'  {n_done} query chains done  {time.time()-t0:.0f}s', flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(out, open(CACHE, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'[written] identity matrix cached to {CACHE} ({time.time()-t0:.0f}s)', flush=True)
    return _add_exact(out)


def _add_exact(out):
    
    data = dl.load_all()
    tr = {s['rna_id']: s['sequence'] for s in data['train']}
    exact = {}
    for split in ['val'] + T:
        for s in data[split]:
            exact[s['rna_id']] = [t for t, seq in tr.items()
                                  if len(seq) == len(s['sequence'])
                                  and seq == s['sequence']]
    out['exact_equal'] = exact
    return out


@torch.no_grad()
def per_chain_mcc(ckpt_tag, split, graph_file):
    
    gobj = torch.load(os.path.join(ROOT, 'data', 'processed', graph_file),
                      map_location='cpu', weights_only=False)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    acc = {}
    for seed in SEEDS:
        ck = torch.load(os.path.join(ROOT, 'checkpoints', 'controlled',
                                     f'{ckpt_tag}_seed{seed}.pt'),
                        map_location='cpu', weights_only=False)
        mcfg = dict(ck['config']['model'])
        mcfg['num_relations'] = ck['num_relations']
        model = build_model(mcfg, ck['num_relations']).to(dev)
        model.load_state_dict(ck['state_dict'])
        model.eval()
        th = ck['threshold']
        for g in gobj['graphs'][split]:
            z = model(g.x.to(dev), g.edge_index.to(dev), g.edge_type.to(dev),
                      torch.zeros(g.x.size(0), dtype=torch.long, device=dev))
            p = torch.sigmoid(z).float().cpu().numpy()
            y = g.y.numpy().astype(int)
            if len(y) < 2 or len(np.unique(y)) < 2:
                continue
            acc.setdefault(g.rna_id, []).append(compute_metrics(y, p, th)['MCC'])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    mat = build_matrix()
    print('\n=== Maximum identity of test chains against the training set ===', flush=True)
    groups = {}
    for split in T:
        recs = mat['queries'][split]
        pids = np.array([v['pid'] for v in recs.values()])
        for thr, name in [(0.999, '100%'), (0.95, '≥95%'), (0.90, '≥90%'),
                          (0.80, '≥80%'), (0.70, '≥70%')]:
            groups.setdefault(name, {'hit': 0, 'total': 0})
        n = len(pids)
        print(f'{split}: n={n}  100%={int((pids>=0.999).sum())}  '
              f'≥95%={int((pids>=0.95).sum())}  ≥90%={int((pids>=0.90).sum())}  '
              f'≥80%={int((pids>=0.80).sum())}  ≥70%={int((pids>=0.70).sum())}  '
              f'min={pids.min():.2f}')
    allp = np.array([v['pid'] for split in T for v in mat['queries'][split].values()])
    summary = {name: dict(hit=int((allp >= thr).sum()), total=int(len(allp)))
               for thr, name in [(0.999, '100%'), (0.95, '≥95%'), (0.90, '≥90%'),
                                 (0.80, '≥80%'), (0.70, '≥70%')]}

    n_exact = sum(1 for split in T for rid in mat['queries'][split]
                  if mat['exact_equal'].get(rid))
    print(f'Exact identical sequences (equal length and fully identical): {n_exact}/49 test chains '
          f'(ledger §3.1 records 16/49; the "containment-style 100%" criterion of this script '
          f'also counts short training chains matched by containment)')
    summary['exact_equal'] = dict(hit=int(n_exact), total=49)
    print(f'\nTotal {len(allp)} test chains: ' + '  '.join(
        f"{k}={v['hit']}/{v['total']}" for k, v in summary.items()))


    print('\n=== Stratified MCC (mean over 5 seeds, per chain) ===', flush=True)
    strata = {}
    for tag, label, gf in [('clean_seq_ss', 'seq+ss', 'graphs_seq_ss_clean.pt'),
                           ('clean_mlp', 'MLP', 'graphs_seq_ss_clean.pt')]:
        mccs = {}
        for split in T:
            mccs.update(per_chain_mcc(tag, split, gf))
        strata[label] = mccs
    def bucket(pid):
        return ('100%' if pid >= 0.999 else '95-99%' if pid >= 0.95 else
                '90-94%' if pid >= 0.90 else '80-89%' if pid >= 0.80 else
                '70-79%' if pid >= 0.70 else '<70%')
    table = {}
    for split in T:
        for rid, v in mat['queries'][split].items():
            b = bucket(v['pid'])
            table.setdefault(b, {'n': 0, 'seq+ss': [], 'MLP': [], 'pids': []})
            table[b]['n'] += 1
            table[b]['pids'].append(v['pid'])
            for label in ('seq+ss', 'MLP'):
                if rid in strata[label]:
                    table[b][label].append(strata[label][rid])
    for b in ['100%', '95-99%', '90-94%', '80-89%', '70-79%', '<70%']:
        if b not in table:
            continue
        row = table[b]
        s = np.array(row['seq+ss']) if row['seq+ss'] else np.array([np.nan])
        m = np.array(row['MLP']) if row['MLP'] else np.array([np.nan])
        print(f'  {b:<8} n={row["n"]:<3} seq+ss={np.nanmean(s):+.3f}  '
              f'MLP={np.nanmean(m):+.3f}  Δ={np.nanmean(s)-np.nanmean(m):+.3f}')
        row['seq+ss_mean'] = float(np.nanmean(s))
        row['MLP_mean'] = float(np.nanmean(m))

    out = {'summary': summary, 'per_chain': mat['queries'], 'strata': table,
           'per_chain_mcc': strata, 'min_len_ratio': MIN_LEN_RATIO}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f'\n[written] {OUT_JSON}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
