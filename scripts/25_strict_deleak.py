from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from Bio import Align

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import data_loader as dl

T = ['TE18', 'TL12', 'JL10', 'RB9']
THRESHOLDS = [0.90, 0.80, 0.70, 0.60, 0.50]
MIN_LEN_RATIO = 0.8
CACHE = os.path.join(ROOT, 'results', 'identity_train_to_test.json')
OUT = os.path.join(ROOT, 'results', 'strict_deleak.json')

aligner = Align.PairwiseAligner()
aligner.mode = 'global'
aligner.match_score = 1.0
aligner.mismatch_score = -1.0
aligner.open_gap_score = -4.0
aligner.extend_gap_score = -1.0


def identity(a, b):
    la, lb = len(a), len(b)
    if min(la, lb) / max(la, lb) < MIN_LEN_RATIO:
        return float('nan')
    try:
        aln = aligner.align(a, b)[0]
    except Exception:
        return float('nan')
    matches = 0
    for (sa, ea), (sb, eb) in zip(aln.aligned[0], aln.aligned[1]):
        matches += sum(1 for x, y in zip(a[sa:ea], b[sb:eb]) if x == y)
    return matches / min(la, lb)


def max_identity_to_tests(seqs, test_seqs):
    
    out = {}
    n = 0
    t0 = time.time()
    for rid, seq in seqs.items():
        best, best_to, n_valid = 0.0, None, 0
        for trid, tseq in test_seqs.items():
            v = identity(seq, tseq)
            if np.isnan(v):
                continue
            n_valid += 1
            if v > best:
                best, best_to = v, trid
        out[rid] = (best, best_to, n_valid)
        n += 1
        if n % 50 == 0:
            print(f'    {n}/{len(seqs)} chains  {time.time()-t0:.0f}s', flush=True)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    data = dl.load_all()
    train = {s['rna_id']: s['sequence'] for s in data['train']}
    val = {s['rna_id']: s['sequence'] for s in data['val']}
    tests = {sp: {s['rna_id']: s['sequence'] for s in data[sp]} for sp in T}
    test_all = {rid: seq for sp in T for rid, seq in tests[sp].items()}

    test_all = {rid: seq for rid, seq in test_all.items()}
    print(f'train={len(train)}  val={len(val)}  test={len(test_all)}')

    if os.path.exists(CACHE):
        cache = json.load(open(CACHE, encoding='utf-8'))
        print('reusing', CACHE)
        mt = {k: tuple(v) for k, v in cache['train'].items()}
        mv = {k: tuple(v) for k, v in cache['val'].items()}
        tt = {k: tuple(v) for k, v in cache['test_all'].items()}
    else:
        print('train -> test')
        mt = max_identity_to_tests(train, test_all)
        print('val -> test')
        mv = max_identity_to_tests(val, test_all)
        print('test -> test (all against all)')
        tt = max_identity_to_tests(test_all, test_all)
        json.dump({'train': {k: list(v) for k, v in mt.items()},
                   'val': {k: list(v) for k, v in mv.items()},
                   'test_all': {k: list(v) for k, v in tt.items()}},
                  open(CACHE, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('cached', CACHE)

    res = {'protocol': {'identity': 'global alignment, matches / min(len); length ratio >= 0.8',
                        'thresholds': THRESHOLDS,
                        'train': len(train), 'val': len(val), 'test': len(test_all)},
           'splits': {}}

    print(f'\n{"threshold":>10} | {"train removed":>13} | {"train left":>10} | '
          f'{"val removed":>11} | {"val left":>8}')
    for thr in THRESHOLDS:
        rem_t = sorted(k for k, v in mt.items() if v[0] >= thr)
        rem_v = sorted(k for k, v in mv.items() if v[0] >= thr)
        res['splits'][f'p{int(thr*100)}'] = {
            'threshold': thr, 'remove_train_ids': rem_t, 'remove_val_ids': rem_v,
            'n_train_left': len(train) - len(rem_t), 'n_val_left': len(val) - len(rem_v)}
        print(f'{thr:>10.2f} | {len(rem_t):>13} | {len(train) - len(rem_t):>10} | '
              f'{len(rem_v):>11} | {len(val) - len(rem_v):>8}')
        leak = {'remove_train_ids': rem_t, 'remove_val_ids': rem_v,
                'original_train_size': len(train), 'cleaned_train_size': len(train) - len(rem_t),
                'min_len_ratio': MIN_LEN_RATIO, 'threshold': thr,
                '_note': f'training/validation chains with >= {int(thr*100)}% identity to a test '
                         f'chain, built by scripts/25_strict_deleak.py'}
        json.dump(leak, open(os.path.join(ROOT, 'data', f'leaked_pairs_p{int(thr*100)}.json'),
                             'w', encoding='utf-8'), ensure_ascii=False, indent=1)


    print('\ntest-set redundancy (chain -> nearest *other* test chain):')
    redundant = {}
    for thr in THRESHOLDS:
        pairs = []
        for rid, (v, to, _) in tt.items():
            if to and to != rid and v >= thr:
                pairs.append((rid, to, round(v, 3)))
        redundant[f'p{int(thr*100)}'] = pairs
        print(f'  >= {int(thr*100)}%: {len(pairs)} directed pairs, '
              f'{len({p[0] for p in pairs})} chains involved')
    res['test_redundancy'] = redundant
    res['per_chain_train'] = {k: {'pid': round(v[0], 4), 'to': v[1]} for k, v in mt.items()}

    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n[written] {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
