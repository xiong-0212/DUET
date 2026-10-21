from __future__ import annotations

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)

PROC = os.path.join(ROOT, 'data', 'processed')
THRESHOLDS = [80, 70, 50]
SOURCES = {'graphs_seq_ss_clean.pt': 'graphs_seq_ss_clean_p{}.pt',
           'graphs_dual_clean.pt': 'graphs_dual_clean_p{}.pt'}


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    for thr in THRESHOLDS:
        leak = json.load(open(os.path.join(ROOT, 'data', f'leaked_pairs_p{thr}.json'),
                              encoding='utf-8'))
        drop_train = set(leak['remove_train_ids'])
        drop_val = set(leak['remove_val_ids'])
        for src, dst in SOURCES.items():
            gobj = torch.load(os.path.join(PROC, src), map_location='cpu', weights_only=False)
            g = gobj['graphs']
            out = dict(gobj)
            out['graphs'] = dict(g)
            out['graphs']['train'] = [x for x in g['train'] if x.rna_id not in drop_train]
            out['graphs']['val'] = [x for x in g['val'] if x.rna_id not in drop_val]
            out['strict_deleak'] = {'threshold': thr, 'source': src,
                                    'n_train': len(out['graphs']['train']),
                                    'n_val': len(out['graphs']['val'])}
            path = os.path.join(PROC, dst.format(thr))
            torch.save(out, path)
            print(f'[written] {dst.format(thr)}  train={len(out["graphs"]["train"])} '
                  f'val={len(out["graphs"]["val"])} '
                  f'(from {len(g["train"])}/{len(g["val"])})')


    print('\nconsistency check (no training/validation chain at >= the threshold identity):')
    for thr in THRESHOLDS:
        gobj = torch.load(os.path.join(PROC, f'graphs_seq_ss_clean_p{thr}.pt'),
                          map_location='cpu', weights_only=False)
        leak = json.load(open(os.path.join(ROOT, 'data', f'leaked_pairs_p{thr}.json'),
                             encoding='utf-8'))
        tr = set(x.rna_id for x in gobj['graphs']['train'])
        va = set(x.rna_id for x in gobj['graphs']['val'])
        assert not (tr & set(leak['remove_train_ids'])), f'p{thr}: train not filtered'
        assert not (va & set(leak['remove_val_ids'])), f'p{thr}: val not filtered'
        print(f'  p{thr}: train={len(tr)} val={len(va)}  OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
