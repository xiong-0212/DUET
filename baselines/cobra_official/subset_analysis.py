from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
for p in (os.path.join(ROOT, '_dl', 'stubs'), os.path.join(ROOT, '_pylibs'),
          os.path.join(ROOT, '_dl', 'RiNALMo-main'), HERE,
          os.path.join(ROOT, 'CoBRA', 'CoBRA-main')):
    sys.path.insert(0, p)

from run_cobra_eval import (BENCH, DATASET_DIR, FILES, MAX_LEN, PAPER,
                            load_rinalmo, parse_bindingsite, _apply_rotary_pos_emb)
import rinalmo.model.rope as _rope
_rope.apply_rotary_pos_emb = _apply_rotary_pos_emb
from CoBRA.predictor import CoBRA

CKPT = os.path.join(ROOT, 'CoBRA', 'CoBRA-main', 'CoBRA', 'weight', 'CoBRA.pt')
LEAKED = os.path.join(ROOT, 'PRISM', 'data', 'leaked_pairs.json')


@torch.no_grad()
def collect(model, alphabet, cobra, device):
    out = {}
    for ds in BENCH:
        recs = parse_bindingsite(os.path.join(DATASET_DIR, FILES[ds]))
        per = {}
        for r in recs:
            seq = r['seq']
            if len(seq) < 5 or len(seq) > MAX_LEN:
                continue
            tokens = torch.tensor(alphabet.batch_tokenize([seq]), dtype=torch.int64,
                                  device=device)
            rep = model(tokens)['representation'].squeeze(0)
            emb = rep[1:1 + len(seq)]
            padded = F.pad(emb, (0, 0, 0, MAX_LEN - emb.size(0))).unsqueeze(0)
            logits, _ = cobra(padded)
            logits = logits.squeeze(0)[:len(seq)]
            per[r['id']] = dict(prob=torch.softmax(logits, dim=-1)[:, 1].cpu().numpy(),
                                label=r['labels'])
        out[ds] = per
    return out


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    return dict(MCC=float(matthews_corrcoef(y, pred)),
                Precision=float(precision_score(y, pred, zero_division=0)),
                Recall=float(recall_score(y, pred, zero_division=0)),
                AUC=float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float('nan'))


def main():
    leaked = set(json.load(open(LEAKED, encoding='utf-8'))['leak_map'].keys())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, alphabet = load_rinalmo(device)
    cobra = CoBRA().to(device)
    cobra.load_state_dict(torch.load(CKPT, map_location=device))
    cobra.eval()
    data = collect(model, alphabet, cobra, device)

    out = {}
    print(f'leaked chains (sequence identical to a training chain): {len(leaked)}')
    for ds in BENCH:
        per = data[ds]
        sub = {k: v for k, v in per.items() if k not in leaked}
        full = {k: v for k, v in per.items()}
        print(f'\n===== {ds} =====  chains {len(full)} -> {len(sub)} after de-leaking')
        res = {'n_full': len(full), 'n_clean': len(sub)}
        for name, grp in [('all', full), ('de-leaked subset', sub)]:
            y = np.concatenate([v['label'] for v in grp.values()])
            p = np.concatenate([v['prob'] for v in grp.values()])
            row = {}
            for t in [0.5]:
                m = metrics(y, p, t)
                row[t] = m
                pv = PAPER.get(ds, {})
                dm = m['MCC'] - pv.get('MCC', float('nan'))
                print(f'  {name:<12} t={t}: MCC={m["MCC"]:.3f} P={m["Precision"]:.3f} '
                      f'R={m["Recall"]:.3f} AUC={m["AUC"]:.3f}   '
                      + (f'| paper MCC={pv["MCC"]:.3f} Δ={dm:+.3f} '
                         f'paper P={pv["Precision"]:.3f}/{m["Precision"]:.3f} '
                         f'paper R={pv["Recall"]:.3f}/{m["Recall"]:.3f}'
                         if pv else ''))
            res[name] = {str(t): row[t] for t in row}
        out[ds] = res
    with open(os.path.join(HERE, 'subset_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n[written] cobra_repro/subset_analysis.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
