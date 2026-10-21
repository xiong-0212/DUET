from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = ROOT
for p in (os.path.join(REPO, '_dl', 'stubs'), os.path.join(REPO, '_pylibs'),
          os.path.join(REPO, '_dl', 'RiNALMo-main'), HERE,
          os.path.join(REPO, 'CoBRA', 'CoBRA-main')):
    sys.path.insert(0, p)

from run_cobra_eval import (BENCH, DATASET_DIR, FILES, MAX_LEN,
                            PAPER, load_rinalmo, parse_bindingsite)

from CoBRA.predictor import CoBRA

import rinalmo.model.rope as _rope
from run_cobra_eval import _apply_rotary_pos_emb
_rope.apply_rotary_pos_emb = _apply_rotary_pos_emb

CKPT = os.path.join(REPO, 'CoBRA', 'CoBRA-main', 'CoBRA', 'weight', 'CoBRA.pt')


@torch.no_grad()
def collect(model, alphabet, cobra, device):
    out = {}
    for ds in BENCH:
        recs = parse_bindingsite(os.path.join(DATASET_DIR, FILES[ds]))
        P, Y = [], []
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
            prob = torch.softmax(logits, dim=-1)[:, 1]
            P.append(prob.cpu().numpy())
            Y.append(r['labels'])
        out[ds] = (np.concatenate(Y), np.concatenate(P))
    return out


def sweep(y, p, thresholds):
    rows = []
    for t in thresholds:
        pred = (p >= t).astype(int)
        rows.append(dict(threshold=float(t),
                         MCC=float(matthews_corrcoef(y, pred)),
                         Precision=float(precision_score(y, pred, zero_division=0)),
                         Recall=float(recall_score(y, pred, zero_division=0))))
    return rows


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, alphabet = load_rinalmo(device)
    cobra = CoBRA().to(device)
    cobra.load_state_dict(torch.load(CKPT, map_location=device))
    cobra.eval()
    data = collect(model, alphabet, cobra, device)

    thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    out = {'paper': PAPER, 'sweep': {}, 'argmax_row': {}}
    for ds in BENCH:
        y, p = data[ds]
        rows = sweep(y, p, thresholds)
        out['sweep'][ds] = dict(
            AUC=float(roc_auc_score(y, p)), AUPRC=float(average_precision_score(y, p)),
            rows=rows, n_residue=int(len(y)), n_pos=int(y.sum()))

        pv = PAPER.get(ds)
        if pv:
            best = min(rows, key=lambda r: abs(r['MCC'] - pv['MCC']))
            out['argmax_row'][ds] = dict(
                best_match=best,
                delta_mcc=best['MCC'] - pv['MCC'],
                delta_prec=best['Precision'] - pv['Precision'],
                delta_rec=best['Recall'] - pv['Recall'])
        print(f'\n===== {ds} =====  n={len(y)} pos={int(y.sum())} '
              f'AUC={out["sweep"][ds]["AUC"]:.4f}')
        if pv:
            print(f'  paper: MCC={pv["MCC"]:.3f} P={pv["Precision"]:.3f} R={pv["Recall"]:.3f}')
            print(f'  closest threshold {best["threshold"]:.2f}: MCC={best["MCC"]:.3f} '
                  f'P={best["Precision"]:.3f} R={best["Recall"]:.3f} '
                  f'(ΔMCC={best["MCC"]-pv["MCC"]:+.3f})')
        r05 = [r for r in rows if abs(r['threshold'] - 0.5) < 1e-6][0]
        print(f'  threshold 0.50: MCC={r05["MCC"]:.3f} P={r05["Precision"]:.3f} '
              f'R={r05["Recall"]:.3f}')

        sel = [r for r in rows if 0.45 <= r['threshold'] <= 0.95]
        print('  MCC curve: ' + '  '.join(f'{r["threshold"]:.2f}:{r["MCC"]:.3f}'
                                         for r in sel))

    with open(os.path.join(HERE, 'threshold_sweep.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n[written] cobra_repro/threshold_sweep.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
