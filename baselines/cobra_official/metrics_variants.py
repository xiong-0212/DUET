import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, HERE)

import numpy as np
import torch
from sklearn.metrics import (matthews_corrcoef, roc_auc_score, precision_score,
                             recall_score)

import run_cobra_eval as R

PAPER = R.PAPER


def pooled_and_macro(dataset, rinalmo, alphabet, cobra, device):
    recs = R.parse_bindingsite(os.path.join(R.DATASET_DIR, R.FILES[dataset]))
    Y, P, S = [], [], []
    with torch.no_grad():
        for r in recs:
            seq = r['seq']
            if not (5 <= len(seq) <= R.MAX_LEN):
                continue
            emb = R.embed_sequence(seq, rinalmo, alphabet, device).unsqueeze(0)
            logits, _ = cobra(emb)
            logits = logits.squeeze(0)[:len(seq)]
            Y.append(r['labels'])
            P.append(torch.argmax(logits, -1).cpu().numpy())
            S.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())

    y = np.concatenate(Y)
    p = np.concatenate(P)
    s = np.concatenate(S)
    pooled = dict(MCC=matthews_corrcoef(y, p), AUROC=roc_auc_score(y, s),
                  Precision=precision_score(y, p, zero_division=0),
                  Recall=recall_score(y, p, zero_division=0))

    keys = ['MCC', 'AUROC', 'Precision', 'Recall']
    per = {k: [] for k in keys}
    for yi, pi, si in zip(Y, P, S):
        if yi.min() == yi.max():
            continue
        per['MCC'].append(matthews_corrcoef(yi, pi))
        per['AUROC'].append(roc_auc_score(yi, si))
        per['Precision'].append(precision_score(yi, pi, zero_division=0))
        per['Recall'].append(recall_score(yi, pi, zero_division=0))
    macro = {k: float(np.mean(v)) for k, v in per.items()}
    return pooled, macro


def main():
    device = torch.device('cpu')
    rinalmo, alphabet = R.load_rinalmo(device)
    from CoBRA.predictor import CoBRA
    cobra = CoBRA()
    cobra.load_state_dict(torch.load(R.COBRA_CKPT, map_location=device))
    cobra.eval()

    for ds in ['TE18', 'TL12', 'JL10', 'RB9']:
        pooled, macro = pooled_and_macro(ds, rinalmo, alphabet, cobra, device)
        print(f'\n=== {ds} ===')
        print(f'{"Metric":<10s}{"Paper":>9s}{"Pooled":>14s}{"Δ":>9s}'
              f'{"Per-RNA mean":>12s}{"Δ":>9s}')
        for k in ['MCC', 'AUROC', 'Precision', 'Recall']:
            pv = PAPER[ds][k]
            print(f'{k:<10s}{pv:>9.3f}{pooled[k]:>14.4f}{pooled[k]-pv:>+9.4f}'
                  f'{macro[k]:>12.4f}{macro[k]-pv:>+9.4f}')


if __name__ == '__main__':
    main()
