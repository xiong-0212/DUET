from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)


spec = importlib.util.spec_from_file_location('gd', os.path.join(HERE, 'retrain_deleaked.py'))
gd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gd)
sys.path.insert(0, os.path.join(gd.GRASSP_ROOT, 'model'))
from RNABP import HybridRNABindingSiteModel

PAPER = {'TE18': 0.359, 'TL12': 0.611, 'JL10': 0.343}
SCRIPT_THR = {'TE18': 0.7, 'TL12': 0.8, 'JL10': 0.8}


def build(ck_name):
    ck = torch.load(os.path.join(gd.GRASSP_ROOT, 'checkpoints', ck_name),
                    map_location='cpu', weights_only=False)
    hidden = ck['model_state_dict']['fuse.0.weight'].shape[0]
    model = HybridRNABindingSiteModel(rna_dim=768, ss_dim=4, hidden=hidden, dropout=0.6)
    model.load_state_dict(ck['model_state_dict'])
    model.to(gd.DEV).eval()
    return model, hidden, ck


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    grid = np.linspace(0.05, 0.95, 91)
    for ck_name in ['COBRA_SEED1.pt', 'TR60_SEED39.pt']:
        model, hidden, ck = build(ck_name)
        n_par = sum(p.numel() for p in model.parameters())
        print(f'\n=== {ck_name}  (hidden={hidden}, {n_par:,} parameters, '
              f'trained epoch {ck.get("epoch")}) ===')
        print(f"{'set':<6}{'0.5':>8}{'script thr':>12}{'val-best':>10}{'paper':>8}"
              f"{'AUC':>8}{'AUPRC':>8}")
        for b, thr in SCRIPT_THR.items():
            p, y = gd.predict(model, list(gd.load_chains(b).items()))
            mcc_05 = matthews_corrcoef(y, (p >= 0.5).astype(int))
            mcc_scr = matthews_corrcoef(y, (p >= thr).astype(int))
            best_t = float(grid[int(np.argmax([matthews_corrcoef(y, (p >= t).astype(int))
                                               for t in grid]))])
            mcc_best = matthews_corrcoef(y, (p >= best_t).astype(int))
            r = gd.metrics(y, p, thr)
            print(f'{b:<6}{mcc_05:>8.3f}{mcc_scr:>9.3f}({thr}){mcc_best:>7.3f}({best_t:.2f})'
                  f'{PAPER[b]:>8.3f}{r["AUC"]:>8.3f}{r["AUPRC"]:>8.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
