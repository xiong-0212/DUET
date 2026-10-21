from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from scipy.stats import ttest_rel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

SEEDS = [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]
EXT_DIR = os.path.join(ROOT, 'data', 'external')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
METRICS = ['MCC', 'AUC', 'AUPRC', 'F1', 'Precision', 'Sensitivity', 'Specificity', 'GMean']
OUT = os.path.join(ROOT, 'results', 'external_eval_baseline.json')


def gmean(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    se = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    return float(np.sqrt(se * sp))


def specificity(y, pred):
    tn, fp, _, _ = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def met(y, p, thr):
    pred = (p >= thr).astype(int)
    return {'MCC': float(matthews_corrcoef(y, pred)), 'AUC': float(roc_auc_score(y, p)),
            'AUPRC': float(average_precision_score(y, p)),
            'F1': float(f1_score(y, pred, zero_division=0)),
            'Precision': float(precision_score(y, pred, zero_division=0)),
            'Sensitivity': float(recall_score(y, pred, zero_division=0)),
            'Specificity': specificity(y, pred), 'GMean': gmean(y, pred)}


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    spec = importlib.util.spec_from_file_location(
        'base06', os.path.join(ROOT, 'scripts', '06_train_baselines.py'))
    m6 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m6)

    sets = json.load(open(os.path.join(EXT_DIR, 'external_sets.json'), encoding='utf-8'))
    emb = torch.load(os.path.join(ROOT, 'data', 'processed', 'rinalmo_embeddings.pt'),
                     map_location='cpu', weights_only=False)
    emb = emb['embeddings'] if 'embeddings' in emb else emb
    ext_emb = torch.load(os.path.join(EXT_DIR, 'emb_rinalmo.pt'), map_location='cpu',
                         weights_only=False)['embeddings']
    all_emb = {**emb, **ext_emb}


    data = m6.load_samples(True)
    res = {'per_set': {}, 'n_seeds': len(SEEDS)}
    per_seed = {name: {m: [] for m in METRICS} for name in sets}
    for s in SEEDS:
        ck_path = os.path.join(ROOT, 'checkpoints', 'baseline', f'm3equiv_bce_seed{s}.pt')
        if not os.path.exists(ck_path):
            print(f'[skip] no checkpoint for seed {s}')
            continue
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        model = m6.CoBRAHead().to(DEV)
        model.load_state_dict(ck['state_dict'])
        model.eval()

        def predict(rid, seq, labels):
            x = all_emb[rid].float().to(DEV)
            with torch.no_grad():
                z = model(x)
            return torch.sigmoid(z).float().cpu().numpy(), np.asarray(labels, dtype=int)

        yv = np.concatenate([r['y'].numpy().astype(int) for r in data['val']])
        pv = np.concatenate([predict(r['rid'], None, r['y'])[0] for r in data['val']])
        grid = np.linspace(0.05, 0.95, 91)
        thr = float(grid[int(np.argmax([matthews_corrcoef(yv, (pv >= t).astype(int))
                                        for t in grid]))])
        for name, recs in sets.items():
            ys, ps = [], []
            for cid, r in recs.items():
                if cid not in all_emb:
                    continue
                p, y = predict(cid, r['sequence'], r['labels'])
                ys.append(y)
                ps.append(p)
            y = np.concatenate(ys)
            p = np.concatenate(ps)
            mm = met(y, p, thr)
            for k in METRICS:
                per_seed[name][k].append(mm[k])
        print(f'  seed {s}: thr {thr:.2f}  '
              f'MCC time_dependent {per_seed["time_dependent"]["MCC"][-1]:.3f}', flush=True)

    for name in sets:
        if not per_seed[name]['MCC']:
            continue
        res['per_set'][name] = {m: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1))}
                                for m, v in per_seed[name].items()}
        res['per_set'][name]['n_seeds'] = len(per_seed[name]['MCC'])
    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


    ours = json.load(open(os.path.join(ROOT, 'results', 'external_eval.json'), encoding='utf-8'))
    for name in res['per_set']:
        a = np.array(per_seed[name]['MCC'])
        b = np.array(ours['per_set'][name]['MCC']['per_seed'])
        if len(a) == len(b) and len(a) > 1:
            res['per_set'][name]['paired_vs_duet'] = {
                'delta': float(b.mean() - a.mean()), 'p': float(ttest_rel(b, a).pvalue)}
    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for name, d in res['per_set'].items():
        p = d.get('paired_vs_duet', {})
        print(f"  {name}: baseline MCC {d['MCC']['mean']:.3f} ± {d['MCC']['sd']:.3f}  "
              + (f"(DUET − baseline {p['delta']:+.3f}, p={p['p']:.3f})" if p else ''))
    print(f'[written] {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
