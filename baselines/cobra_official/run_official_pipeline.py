from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COBRA_MAIN = os.path.join(ROOT, 'CoBRA', 'CoBRA-main')

for p in (os.path.join(ROOT, '_dl', 'stubs'),
          os.path.join(ROOT, '_pylibs'),
          COBRA_MAIN,
          os.path.join(COBRA_MAIN, 'RiNALMo')):
    sys.path.insert(0, p)

import torch
from sklearn.metrics import (matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, average_precision_score, f1_score)




import rinalmo.config as _rc
import rinalmo.pretrained as _rp
_orig_cfg = _rc.model_config


def _cfg_no_flash(name, *a, **k):
    cfg = _orig_cfg(name, *a, **k)
    if name == 'giga':
        cfg.model.transformer.use_flash_attn = False
    return cfg


_rc.model_config = _cfg_no_flash
_rp.model_config = _cfg_no_flash

from CoBRA.embedder import load_model, embed_fasta
from CoBRA.predictor import run_folder

BENCH = ['TE18', 'TL12', 'JL10', 'RB9']
DROP_FROM_SPLIT = {'RB9': {'1nem_A', '1q8n_A', '2tob_A'}}
DATASET_DIR = os.path.join(COBRA_MAIN, 'CoBRA', 'Dataset')
FILES = {s: f'{s}/{s}-bindingsite.txt' for s in BENCH}
WORK = os.path.join(HERE, 'official_run')
PAPER = {
    'RB9':  dict(MCC=0.557, AUROC=0.844, Precision=0.750, Recall=0.650),
    'TL12': dict(MCC=0.546, AUROC=0.816, Precision=0.816, Recall=0.599),
    'JL10': dict(MCC=0.293, AUROC=0.739, Precision=0.582, Recall=0.531),
    'TE18': dict(MCC=0.190, AUROC=0.647, Precision=0.562, Recall=0.357),
}


def parse_bindingsite(path):
    lines = [l.strip() for l in open(path, encoding='utf-8-sig',
                                     errors='replace').read().split('\n')]
    lines = [l for l in lines if l]
    out = []
    for i in range(0, len(lines), 3):
        out.append(dict(id=lines[i][1:], seq=lines[i + 1].upper(),
                        labels=np.array([int(c) for c in lines[i + 2]],
                                        dtype=np.int64)))
    return out


def main():
    os.makedirs(WORK, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, alphabet = load_model(device)
    print('official RiNALMo loaded', flush=True)

    results = {}
    for ds in BENCH:
        recs = parse_bindingsite(os.path.join(DATASET_DIR, FILES[ds]))
        drop = DROP_FROM_SPLIT.get(ds, set())
        recs = [r for r in recs if r['id'] not in drop]
        recs = [r for r in recs if 5 <= len(r['seq']) <= 161]
        ds_dir = os.path.join(WORK, ds)
        fasta = os.path.join(ds_dir, 'input.fasta')
        os.makedirs(ds_dir, exist_ok=True)
        with open(fasta, 'w', newline='\n') as f:
            for r in recs:
                f.write(f'>{r["id"]}\n{r["seq"]}\n')
        emb_dir = os.path.join(ds_dir, 'emb')
        csv_path = os.path.join(ds_dir, 'pred.csv')

        embed_fasta(fasta, emb_dir, model, alphabet)
        run_folder(emb_dir, csv_path, device)

        lab = {r['id']: r['labels'] for r in recs}
        Y, P, PRED = [], [], []
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                rid = row['ID']
                bits = np.array([int(c) for c in row['BINDING_SITE']], dtype=int)
                y = lab[rid]
                assert len(bits) == len(y), f'{rid}: prediction length mismatch'
                Y.append(y); PRED.append(bits)
        y = np.concatenate(Y); pred = np.concatenate(PRED)
        m = dict(n_chain=len(Y), n_residue=int(len(y)), n_pos=int(y.sum()),
                 MCC=float(matthews_corrcoef(y, pred)),
                 Precision=float(precision_score(y, pred, zero_division=0)),
                 Recall=float(recall_score(y, pred, zero_division=0)),
                 F1=float(f1_score(y, pred, zero_division=0)))
        results[ds] = m
        pv = PAPER[ds]
        print(f'\n=== {ds} (official pipeline, {len(Y)} de-duplicated chains / {m["n_residue"]} residues) ===')
        for k in ('MCC', 'Precision', 'Recall'):
            print(f'   {k:<10} {m[k]:.4f}   paper {pv[k]:.3f}   Δ {m[k]-pv[k]:+.4f}')
    avg = float(np.mean([results[s]['MCC'] for s in BENCH]))
    print(f'\n4-benchmark mean MCC = {avg:.4f}   paper 0.396   Δ {avg-0.396:+.4f}')
    with open(os.path.join(HERE, 'official_pipeline_results.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'results': results, 'avg_MCC': avg, 'paper': PAPER,
                   'note': 'official code end to end (including AMP embedding); test sets de-duplicated'},
                  f, ensure_ascii=False, indent=1)
    print('[written] cobra_repro/official_pipeline_results.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
