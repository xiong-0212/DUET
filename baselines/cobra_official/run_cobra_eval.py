import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)

sys.path.insert(0, os.path.join(ROOT, '_dl', 'stubs'))
sys.path.insert(0, os.path.join(ROOT, '_pylibs'))
sys.path.insert(0, os.path.join(ROOT, '_dl', 'RiNALMo-main'))
sys.path.insert(0, os.path.join(ROOT, 'CoBRA', 'CoBRA-main'))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (matthews_corrcoef, roc_auc_score,
                             average_precision_score, precision_score,
                             recall_score, f1_score)

from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet
from rinalmo.model.model import RiNALMo
import rinalmo.model.rope as _rope




def _apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


_rope.apply_rotary_pos_emb = _apply_rotary_pos_emb

from CoBRA.predictor import CoBRA

DATASET_DIR = os.path.join(ROOT, 'CoBRA', 'CoBRA-main', 'CoBRA', 'Dataset')
WEIGHT = os.path.join(ROOT, '_dl', 'rinalmo-giga', 'giga-v1.pt')
COBRA_CKPT = os.path.join(ROOT, 'CoBRA', 'CoBRA-main', 'CoBRA', 'weight', 'CoBRA.pt')
MAX_LEN = 161

BENCH = ['TE18', 'TL12', 'JL10', 'RB9']



DROP_FROM_SPLIT = {'RB9': {'1nem_A', '1q8n_A', '2tob_A'}}
FILES = {
    'TE18': 'TE18/TE18-bindingsite.txt',
    'TL12': 'TL12/TL12-bindingsite.txt',
    'JL10': 'JL10/JL10-bindingsite.txt',
    'RB9': 'RB9/RB9-bindingsite.txt',
    'Hariboss_TR60': 'Hariboss_TR60/Hariboss_TR60-bindingsite.txt',
    'RNABindset': 'RNABindset/RNABindset-bindingsite.txt',
}


PAPER = {
    'RB9':  dict(MCC=0.557, AUROC=0.844, Precision=0.750, Recall=0.650),
    'TL12': dict(MCC=0.546, AUROC=0.816, Precision=0.816, Recall=0.599),
    'JL10': dict(MCC=0.293, AUROC=0.739, Precision=0.582, Recall=0.531),
    'TE18': dict(MCC=0.190, AUROC=0.647, Precision=0.562, Recall=0.357),
}

PAPER_AVG = dict(MCC=0.397, AUROC=0.762, Precision=0.678)


def parse_bindingsite(path):
    
    lines = [l.strip() for l in open(path, encoding='utf-8', errors='replace').read().split('\n')]
    lines = [l for l in lines if l]
    assert len(lines) % 3 == 0, f'{path}: {len(lines)} non-empty lines is not a multiple of 3'
    recs = []
    for i in range(0, len(lines), 3):
        hid, seq, lab = lines[i], lines[i + 1], lines[i + 2]
        assert hid.startswith('>'), f'line {i} is not a header: {hid!r}'
        assert len(seq) == len(lab), f'{hid}: sequence length {len(seq)} != label length {len(lab)}'
        recs.append(dict(id=hid[1:], seq=seq.upper(), labels=np.array([int(c) for c in lab], dtype=np.int64)))
    return recs


def load_rinalmo(device):
    cfg = model_config('giga')
    cfg.model.transformer.use_flash_attn = False
    model = RiNALMo(cfg)
    sd = torch.load(WEIGHT, map_location='cpu')
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    alphabet = Alphabet(**cfg['alphabet'])
    return model, alphabet


@torch.no_grad()
def embed_sequence(seq, rinalmo, alphabet, device):
    token_ids = alphabet.batch_tokenize([seq])
    tokens = torch.tensor(token_ids, dtype=torch.int64, device=device)
    rep = rinalmo(tokens)['representation'].squeeze(0)
    token_emb = rep[1:1 + len(seq)]
    assert token_emb.size(0) == len(seq), f'length mismatch {token_emb.size(0)} != {len(seq)}'
    padded = F.pad(token_emb, (0, 0, 0, MAX_LEN - token_emb.size(0)))
    return padded


@torch.no_grad()
def evaluate(dataset, rinalmo, alphabet, cobra, device, verbose=True):
    recs = parse_bindingsite(os.path.join(DATASET_DIR, FILES[dataset]))
    skipped = []
    y_true, y_pred, y_score = [], [], []
    t0 = time.time()
    drop = DROP_FROM_SPLIT.get(dataset, set())
    for r in recs:
        seq = r['seq']
        if r['id'] in drop:
            continue
        if len(seq) < 5 or len(seq) > MAX_LEN:
            skipped.append((r['id'], len(seq)))
            continue
        emb = embed_sequence(seq, rinalmo, alphabet, device).unsqueeze(0)
        logits, _ = cobra(emb)
        logits = logits.squeeze(0)[:len(seq)]
        prob = torch.softmax(logits, dim=-1)[:, 1]
        pred = torch.argmax(logits, dim=-1)
        y_true.append(r['labels'])
        y_pred.append(pred.cpu().numpy())
        y_score.append(prob.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_score = np.concatenate(y_score)
    res = dict(
        n_seq=len(y_true) and len([r for r in recs if 5 <= len(r['seq']) <= MAX_LEN]),
        n_residue=len(y_true),
        n_pos=int(y_true.sum()),
        MCC=matthews_corrcoef(y_true, y_pred),
        AUROC=roc_auc_score(y_true, y_score),
        AUPRC=average_precision_score(y_true, y_score),
        Precision=precision_score(y_true, y_pred, zero_division=0),
        Recall=recall_score(y_true, y_pred, zero_division=0),
        F1=f1_score(y_true, y_pred, zero_division=0),
        sec=time.time() - t0,
    )
    if skipped:
        res['skipped'] = skipped
    return res


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', device, flush=True)
    if device.type == 'cuda':
        print('gpu =', torch.cuda.get_device_name(0), flush=True)

    print('loading RiNALMo giga-v1 ...', flush=True)
    t = time.time()
    rinalmo, alphabet = load_rinalmo(device)
    print(f'  done in {time.time()-t:.1f}s', flush=True)
    print('alphabet size =', len(alphabet), ' idx_to_tkn =', alphabet.idx_to_tkn, flush=True)

    cobra = CoBRA().to(device)
    cobra.load_state_dict(torch.load(COBRA_CKPT, map_location=device))
    cobra.eval()
    print('CoBRA checkpoint loaded; params =',
          sum(p.numel() for p in cobra.parameters()), flush=True)

    out = {}
    for ds in BENCH:
        r = evaluate(ds, rinalmo, alphabet, cobra, device)
        out[ds] = r
        print(f'\n=== {ds} ===  seqs={r["n_seq"]} residues={r["n_residue"]} pos={r["n_pos"]} ({r["sec"]:.1f}s)')
        for k in ['MCC', 'AUROC', 'AUPRC', 'Precision', 'Recall', 'F1']:
            pv = PAPER.get(ds, {}).get(k)
            line = f'   {k:10s} {r[k]:.4f}'
            if pv is not None:
                line += f'   | paper {pv:.3f}   Δ {r[k]-pv:+.4f}'
            print(line)
        if 'skipped' in r:
            print('   skipped:', r['skipped'])

    dst = os.path.join(ROOT, 'cobra_repro', 'results.json')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump({'paper_M3': PAPER, 'reproduced': out}, open(dst, 'w'), indent=1)
    print('\nresults saved ->', dst)


if __name__ == '__main__':
    main()
