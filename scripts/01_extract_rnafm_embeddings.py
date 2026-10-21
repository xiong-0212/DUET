from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import torch

import data_loader as dl

EMB_DIM = 640
RNAFM_REPO = os.environ.get('RNAFM_REPO', os.path.join(ROOT, 'third_party', 'RNA-FM'))
RNAFM_CKPT = os.environ.get('RNAFM_CKPT',
                            os.path.join(RNAFM_REPO, 'pretrained', 'RNA-FM_pretrained.pth'))
EMB_OUT = os.path.join(ROOT, 'data', 'processed', 'embeddings.pt')


def load_rna_fm(repo=RNAFM_REPO, ckpt=RNAFM_CKPT, device='cpu'):
    if repo not in sys.path:
        sys.path.insert(0, repo)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f'RNA-FM checkpoint not found: {ckpt}\n'
            f'Download the RNA-FM release and set RNAFM_REPO / RNAFM_CKPT.')
    import fm
    model, alphabet = fm.pretrained.rna_fm_t12(ckpt)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, alphabet


@torch.no_grad()
def embed_sequences(sequences, model, alphabet, device='cpu', max_len=1024, progress=True):
    
    bc = alphabet.get_batch_converter()
    out = {}
    t0 = time.time()
    for n, (rid, seq) in enumerate(sequences, 1):
        if len(seq) > max_len:
            seq = seq[:max_len]
        _, _, toks = bc([(rid, seq)])
        toks = toks.to(device)
        rep = model(toks, repr_layers=[12], return_contacts=False)['representations'][12]
        emb = rep[0, 1:len(seq) + 1]
        if emb.size(0) != len(seq):
            raise RuntimeError(f'{rid}: embedding length {emb.size(0)} != sequence length {len(seq)}')
        out[rid] = emb.half().cpu()
        if progress and (n % 100 == 0 or n == len(sequences)):
            print(f'  {n}/{len(sequences)} chains  {time.time() - t0:.1f}s', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=EMB_OUT)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', device, flush=True)

    data = dl.load_all()
    seen, seqs = set(), []
    for split, samples in data.items():
        for s in samples:
            if s['rna_id'] in seen:
                continue
            seen.add(s['rna_id'])
            seqs.append((s['rna_id'], s['sequence']))
    print(f'{len(seqs)} unique sequences to embed', flush=True)

    model, alphabet = load_rna_fm(device=device)
    print('RNA-FM parameters =', sum(p.numel() for p in model.parameters()), flush=True)

    embs = embed_sequences(seqs, model, alphabet, device=device)
    bad = [k for k, v in embs.items() if v.shape[1] != EMB_DIM]
    if bad:
        raise RuntimeError(f'unexpected embedding dimension: {bad[:3]}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(embs, args.out)
    print(f'[written] {args.out}  ({len(embs)} chains, {EMB_DIM}-d, fp16)')


if __name__ == '__main__':
    main()
