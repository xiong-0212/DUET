from __future__ import annotations

import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from ernie_embed_rb9 import build_ernierna
import data_loader as dl

OUT = os.path.join(ROOT, 'data', 'processed', 'ernie_embeddings.pt')


@torch.no_grad()
def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, vocab = build_ernierna()
    model = model.to(dev).eval()
    print('ERNIE-RNA loaded', flush=True)
    base_map = {'A': vocab['A'], 'C': vocab['C'], 'G': vocab['G'], 'U': vocab['U'],
                'T': vocab.get('U', 3), 'N': vocab.get('N', 3)}

    data = dl.load_all()
    out, seen = {}, set()
    t0 = time.time()
    for split, samples in data.items():
        for s in samples:
            rid = s['rna_id']
            if rid in seen:
                continue
            seen.add(rid)
            seq = s['sequence'].upper()
            ids = [base_map.get(c, vocab.get('<unk>', 3)) for c in seq]
            input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
            h = model(input_ids=input_ids,
                      attention_mask=torch.ones_like(input_ids)).last_hidden_state
            emb = h.squeeze(0)[:len(seq)]
            assert emb.shape[0] == len(seq), f'{rid} length mismatch'
            out[rid] = emb.half().cpu()
        print(f'  {split}: {len(out)} chains so far  {time.time()-t0:.0f}s', flush=True)
    torch.save({'embeddings': out, 'dim': int(out[list(out)[0]].shape[1])}, OUT)
    print(f'saved {OUT}  {len(out)} chains  elapsed {time.time()-t0:.0f}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
