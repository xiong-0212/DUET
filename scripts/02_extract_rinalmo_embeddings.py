from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

RINALMO_REPO = os.environ.get('RINALMO_REPO', os.path.join(ROOT, 'third_party', 'RiNALMo-main'))
RINALMO_CKPT = os.environ.get('RINALMO_CKPT',
                              os.path.join(ROOT, 'third_party', 'rinalmo-giga', 'giga-v1.pt'))
OUT = os.path.join(ROOT, 'data', 'processed', 'rinalmo_embeddings.pt')

sys.path.insert(0, RINALMO_REPO)

import torch

import data_loader as dl
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


@torch.no_grad()
def main():
    if not os.path.isfile(RINALMO_CKPT):
        raise FileNotFoundError(
            f'RiNALMo checkpoint not found: {RINALMO_CKPT}\n'
            f'Download the RiNALMo release and set RINALMO_REPO / RINALMO_CKPT.')

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', dev, flush=True)
    cfg = model_config('giga')
    cfg.model.transformer.use_flash_attn = False
    model = RiNALMo(cfg)
    model.load_state_dict(torch.load(RINALMO_CKPT, map_location='cpu'), strict=True)
    model = model.to(dev).eval()
    alphabet = Alphabet(**cfg['alphabet'])
    print('RiNALMo loaded', flush=True)

    data = dl.load_all()
    seen, out = set(), {}
    t0 = time.time()
    for split, samples in data.items():
        for s in samples:
            rid = s['rna_id']
            if rid in seen:
                continue
            seen.add(rid)
            seq = s['sequence']
            tokens = torch.tensor(alphabet.batch_tokenize([seq]), dtype=torch.int64, device=dev)
            rep = model(tokens)['representation'].squeeze(0)
            emb = rep[1:1 + len(seq)]
            assert emb.size(0) == len(seq), f'{rid}: {emb.size(0)} != {len(seq)}'
            out[rid] = emb.half().cpu()
        print(f'  {split}: {len(out)} chains so far  {time.time() - t0:.0f}s', flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save({'embeddings': out, 'dim': int(out[list(out)[0]].shape[1])}, OUT)
    print(f'[written] {OUT}  ({len(out)} chains, {time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
