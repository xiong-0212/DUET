from __future__ import annotations

import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
SRC = os.path.join(ROOT, 'data', 'processed', 'graphs_seq_ss_clean.pt')
EMB = os.path.join(ROOT, 'data', 'processed', 'rinalmo_embeddings.pt')
OUT = os.path.join(ROOT, 'data', 'processed', 'graphs_dual_clean.pt')


def main():
    rina = torch.load(EMB, map_location='cpu', weights_only=False)['embeddings']
    obj = torch.load(SRC, map_location='cpu', weights_only=False)
    ok = bad = 0
    for split, graphs in obj['graphs'].items():
        for g in graphs:
            e = rina.get(g.rna_id)
            if e is None or e.shape[0] != g.x.shape[0]:
                bad += 1
                continue
            g.x = torch.cat([g.x.float(), e.float()], dim=-1)
            ok += 1
        print(f'  {split}: {len(graphs)} graphs', flush=True)
    assert bad == 0, f'{bad} chains with missing or length-mismatched RiNALMo embeddings'
    obj['in_dim'] = 1920
    obj['feature_layout'] = {'rnafm': [0, 640], 'rinalmo': [640, 1920]}
    obj['note'] = ('Dual-backbone feature graphs (RNA-FM 640 | RiNALMo 1280), '
                   'used for the two strong-tower fusion')
    torch.save(obj, OUT)
    print(f'[written] {OUT}  in_dim={obj["in_dim"]}  graphs={ok}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
