from __future__ import annotations
import os, torch
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
SRC = os.path.join(ROOT,'data','processed','graphs_seq_ss_clean.pt')
EMB = os.path.join(ROOT,'data','processed','ernie_official_embeddings.pt')
OUT = os.path.join(ROOT,'data','processed','graphs_ernie_clean.pt')
def main():
    emb = torch.load(EMB, map_location='cpu', weights_only=False)['embeddings']
    obj = torch.load(SRC, map_location='cpu', weights_only=False)
    ok = bad = missing = 0
    for split, graphs in obj['graphs'].items():
        for g in graphs:
            e = emb.get(g.rna_id)
            if e is None:

                g.x = torch.zeros(g.x.shape[0], 768); missing += 1; continue
            if e.shape[0] != g.x.shape[0]: bad += 1; continue
            g.x = e.float(); ok += 1
    assert bad == 0, f'{bad} chains missing ERNIE embeddings'
    obj['in_dim'] = 768; obj['node_feat_source'] = 'ernie_rna_official_grasp'
    torch.save(obj, OUT); print(f'saved {OUT} in_dim=768 graphs={ok}  missing (zeroed)={missing}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
