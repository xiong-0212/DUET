from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from rgcn_model import build_model
import ss as SS

T = ['TE18', 'TL12', 'JL10', 'RB9']
SEEDS = [42, 0, 1, 2, 3]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
GF = 'graphs_dual_clean.pt'


@torch.no_grad()
def gate_of_model(model, g, device):
    
    x, ei, et = g.x.to(device), g.edge_index.to(device), g.edge_type.to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    h = x[:, :model.split_dim]
    for blk in model.graph:
        h = blk(h, ei, et, batch)
    h_g = model.proj_g(h)
    h_m = model.proj_m(model.mlp[0](x[:, model.split_dim:]))
    gate = torch.sigmoid(model.fuse_gate(torch.cat([h_g, h_m], dim=-1)))
    return gate.float().cpu().numpy()


def main():
    gobj = torch.load(os.path.join(ROOT, 'data', 'processed', GF),
                      map_location='cpu', weights_only=False)
    fold = SS.load_folds(os.path.join(ROOT, 'data', 'raw', 'ss_output.tsv'))
    bp = SS.load_bp_prob(os.path.join(ROOT, 'data', 'raw', 'ss_bp_prob.tsv'))

    recs = {}
    for seed in SEEDS:
        ck = torch.load(os.path.join(ROOT, 'checkpoints', 'controlled',
                                     f'clean_dual2_seed{seed}.pt'),
                        map_location='cpu', weights_only=False)
        mcfg = dict(ck['config']['model'])
        mcfg['num_relations'] = ck['num_relations']
        mcfg['in_dim'] = gobj['in_dim']
        model = build_model(mcfg, ck['num_relations']).to(DEV)
        model.load_state_dict(ck['state_dict'])
        model.eval()
        for split in T:
            for g in gobj['graphs'][split]:
                gate = gate_of_model(model, g, DEV)
                rec = recs.setdefault((split, g.rna_id), {'g': [], 'y': None})
                rec['g'].append(gate)
                rec['y'] = g.y.numpy().astype(int)
        print(f'seed {seed} done', flush=True)

    rows = []
    for (split, rid), rec in recs.items():
        gm = np.mean(rec['g'], axis=0)
        gbar = gm.mean(axis=1)
        y = rec['y']
        seq = ''.join('AUGC'[0:0]) or None
        db = fold.get(rid, '')
        paired = np.array([1.0 if c in '()' else 0.0 for c in db]) \
            if len(db) == len(y) else np.zeros(len(y))
        probs = bp.get(rid, {})
        conf = np.zeros(len(y))
        for (i, j), p in probs.items():
            if j < len(y):
                conf[i] = max(conf[i], p)
                conf[j] = max(conf[j], p)
        rows.append(dict(split=split, rid=rid, n=len(y),
                         g_mean=float(gbar.mean()), g_std=float(gbar.std()),
                         g_pos=float(gbar[y == 1].mean()) if (y == 1).any() else float('nan'),
                         g_neg=float(gbar[y == 0].mean()) if (y == 0).any() else float('nan'),
                         g_paired=float(gbar[paired == 1].mean()) if (paired == 1).any() else float('nan'),
                         g_unpaired=float(gbar[paired == 0].mean()) if (paired == 0).any() else float('nan'),
                         gate_dim_std=float(gm.std(axis=0).mean()),
                         _gbar=gbar.tolist(), _y=y.tolist()))


    allg = np.concatenate([np.array(r['_gbar']) for r in rows])
    ally = np.concatenate([np.array(r['_y']) for r in rows])
    summary = dict(
        n_residue=int(len(allg)),
        gate_overall_mean=float(allg.mean()), gate_overall_std=float(allg.std()),
        gate_binding_mean=float(allg[ally == 1].mean()),
        gate_nonbinding_mean=float(allg[ally == 0].mean()),

        within_chain_std_mean=float(np.mean([r['g_std'] for r in rows])),

        mean_abs_dev_from_half=float(np.mean(np.abs(allg - 0.5))),
        cross_chain_mean_std=float(np.std([r['g_mean'] for r in rows])),
        frac_gate_in_045_055=float(np.mean((allg > 0.45) & (allg < 0.55))),
    )
    from scipy import stats
    t, p = stats.ttest_ind(allg[ally == 1], allg[ally == 0], equal_var=False)
    summary['binding_vs_nonbinding_ttest_p'] = float(p)
    per_split = {}
    for s in T:
        sub = [r for r in rows if r['split'] == s]
        g = np.concatenate([np.array(r['_gbar']) for r in sub])
        y = np.concatenate([np.array(r['_y']) for r in sub])
        per_split[s] = dict(n_chain=len(sub), n_res=int(len(g)),
                            gate_mean=float(g.mean()),
                            gate_binding=float(g[y == 1].mean()) if (y == 1).any() else float('nan'),
                            gate_nonbinding=float(g[y == 0].mean()) if (y == 0).any() else float('nan'))

    out = dict(summary=summary, per_split=per_split,
               chains=[{k: v for k, v in r.items() if not k.startswith('_')} for r in rows],
               examples={r['rid']: dict(g=r['_gbar'], y=r['_y'])
                         for r in rows if r['split'] in ('RB9', 'TL12')})
    with open(os.path.join(ROOT, 'results', 'gate_analysis.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)

    print('\n=== gate statistics ===')
    for k, v in summary.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
    print('\nper test set:')
    for s, d in per_split.items():
        print(f"  {s:<6} chains={d['n_chain']:<3} gate_mean={d['gate_mean']:.3f} "
              f"binding={d['gate_binding']:.3f} nonbinding={d['gate_nonbinding']:.3f}")
    print('\nwrote results/gate_analysis.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
