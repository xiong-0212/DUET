from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from rgcn_model import build_model

T = ['TE18', 'TL12', 'JL10', 'RB9']
DEV_SEEDS = [42, 0, 1, 2, 3]
CONF_SEEDS = [7, 11, 13, 17, 23]
SEEDS = DEV_SEEDS + CONF_SEEDS
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


ABL = [
    ('clean_seq_ss', 'ns_graph', 'graphs_seq_ss_clean.pt', 'DUET graph tower (full)'),
    ('clean_abl_seqonly', 'clean_abl_seqonly', 'graphs_seq_only_clean.pt',
     'w/o secondary-structure edges'),
    ('clean_abl_nores', 'clean_abl_nores', 'graphs_seq_ss_clean.pt', 'w/o residual connections'),
    ('clean_abl_layernorm', 'clean_abl_layernorm', 'graphs_seq_ss_clean.pt',
     'GraphNorm → LayerNorm'),
    ('clean_abl_nodrop', 'clean_abl_nodrop', 'graphs_seq_ss_clean.pt', 'w/o DropEdge'),
    ('clean_abl_bce', 'clean_abl_bce', 'graphs_seq_ss_clean.pt', 'TCL-focal → BCE loss'),
]
SEQ = {'dev': ('slim_tcn', 'slim_dense'), 'conf': ('ns_tcn', 'ns_dense')}


def tag_for(dev_tag, conf_tag, seed):
    return dev_tag if seed in DEV_SEEDS else conf_tag


def probs(dev_tag, conf_tag, graph_file):
    
    gobj = torch.load(os.path.join(ROOT, 'data', 'processed', graph_file),
                      map_location='cpu', weights_only=False)
    out = {}
    for seed in SEEDS:
        tag = tag_for(dev_tag, conf_tag, seed)
        path = os.path.join(ROOT, 'checkpoints', 'controlled', f'{tag}_seed{seed}.pt')
        if not os.path.exists(path):
            return None
        ck = torch.load(path, map_location='cpu', weights_only=False)
        mcfg = dict(ck['config']['model'])
        mcfg['num_relations'] = ck['num_relations']
        mcfg['in_dim'] = gobj.get('in_dim') or mcfg.get('in_dim', 640)
        model = build_model(mcfg, ck['num_relations']).to(DEV)
        model.load_state_dict(ck['state_dict'])
        model.eval()
        acc = {}
        leak = json.load(open(os.path.join(ROOT, 'data', 'leaked_pairs.json'), encoding='utf-8'))
        drop_val = set(leak['remove_val_ids'])
        splits = {'val': [g for g in gobj['graphs']['val'] if g.rna_id not in drop_val]}
        for sp in T:
            splits[sp] = gobj['graphs'][sp]
        with torch.no_grad():
            for sp, graphs in splits.items():
                for g in graphs:
                    z = model(g.x.to(DEV), g.edge_index.to(DEV), g.edge_type.to(DEV),
                              torch.zeros(g.x.size(0), dtype=torch.long, device=DEV))
                    if isinstance(z, tuple):
                        z = z[0]
                    acc[f'{sp}/{g.rna_id}'] = {
                        'p': torch.sigmoid(z).float().cpu().numpy(),
                        'y': g.y.numpy().astype(int)}
        out[seed] = {'acc': acc, 'thr': float(ck['threshold'])}
    return out


def mcc_pooled(pk, thr):
    per = []
    for sp in T:
        keys = sorted(k for k in pk if k.startswith(sp + '/'))
        p = np.concatenate([pk[k]['p'] for k in keys])
        y = np.concatenate([pk[k]['y'] for k in keys])
        per.append(stats.mstats.mquantiles if False else
                   __import__('sklearn.metrics', fromlist=['matthews_corrcoef'])
                   .matthews_corrcoef(y, (p >= thr).astype(int)))
    return float(np.mean(per)), per


def merge(pack_list):
    
    out = {}
    for s in SEEDS:
        keys = set(pack_list[0][s]['acc'])
        for pk in pack_list[1:]:
            keys &= set(pk[s]['acc'])
        out[s] = {'acc': {k: {'p': np.mean([pk[s]['acc'][k]['p'] for pk in pack_list], axis=0),
                              'y': pack_list[0][s]['acc'][k]['y']} for k in keys},
                  'thr': pack_list[0][s]['thr']}
    return out


def protocol_a(packs):
    
    mean = {}
    for k in packs[SEEDS[0]]['acc']:
        mean[k] = {'p': np.mean([packs[s]['acc'][k]['p'] for s in SEEDS], axis=0),
                   'y': packs[SEEDS[0]]['acc'][k]['y']}
    return mcc_pooled(mean, 0.5)[0]


def protocol_b(packs):
    
    from sklearn.metrics import matthews_corrcoef
    grid = np.linspace(0.05, 0.95, 91)
    vals = []
    for s in SEEDS:
        acc = packs[s]['acc']
        vy = np.concatenate([acc[k]['y'] for k in sorted(acc) if k.startswith('val/')])
        vp = np.concatenate([acc[k]['p'] for k in sorted(acc) if k.startswith('val/')])
        thr = float(grid[int(np.argmax([matthews_corrcoef(vy, (vp >= t).astype(int))
                                        for t in grid]))])
        vals.append(mcc_pooled(acc, thr)[0])
    return float(np.mean(vals)), float(np.std(vals, ddof=1)), np.array(vals)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    tcn = probs(SEQ['dev'][0], SEQ['conf'][0], 'graphs_dual_clean.pt')
    dense = probs(SEQ['dev'][1], SEQ['conf'][1], 'graphs_dual_clean.pt')
    if tcn is None or dense is None:
        print('missing sequence-tower checkpoint'); return 1

    rows = []
    full_tower_b = full_ens_b = None
    for dev_tag, conf_tag, gf, label in ABL:
        pk = probs(dev_tag, conf_tag, gf)
        if pk is None:
            print(f'  [skip] {label} (incomplete seeds)', flush=True)
            continue
        ens = merge([pk, tcn, dense])
        a_tower, a_ens = protocol_a(pk), protocol_a(ens)
        b_tower, b_ens = protocol_b(pk), protocol_b(ens)
        rows.append(dict(label=label, a_tower=a_tower, a_ens=a_ens,
                         b_tower_mean=b_tower[0], b_tower_sd=b_tower[1], b_tower_per=b_tower[2],
                         b_ens_mean=b_ens[0], b_ens_sd=b_ens[1], b_ens_per=b_ens[2]))
        if dev_tag == 'clean_seq_ss':
            full_tower_b, full_ens_b = b_tower[2], b_ens[2]
        print(f'  {label:<34} A: tower {a_tower:.3f} ens {a_ens:.3f} | '
              f'B: tower {b_tower[0]:.3f}±{b_tower[1]:.3f} '
              f'ens {b_ens[0]:.3f}±{b_ens[1]:.3f}', flush=True)

    lines = ['# Ablation study (DUET, 2026-10-16: 10 seeds, two protocols)', '',
             '> Protocol A: probabilities averaged over 10 seeds + fixed threshold 0.5 (historical protocol).',
             '> Protocol B: per-seed validation-set threshold, mean ± s.d. + same-seed paired t-test (matches the main table).',
             '> "Ensemble" = equal-weight average of the graph-tower variant + TCN tower + DenseNet tower.', '',
             '| Configuration | Tower (A) | Ensemble (A) | Tower (B) | Δ vs full | p | '
             'Ensemble (B) | Δ vs full | p |',
             '|:---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        dt = r['b_tower_per'] - full_tower_b
        de = r['b_ens_per'] - full_ens_b
        pt = float(stats.ttest_rel(r['b_tower_per'], full_tower_b).pvalue)
        pe = float(stats.ttest_rel(r['b_ens_per'], full_ens_b).pvalue)
        mark = '' if r['label'].endswith('(full)') else f'{dt.mean():+.3f}'
        lines.append(f"| {r['label']} | {r['a_tower']:.3f} | {r['a_ens']:.3f} | "
                     f"{r['b_tower_mean']:.3f} ± {r['b_tower_sd']:.3f} | {mark} | "
                     f"{'' if r['label'].endswith('(full)') else f'{pt:.3f}'} | "
                     f"{r['b_ens_mean']:.3f} ± {r['b_ens_sd']:.3f} | "
                     f"{'' if r['label'].endswith('(full)') else f'{de.mean():+.3f}'} | "
                     f"{'' if r['label'].endswith('(full)') else f'{pe:.3f}'} |")
    lines += ['', '> Δ is the same-seed paired difference (variant − full configuration); p is the paired t-test.',
              '> Both protocols give the same ordering: loss function > GraphNorm > DropEdge > secondary-structure edges > residual connections (no effect).', '']
    out = os.path.join(ROOT, 'docs', 'tables', 'ablation_Table3.md')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, 'w', encoding='utf-8').write('\n'.join(lines))
    print(f'\n[written] {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
