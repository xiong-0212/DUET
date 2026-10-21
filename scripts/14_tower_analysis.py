from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from scipy.stats import ttest_rel
from sklearn.metrics import matthews_corrcoef

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from rgcn_model import build_model

T = ['TE18', 'TL12', 'JL10', 'RB9']
SEEDS = [42, 0, 1, 2, 3]
IDX = {'graph': 640, 'sequence': 1280}
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
THR_GRID = np.linspace(0.05, 0.95, 91)
OUT = os.path.join(ROOT, 'results', 'tower_analysis.json')



CLEAN = {
    'mlp_rnafm': ('clean_mlp', 'graphs_seq_ss_clean.pt'),
    'graph': ('clean_seq_ss', 'graphs_seq_ss_clean.pt'),
    'sequence': ('clean_dual2_mlp', 'graphs_dual_clean.pt'),
    'fusion_gated': ('clean_dual2', 'graphs_dual_clean.pt'),
    'fusion_avgrepr': ('clean_dual2_avg', 'graphs_dual_clean.pt'),
    'fusion_logit': ('clean_dual2_logit', 'graphs_dual_clean.pt'),


    'cap_match': ('capm_m1275', 'graphs_dual_clean.pt'),
    'cap_large': ('capm_m2048', 'graphs_dual_clean.pt'),
    'cap_union': ('capm_m1920', 'graphs_dual_clean.pt'),
}
ORIG = {
    'mlp_rnafm': ('orig_mlp', 'graphs_seq_ss.pt'),
    'graph': ('orig_seq_ss', 'graphs_seq_ss.pt'),
}


def pack(tag, graphs_file, drop_val_leaked=True):
    
    gobj = torch.load(os.path.join(ROOT, 'data', 'processed', graphs_file),
                      map_location='cpu', weights_only=False)
    leak = json.load(open(os.path.join(ROOT, 'data', 'leaked_pairs.json'), encoding='utf-8'))
    drop = set(leak['remove_val_ids']) if drop_val_leaked else set()
    splits = {'val': [g for g in gobj['graphs']['val'] if g.rna_id not in drop]}
    splits.update({s: gobj['graphs'][s] for s in T})
    out = {}
    for seed in SEEDS:
        ck = torch.load(os.path.join(ROOT, 'checkpoints', 'controlled', f'{tag}_seed{seed}.pt'),
                        map_location='cpu', weights_only=False)
        cfg = dict(ck['config']['model'])
        cfg['num_relations'] = ck['num_relations']
        cfg['in_dim'] = gobj['in_dim']
        model = build_model(cfg, ck['num_relations']).to(DEV)
        model.load_state_dict(ck['state_dict'])
        model.eval()
        acc = {}
        with torch.no_grad():
            for split, graphs in splits.items():
                for g in graphs:
                    z = model(g.x.to(DEV), g.edge_index.to(DEV), g.edge_type.to(DEV),
                              torch.zeros(g.x.size(0), dtype=torch.long, device=DEV))
                    if isinstance(z, tuple):
                        z = z[0]
                    acc[f'{split}/{g.rna_id}'] = {'p': torch.sigmoid(z).float().cpu().numpy(),
                                                  'y': g.y.numpy().astype(int)}
        out[seed] = {'acc': acc, 'thr': float(ck['threshold'])}
    return out


def combine_decision_level(a, b):
    
    out = {}
    for s in SEEDS:
        acc = {k: {'p': 0.5 * (a[s]['acc'][k]['p'] + b[s]['acc'][k]['p']), 'y': a[s]['acc'][k]['y']}
               for k in a[s]['acc']}
        vy, vp = cat(acc, 'val')
        out[s] = {'acc': acc,
                  'thr': float(THR_GRID[int(np.argmax(
                      [matthews_corrcoef(vy, (vp >= t).astype(int)) for t in THR_GRID]))])}
    return out


def cat(acc, split):
    ks = sorted(k for k in acc if k.startswith(split + '/'))
    return (np.concatenate([acc[k]['y'] for k in ks]),
            np.concatenate([acc[k]['p'] for k in ks]))


def mcc(pack_, seed, split):
    y, p = cat(pack_[seed]['acc'], split)
    return float(matthews_corrcoef(y, (p >= pack_[seed]['thr']).astype(int)))


def avg4(pack_):
    return np.array([np.mean([mcc(pack_, s, sp) for sp in T]) for s in SEEDS])


def per_set(pack_, split):
    return np.array([mcc(pack_, s, split) for s in SEEDS])


def cmp(a, b):
    d = a - b
    return {'delta': float(d.mean()), 'p': float(ttest_rel(a, b).pvalue),
            'seeds_in_favour': int((d > 0).sum())}


def r3(x):
    return f'{x:.3f}'


def s3(x):
    return f'{x:+.3f}'


def fmt_p(p):
    return '{<}0.001' if p < 0.001 else r3(p)


def m3_pack(which):
    rows = json.load(open(os.path.join(ROOT, 'results', 'cobra_m3_equiv.json'),
                          encoding='utf-8'))[which]['bce']
    out = {}
    for r in rows:
        acc = {}
        for split, chains in r['per_chain'].items():
            for rid, v in chains.items():
                acc[f'{split}/{rid}'] = {'p': np.asarray(v['prob'], dtype=float),
                                         'y': np.asarray(v['label'], dtype=int)}
        out[r['seed']] = {'acc': acc, 'thr': float(r['threshold'])}
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    packs = {k: pack(*v) for k, v in CLEAN.items()}
    packs['ensemble'] = combine_decision_level(packs['graph'], packs['sequence'])

    rows = [('MLP on RNA-FM (no graph)', packs['mlp_rnafm']),
            ('DUET graph tower (R-GCN)', packs['graph']),
            ('DUET sequence tower (MLP on RiNALMo)', packs['sequence']),
            ('Fusion, gated representations', packs['fusion_gated']),
            ('Fusion, averaged representations', packs['fusion_avgrepr']),
            ('Fusion, averaged logits', packs['fusion_logit']),
            ('Single MLP, capacity-matched (2.04 M)', packs['cap_match']),
            ('Single MLP, 1.6x capacity (3.23 M)', packs['cap_large']),
            ('Single MLP, both backbones (2.31 M, 1920-d)', packs['cap_union']),
            ('Ensemble, averaged probabilities (DUET)', packs['ensemble'])]

    print(f"{'configuration':<44}{'MCC (four-benchmark average)':>32}")
    values = {}
    for name, p in rows:
        v = avg4(p)
        values[name] = v
        print(f'{name:<44}{v.mean():>16.3f} ± {v.std(ddof=1):.3f}')

    res = {'protocol': {'seeds': SEEDS, 'threshold': 'per seed, selected on the validation split',
                        'average': 'mean of the four benchmark MCCs'},
           'configurations': {n: {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)),
                                  'per_seed': [float(x) for x in v]} for n, v in values.items()},
           'vs_ensemble': {}, 'ensemble_vs_member_per_set': {}, 'leakage': {}}

    print('\npaired t-tests vs the decision-level ensemble (same seed, n=5):')
    for name, _ in rows[:-1]:
        c = cmp(values[name], values['Ensemble, averaged probabilities (DUET)'])
        res['vs_ensemble'][name] = c
        print(f'  {name:<44}Δ={c["delta"]:+.4f}  p={c["p"]:.4f}  {c["seeds_in_favour"]}/5')

    for sp in T:
        c = cmp(per_set(packs['ensemble'], sp), per_set(packs['sequence'], sp))
        res['ensemble_vs_member_per_set'][sp] = c
        print(f'  ensemble − sequence tower, {sp:<5} Δ={c["delta"]:+.4f}  p={c["p"]:.4f}')

    g_vs_m = cmp(values['DUET graph tower (R-GCN)'], values['MLP on RNA-FM (no graph)'])
    res['graph_vs_mlp'] = g_vs_m
    print(f'\ngraph tower − MLP on the same RNA-FM features: Δ={g_vs_m["delta"]:+.4f} '
          f'p={g_vs_m["p"]:.4f} {g_vs_m["seeds_in_favour"]}/5')

    print('\nleakage: same architecture, original (leaked) vs de-leaked training split')
    for key, label in [('mlp_rnafm', 'MLP on RNA-FM'), ('graph', 'graph tower')]:
        leaked = avg4(pack(*ORIG[key], drop_val_leaked=False))
        clean = values[{'mlp_rnafm': 'MLP on RNA-FM (no graph)',
                        'graph': 'DUET graph tower (R-GCN)'}[key]]
        c = cmp(leaked, clean)
        res['leakage'][label] = {'leaked': float(leaked.mean()), 'deleaked': float(clean.mean()),
                                 'delta': c['delta'], 'p': c['p']}
        print(f'  {label:<16} leaked={leaked.mean():.3f}  de-leaked={clean.mean():.3f}  '
              f'Δ={c["delta"]:+.3f}  p={c["p"]:.4f}')
    base_leaked, base_clean = avg4(m3_pack('orig')), avg4(m3_pack('clean'))
    c = cmp(base_leaked, base_clean)
    res['leakage']['CoBRA-M3 equivalent'] = {'leaked': float(base_leaked.mean()),
                                             'deleaked': float(base_clean.mean()),
                                             'delta': c['delta'], 'p': c['p']}
    print(f'  {"CoBRA-M3 equiv":<16} leaked={base_leaked.mean():.3f}  '
          f'de-leaked={base_clean.mean():.3f}  Δ={c["delta"]:+.3f}  p={c["p"]:.4f}')



    NAME2TAG = {'MLP on RNA-FM (no graph)': 'MlpFm',
                'DUET graph tower (R-GCN)': 'Graph',
                'DUET sequence tower (MLP on RiNALMo)': 'Seq',
                'Fusion, gated representations': 'FusGated',
                'Fusion, averaged representations': 'FusReprAvg',
                'Fusion, averaged logits': 'FusLogit',
                'Single MLP, capacity-matched (2.04 M)': 'CapMatch',
                'Single MLP, 1.6x capacity (3.23 M)': 'CapLarge',
                'Single MLP, both backbones (2.31 M, 1920-d)': 'CapUnion'}
    res['table3_tags'] = NAME2TAG
    m = {
        'MlpFmAvg': r3(values['MLP on RNA-FM (no graph)'].mean()),
        'MlpFmSD': r3(values['MLP on RNA-FM (no graph)'].std(ddof=1)),
        'FusionGatedMCC': r3(values['Fusion, gated representations'].mean()),
        'FusionGatedSD': r3(values['Fusion, gated representations'].std(ddof=1)),
        'FusionReprAvgMCC': r3(values['Fusion, averaged representations'].mean()),
        'FusionReprAvgSD': r3(values['Fusion, averaged representations'].std(ddof=1)),
        'FusionLogitMCC': r3(values['Fusion, averaged logits'].mean()),
        'FusionLogitSD': r3(values['Fusion, averaged logits'].std(ddof=1)),
        'GraphVsMlpDelta': s3(res['graph_vs_mlp']['delta']),
        'GraphVsMlpP': fmt_p(res['graph_vs_mlp']['p']),

        'EnsVsSeqDelta': s3(-res['vs_ensemble']['DUET sequence tower (MLP on RiNALMo)']['delta']),
        'EnsVsSeqP': fmt_p(res['vs_ensemble']['DUET sequence tower (MLP on RiNALMo)']['p']),
        'EnsVsGraphDelta': s3(-res['vs_ensemble']['DUET graph tower (R-GCN)']['delta']),
        'EnsVsGraphP': fmt_p(res['vs_ensemble']['DUET graph tower (R-GCN)']['p']),


        **{f'{tag}VsEnsDelta': s3(c['delta'])
           for name, tag in NAME2TAG.items()
           for c in [res['vs_ensemble'][name]]},
        **{f'{tag}VsEnsP': fmt_p(c['p'])
           for name, tag in NAME2TAG.items()
           for c in [res['vs_ensemble'][name]]},
        'EnsVsSeqTEDelta': s3(res['ensemble_vs_member_per_set']['TE18']['delta']),
        'EnsVsSeqTEP': fmt_p(res['ensemble_vs_member_per_set']['TE18']['p']),
        'LeakMLPLeaked': r3(res['leakage']['MLP on RNA-FM']['leaked']),
        'LeakMLPClean': r3(res['leakage']['MLP on RNA-FM']['deleaked']),
        'LeakMLPDelta': r3(res['leakage']['MLP on RNA-FM']['delta']),
        'LeakMLPP': fmt_p(res['leakage']['MLP on RNA-FM']['p']),
        'LeakGraphLeaked': r3(res['leakage']['graph tower']['leaked']),
        'LeakGraphClean': r3(res['leakage']['graph tower']['deleaked']),
        'LeakGraphDelta': r3(res['leakage']['graph tower']['delta']),
        'LeakGraphP': fmt_p(res['leakage']['graph tower']['p']),
        'LeakBaseLeaked': r3(res['leakage']['CoBRA-M3 equivalent']['leaked']),
        'LeakBaseClean': r3(res['leakage']['CoBRA-M3 equivalent']['deleaked']),
        'LeakBaseDelta': r3(res['leakage']['CoBRA-M3 equivalent']['delta']),
        'LeakBaseP': fmt_p(res['leakage']['CoBRA-M3 equivalent']['p']),
    }
    res['macros'] = m
    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\n[written] {OUT}  ({len(m)} macros)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
