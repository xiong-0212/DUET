from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from scipy.stats import ttest_rel
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE

while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from rgcn_model import build_model

T = ['TE18', 'TL12', 'JL10', 'RB9']
DEV_SEEDS = [42, 0, 1, 2, 3]
CONF_SEEDS = [7, 11, 13, 17, 23]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
THR_GRID = np.linspace(0.05, 0.95, 91)
METRICS = ['MCC', 'AUC', 'AUPRC', 'F1', 'Precision', 'Sensitivity', 'Specificity', 'GMean']
OUT = os.path.join(ROOT, 'results', 'three_tower_tables.json')




SINGLE = {
    'MLP on RNA-FM (no graph)':        ('clean_mlp',      'ns_mlp_rnafm',   'graphs_seq_ss_clean.pt'),
    'DUET graph tower (R-GCN)':        ('clean_seq_ss',   'ns_graph',       'graphs_seq_ss_clean.pt'),
    'DUET sequence tower A (TCN)':     ('slim_tcn',       'ns_tcn',         'graphs_dual_clean.pt'),
    'DUET sequence tower B (DenseNet)': ('slim_dense',    'ns_dense',       'graphs_dual_clean.pt'),
    'Single MLP on RiNALMo (1.65 M)':  ('clean_dual2_mlp', 'ns_mlp',        'graphs_dual_clean.pt'),
    'Single MLP, capacity-matched (2.04 M)': ('capm_m1275', 'ns_cap_m1275', 'graphs_dual_clean.pt'),
    'Single MLP, 1.6x capacity (3.23 M)':    ('capm_m2048', 'ns_cap_m2048', 'graphs_dual_clean.pt'),
    'Single MLP, both backbones (2.31 M, 1920-d)': ('capm_m1920', 'ns_cap_m1920',
                                                    'graphs_dual_clean.pt'),
}
PAIRS = {'graph + TCN': ('DUET graph tower (R-GCN)', 'DUET sequence tower A (TCN)'),
         'graph + DenseNet': ('DUET graph tower (R-GCN)', 'DUET sequence tower B (DenseNet)'),
         'TCN + DenseNet': ('DUET sequence tower A (TCN)', 'DUET sequence tower B (DenseNet)')}
TRIPLE_NAME = 'DUET (three towers)'
TRIPLE_MEMBERS = ('DUET graph tower (R-GCN)', 'DUET sequence tower A (TCN)',
                  'DUET sequence tower B (DenseNet)')
QUAD_MEMBERS = TRIPLE_MEMBERS + ('Single MLP on RiNALMo (1.65 M)',)


def gmean(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    se = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    return float(np.sqrt(se * sp))


def specificity(y, pred):
    tn, fp, _, _ = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def met(y, p, thr):
    pred = (p >= thr).astype(int)
    return {'MCC': float(matthews_corrcoef(y, pred)), 'AUC': float(roc_auc_score(y, p)),
            'AUPRC': float(average_precision_score(y, p)),
            'F1': float(f1_score(y, pred, zero_division=0)),
            'Precision': float(precision_score(y, pred, zero_division=0)),
            'Sensitivity': float(recall_score(y, pred, zero_division=0)),
            'Specificity': specificity(y, pred), 'GMean': gmean(y, pred)}



_CACHE = {}


def model_probs(tag, seed, graphs_file):
    
    key = (tag, seed, graphs_file)
    if key in _CACHE:
        return _CACHE[key]
    gobj = torch.load(os.path.join(ROOT, 'data', 'processed', graphs_file),
                      map_location='cpu', weights_only=False)
    leak = json.load(open(os.path.join(ROOT, 'data', 'leaked_pairs.json'), encoding='utf-8'))
    drop = set(leak['remove_val_ids'])
    splits = {'val': [g for g in gobj['graphs']['val'] if g.rna_id not in drop]}
    splits.update({s: gobj['graphs'][s] for s in T})

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
    out = {'acc': acc, 'tag': tag, 'threshold': float(ck['threshold']),
           'params': int(sum(p.numel() for p in model.parameters() if p.requires_grad))}
    _CACHE[key] = out
    return out


def available_seeds(dev_tag, conf_tag):
    seeds = []
    for batch, tag in ((DEV_SEEDS, dev_tag), (CONF_SEEDS, conf_tag)):
        if tag is None:
            continue
        for s in batch:
            if os.path.exists(os.path.join(ROOT, 'checkpoints', 'controlled',
                                           f'{tag}_seed{s}.pt')):
                seeds.append(s)
    return seeds


def baseline_packs(seeds):
    
    out, cache = {}, {}
    for seed in seeds:
        fname = 'cobra_m3_equiv.json' if seed in DEV_SEEDS else 'cobra_m3_equiv_newseeds.json'
        if fname not in cache:
            cache[fname] = json.load(open(os.path.join(ROOT, 'results', fname), encoding='utf-8'))
        row = next(r for r in cache[fname]['clean']['bce'] if r['seed'] == seed)
        acc = {}
        for split, chains in row['per_chain'].items():
            for rid, v in chains.items():
                acc[f'{split}/{rid}'] = {'p': np.asarray(v['prob'], dtype=float),
                                         'y': np.asarray(v['label'], dtype=int)}
        out[seed] = {'acc': acc, 'thr': float(row['threshold'])}
    return out


def combine(member_packs, seed):
    
    keys = set(member_packs[0]['acc'])
    for m in member_packs[1:]:
        keys &= set(m['acc'])
    acc = {k: {'p': np.mean([m['acc'][k]['p'] for m in member_packs], axis=0),
               'y': member_packs[0]['acc'][k]['y']} for k in keys}
    vy, vp = cat(acc, 'val')
    return {'acc': acc,
            'thr': float(THR_GRID[int(np.argmax(
                [matthews_corrcoef(vy, (vp >= t).astype(int)) for t in THR_GRID]))])}


def cat(pack_, split):
    ks = sorted(k for k in pack_ if k.startswith(split + '/'))
    return (np.concatenate([pack_[k]['y'] for k in ks]),
            np.concatenate([pack_[k]['p'] for k in ks]))


def series(pk, seeds, split, metric):
    return np.array([met(*cat(pk[s]['acc'], split), pk[s]['thr'])[metric] for s in seeds])


def avg_series(pk, seeds, metric):
    return np.array([np.mean([met(*cat(pk[s]['acc'], sp), pk[s]['thr'])[metric] for sp in T])
                     for s in seeds])


def cmp(a, b):
    d = a - b
    return {'delta': float(d.mean()), 'p': float(ttest_rel(a, b).pvalue),
            'seeds_in_favour': int((d > 0).sum()), 'n': int(len(a))}


def r3(x):
    return f'{x:.3f}'


def s3(x):
    return f'{x:+.3f}'


def fmt_p(p):
    return '{<}0.001' if p < 0.001 else r3(p)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


    packs, params, seeds_of = {}, {}, {}
    for name, (dev_tag, conf_tag, graphs_file) in SINGLE.items():
        seeds = available_seeds(dev_tag, conf_tag)
        if not seeds:
            print(f'[skip] {name}: no checkpoints yet')
            continue
        packs[name] = {}
        for s in seeds:
            tag = dev_tag if s in DEV_SEEDS else conf_tag
            p = model_probs(tag, s, graphs_file)
            packs[name][s] = {'acc': p['acc'], 'thr': p['threshold'], 'tag': tag}
            params[name] = p['params']
        seeds_of[name] = seeds
        print(f'[eval] {name:<45} seeds={seeds}  params={params[name]:,}')

    common = [s for s in DEV_SEEDS + CONF_SEEDS
              if all(s in seeds_of.get(n, []) for n in TRIPLE_MEMBERS)]


    base = baseline_packs(common)


    def where(*names):
        return [s for s in DEV_SEEDS + CONF_SEEDS
                if all(s in seeds_of.get(n, []) for n in names)]

    def build_combination(members, seeds):
        return {s: combine([packs[n][s] for n in members], s) for s in seeds}

    combos = {}
    for label, (a, b) in PAIRS.items():
        if a in packs and b in packs:
            seeds = where(a, b)
            combos[label] = (build_combination((a, b), seeds), seeds)
    tri_seeds = where(*TRIPLE_MEMBERS)
    combos[TRIPLE_NAME] = (build_combination(TRIPLE_MEMBERS, tri_seeds), tri_seeds)
    if all(n in packs for n in QUAD_MEMBERS):
        quad_seeds = where(*QUAD_MEMBERS)
        combos['DUET + MLP tower (four towers)'] = (build_combination(QUAD_MEMBERS, quad_seeds),
                                                    quad_seeds)


    def summarise(pk, seeds):
        return {'per_set_mean': {sp: float(series(pk, seeds, sp, 'MCC').mean()) for sp in T},
                'per_set_sd': {sp: float(series(pk, seeds, sp, 'MCC').std(ddof=1)) for sp in T},
                'average_over_sets_mean': float(avg_series(pk, seeds, 'MCC').mean()),
                'average_over_sets_sd': float(avg_series(pk, seeds, 'MCC').std(ddof=1)),
                'per_seed': [float(x) for x in avg_series(pk, seeds, 'MCC')],
                'seeds': list(seeds),
                'metrics_average': {m: {'mean': float(avg_series(pk, seeds, m).mean()),
                                        'sd': float(avg_series(pk, seeds, m).std(ddof=1))}
                                    for m in METRICS},
                'metrics_per_set': {
                    m: {sp: {'mean': float(series(pk, seeds, sp, m).mean()),
                             'sd': float(series(pk, seeds, sp, m).std(ddof=1))} for sp in T}
                    for m in METRICS}}

    res = {'protocol': {'threshold': 'per seed, selected on the validation split',
                        'development_seeds': DEV_SEEDS, 'confirmation_seeds': CONF_SEEDS,
                        'test': 'same-seed paired t-test',
                        'average': 'mean of the four benchmark MCCs'},
           'params': params, 'seeds_of': seeds_of, 'table2': {}, 'table3': {},
           'vs_baseline': {}, 'vs_ensemble': {}, 'four_tower': {}}

    duet = combos[TRIPLE_NAME][0]
    for name in ['DUET graph tower (R-GCN)', 'DUET sequence tower A (TCN)',
                 'DUET sequence tower B (DenseNet)']:
        if name in packs:
            res['table2'][name] = summarise(packs[name], seeds_of[name])
    res['table2'][TRIPLE_NAME] = summarise(duet, tri_seeds)
    bsum = summarise(base, common)
    res['table2']['CoBRA-M3 equiv. (retrained baseline)'] = bsum

    print('\n--- Table 2 (MCC, four-benchmark average) ---')
    for name, d in res['table2'].items():
        print(f'{name:<45} {d["average_over_sets_mean"]:.4f} ± {d["average_over_sets_sd"]:.4f}'
              f'   n={len(d["seeds"])}')

    print('\n--- per benchmark ---')
    for name, d in res['table2'].items():
        row = '  '.join(f'{sp} {d["per_set_mean"][sp]:.3f}±{d["per_set_sd"][sp]:.3f}' for sp in T)
        print(f'{name:<45} {row}')


    print('\n--- DUET vs baseline (same-seed paired t-test) ---')
    for sp in T + ['average']:
        a = series(duet, tri_seeds, sp, 'MCC') if sp != 'average' else avg_series(duet, tri_seeds,
                                                                                  'MCC')
        b = series(base, common, sp, 'MCC') if sp != 'average' else avg_series(base, common, 'MCC')
        c = cmp(a, b)
        res['vs_baseline'][sp] = c
        print(f'  {sp:<8} Δ={c["delta"]:+.4f}  p={c["p"]:.4f}  {c["seeds_in_favour"]}/'
              f'{len(a)}')
    for m in METRICS:
        a, b = avg_series(duet, tri_seeds, m), avg_series(base, common, m)
        res['vs_baseline'][m] = cmp(a, b)
        print(f'  {m:<10} Δ={a.mean()-b.mean():+.4f}  p={res["vs_baseline"][m]["p"]:.4f}')


    print('\n--- Table 3 (single models, pairs, triple) ---')
    for name, (pk, seeds) in combos.items():
        res['table3'][name] = summarise(pk, seeds)
    for name in SINGLE:
        if name in packs:
            res['table3'][name] = res['table2'].get(name) or summarise(packs[name],
                                                                       seeds_of[name])
    for name, d in res['table3'].items():
        if name in (TRIPLE_NAME,):
            continue
        pk = combos[name][0] if name in combos else packs[name]
        seeds = combos[name][1] if name in combos else seeds_of[name]
        use = [s for s in seeds if s in tri_seeds]
        c = cmp(avg_series(pk, use, 'MCC'), avg_series(duet, use, 'MCC'))
        res['vs_ensemble'][name] = c
        print(f'{name:<45} {d["average_over_sets_mean"]:.4f}'
              f'   Δ={c["delta"]:+.4f}  p={c["p"]:.4f}  n={len(use)}')

    if 'DUET + MLP tower (four towers)' in combos:
        pk, seeds = combos['DUET + MLP tower (four towers)']
        c = cmp(avg_series(pk, seeds, 'MCC'), avg_series(duet, seeds, 'MCC'))
        res['four_tower'] = {'vs_three': c,
                             'mcc': float(avg_series(pk, seeds, 'MCC').mean()),
                             'sd': float(avg_series(pk, seeds, 'MCC').std(ddof=1)),
                             'params': params.get('Single MLP on RiNALMo (1.65 M)', 0)
                             + sum(params[n] for n in TRIPLE_MEMBERS if n in params)}
        print(f'\nfour towers (DUET + MLP)  {res["four_tower"]["mcc"]:.4f} ± '
              f'{res["four_tower"]["sd"]:.4f}  vs three towers Δ={c["delta"]:+.4f} '
              f'p={c["p"]:.4f}  {c["seeds_in_favour"]}/{c["n"]}')



    res['fixed_threshold'] = {}
    for name, pk, seeds in ([('CoBRA-M3 equiv. (retrained baseline)', base, common)]
                            + [(n, packs[n], seeds_of[n]) for n in TRIPLE_MEMBERS]
                            + [(TRIPLE_NAME, duet, tri_seeds)]):
        block = {}
        for sp in T + ['average']:
            vals = {m: [] for m in METRICS}
            for s in seeds:
                if sp == 'average':
                    mm = {k: float(np.mean([met(*cat(pk[s]['acc'], x), 0.5)[k] for x in T]))
                          for k in METRICS}
                else:
                    mm = met(*cat(pk[s]['acc'], sp), 0.5)
                for k in METRICS:
                    vals[k].append(mm[k])
            block[sp] = {k: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1))}
                         for k, v in vals.items()}
        res['fixed_threshold'][name] = block








    SG = {'TE18': 'TE', 'TL12': 'TL', 'JL10': 'JL', 'RB9': 'RB', 'average': 'Avg'}
    ABBR = [('MCC', 'MCC'), ('AUC', 'AUC'), ('AUPRC', 'AUPRC'), ('F1', 'FOne'),
            ('Precision', 'Precision'), ('Sensitivity', 'Sensitivity'),
            ('Specificity', 'Specificity'), ('GMean', 'GMean')]
    tag = {'CoBRA-M3 equiv. (retrained baseline)': 'Base', 'DUET graph tower (R-GCN)': 'Graph',
           'DUET sequence tower A (TCN)': 'Tcn', 'DUET sequence tower B (DenseNet)': 'Dense',
           TRIPLE_NAME: 'Duet'}
    m = {}

    def cell(d, full, sp):
        return d['metrics_average'][full] if sp == 'average' else d['metrics_per_set'][full][sp]

    for name, t in tag.items():
        d = res['table2'][name]
        for sp, sg in SG.items():
            m[f'Tb{t}{sg}'] = (f'{cell(d, "MCC", sp)["mean"]:.3f} \\pm '
                               f'{cell(d, "MCC", sp)["sd"]:.3f}')
            for full, ab in ABBR:
                m[f'{t}{ab}{sg}'] = r3(cell(d, full, sp)['mean'])
                m[f'{t}{ab}{sg}SD'] = r3(cell(d, full, sp)['sd'])
            if sg == 'Avg':
                m[f'Tb{t}AvgPoint'] = f'{cell(d, "MCC", sp)["mean"]:.3f}'
        m[f'{t}AvgMCC'] = r3(d['average_over_sets_mean'])
        m[f'{t}AvgSD'] = r3(d['average_over_sets_sd'])

    for full, ab in ABBR:
        for sp, sg in SG.items():
            a = (series(duet, tri_seeds, sp, full) if sp != 'average'
                 else avg_series(duet, tri_seeds, full))
            b = (series(base, common, sp, full) if sp != 'average'
                 else avg_series(base, common, full))
            c = cmp(a, b)
            m[f'D{ab}{sg}'] = s3(c['delta'])
            m[f'P{ab}{sg}'] = fmt_p(c['p'])

        m[f'Delta{ab}'] = m[f'D{ab}Avg']
        m[f'P{ab}'] = m[f'P{ab}Avg']
    m['DeltaAvgMCC'] = m['DMCCAvg']
    m['PAvgMCC'] = m['PMCCAvg']
    for sp, sg in zip(T, ['TE', 'TL', 'JL', 'RB']):
        m[f'Delta{sg}'] = m[f'DMCC{sg}']
        m[f'P{sg}'] = m[f'PMCC{sg}']
    c = res['vs_baseline']['average']
    m['SeedsInFavour'] = str(c['seeds_in_favour'])
    m['NSeedPairs'] = str(c['n'])


    tag3 = {'MLP on RNA-FM (no graph)': 'MlpFm',
            'DUET graph tower (R-GCN)': 'Graph', 'DUET sequence tower A (TCN)': 'Tcn',
            'DUET sequence tower B (DenseNet)': 'Dense',
            'Single MLP on RiNALMo (1.65 M)': 'Mlp',
            'Single MLP, capacity-matched (2.04 M)': 'CapMatch',
            'Single MLP, 1.6x capacity (3.23 M)': 'CapLarge',
            'Single MLP, both backbones (2.31 M, 1920-d)': 'CapUnion',
            'graph + TCN': 'PairGraphTcn', 'graph + DenseNet': 'PairGraphDense',
            'TCN + DenseNet': 'PairTcnDense', TRIPLE_NAME: 'Ens'}
    for name, t in tag3.items():
        if name not in res['table3']:
            continue
        d = res['table3'][name]
        m[f'Tb{t}'] = f'{d["average_over_sets_mean"]:.3f} \\pm {d["average_over_sets_sd"]:.3f}'
        m[f'Tb{t}Point'] = f'{d["average_over_sets_mean"]:.3f}'
        if name in res['vs_ensemble'] and name != TRIPLE_NAME:
            c = res['vs_ensemble'][name]
            m[f'{t}VsEnsDelta'] = s3(c['delta'])
            m[f'{t}VsEnsP'] = fmt_p(c['p'])

    for name, t in [('DUET graph tower (R-GCN)', 'Graph'),
                    ('DUET sequence tower A (TCN)', 'Tcn'),
                    ('DUET sequence tower B (DenseNet)', 'Dense')]:
        if name in res['vs_ensemble']:
            c = res['vs_ensemble'][name]
            m[f'EnsVs{t}Delta'] = s3(-c['delta'])
            m[f'EnsVs{t}P'] = fmt_p(c['p'])

    for name, t in [('graph + TCN', 'GraphTcn'), ('graph + DenseNet', 'GraphDense'),
                    ('TCN + DenseNet', 'TcnDense')]:
        if name in res['vs_ensemble']:
            c = res['vs_ensemble'][name]
            m[f'EnsVs{t}Delta'] = s3(-c['delta'])
            m[f'EnsVs{t}Gap'] = f'{abs(c["delta"]):.3f}'
            m[f'EnsVs{t}P'] = fmt_p(c['p'])
            m[f'TbPair{t}'] = (f'{res["table3"][name]["average_over_sets_mean"]:.3f} \\pm '
                               f'{res["table3"][name]["average_over_sets_sd"]:.3f}')
    if res['four_tower']:
        m['TbFourTower'] = (f'{res["four_tower"]["mcc"]:.3f} '
                            f'\\pm {res["four_tower"]["sd"]:.3f}')
        m['FourVsThreeDelta'] = s3(res['four_tower']['vs_three']['delta'])
        m['FourVsThreeP'] = fmt_p(res['four_tower']['vs_three']['p'])
        m['FourTowerParams'] = f'{res["four_tower"]["params"]:,}'.replace(',', '{,}')


    res['members_vs_baseline'] = {}
    for name, t in [('DUET graph tower (R-GCN)', 'Graph'),
                    ('DUET sequence tower A (TCN)', 'Tcn'),
                    ('DUET sequence tower B (DenseNet)', 'Dense')]:
        if name in packs:
            s = [x for x in seeds_of[name] if x in common]
            c = cmp(avg_series(packs[name], s, 'MCC'), avg_series(base, s, 'MCC'))
            res['members_vs_baseline'][name] = c
            m[f'{t}VsBaseDelta'] = s3(c['delta'])
            m[f'{t}VsBaseP'] = fmt_p(c['p'])


    res['members_pairwise'] = {}
    for a, b in [('DUET graph tower (R-GCN)', 'DUET sequence tower A (TCN)'),
                 ('DUET graph tower (R-GCN)', 'DUET sequence tower B (DenseNet)'),
                 ('DUET sequence tower A (TCN)', 'DUET sequence tower B (DenseNet)')]:
        if a in packs and b in packs:
            s = where(a, b)
            c = cmp(avg_series(packs[a], s, 'MCC'), avg_series(packs[b], s, 'MCC'))
            res['members_pairwise'][f'{a} - {b}'] = c
            m[f'{tag[a]}Vs{tag[b]}Delta'] = s3(c['delta'])
            m[f'{tag[a]}Vs{tag[b]}P'] = fmt_p(c['p'])
            print(f'  {a} - {b}: Δ={c["delta"]:+.4f} p={c["p"]:.4f}')


    if 'MLP on RNA-FM (no graph)' in packs:
        gseeds = where('MLP on RNA-FM (no graph)', 'DUET graph tower (R-GCN)')
        c = cmp(avg_series(packs['DUET graph tower (R-GCN)'], gseeds, 'MCC'),
                avg_series(packs['MLP on RNA-FM (no graph)'], gseeds, 'MCC'))
        res['graph_vs_mlp'] = c
        m['GraphVsMlpDelta'] = s3(c['delta'])
        m['GraphVsMlpP'] = fmt_p(c['p'])
        m['GraphVsMlpN'] = str(c['n'])
        m['GraphVsMlpSeeds'] = str(c['seeds_in_favour'])

    m['GraphParams'] = f'{params.get("DUET graph tower (R-GCN)", 0):,}'.replace(',', '{,}')
    m['TcnParams'] = f'{params.get("DUET sequence tower A (TCN)", 0):,}'.replace(',', '{,}')
    m['DenseParams'] = f'{params.get("DUET sequence tower B (DenseNet)", 0):,}'.replace(',', '{,}')
    m['MlpParams'] = f'{params.get("Single MLP on RiNALMo (1.65 M)", 0):,}'.replace(',', '{,}')
    m['MlpFmParams'] = f'{params.get("MLP on RNA-FM (no graph)", 0):,}'.replace(',', '{,}')
    m['CapMatchParams'] = f'{params.get("Single MLP, capacity-matched (2.04 M)", 0):,}'.replace(
        ',', '{,}')
    m['CapLargeParams'] = f'{params.get("Single MLP, 1.6x capacity (3.23 M)", 0):,}'.replace(
        ',', '{,}')
    m['CapUnionParams'] = f'{params.get("Single MLP, both backbones (2.31 M, 1920-d)", 0):,}'.replace(
        ',', '{,}')

    for label, members in [('PairGraphTcn', ('DUET graph tower (R-GCN)',
                                             'DUET sequence tower A (TCN)')),
                           ('PairGraphDense', ('DUET graph tower (R-GCN)',
                                               'DUET sequence tower B (DenseNet)')),
                           ('PairTcnDense', ('DUET sequence tower A (TCN)',
                                             'DUET sequence tower B (DenseNet)')),
                           ('Ens', TRIPLE_MEMBERS)]:
        tot = sum(params.get(x, 0) for x in members)
        m[f'{label}Params'] = f'{tot:,}'.replace(',', '{,}')
    m['DuetParams'] = m['EnsParams']
    m['NDevSeeds'] = str(len(DEV_SEEDS))
    m['NConfSeeds'] = str(len(CONF_SEEDS))
    m['NSeeds'] = str(len(tri_seeds))
    res['macros'] = m

    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\n[written] {OUT}  ({len(m)} macros)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
