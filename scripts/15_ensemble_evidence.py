from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from rgcn_model import build_model

T = ['TE18', 'TL12', 'JL10', 'RB9']
SEEDS = [42, 0, 1, 2, 3]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
THR_GRID = np.linspace(0.05, 0.95, 91)
MEMBERS = {'graph': ('clean_seq_ss', 'graphs_seq_ss_clean.pt'),
           'sequence': ('clean_dual2_mlp', 'graphs_dual_clean.pt')}
OUT = os.path.join(ROOT, 'results', 'ensemble_evidence.json')


def pack(tag, graphs_file):
    gobj = torch.load(os.path.join(ROOT, 'data', 'processed', graphs_file),
                      map_location='cpu', weights_only=False)
    leak = json.load(open(os.path.join(ROOT, 'data', 'leaked_pairs.json'), encoding='utf-8'))
    drop = set(leak['remove_val_ids'])
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


def ensemble_pack(a, b):
    out = {}
    for s in SEEDS:
        acc = {k: {'p': 0.5 * (a[s]['acc'][k]['p'] + b[s]['acc'][k]['p']), 'y': a[s]['acc'][k]['y']}
               for k in a[s]['acc']}
        ks = sorted(k for k in acc if k.startswith('val/'))
        vy = np.concatenate([acc[k]['y'] for k in ks])
        vp = np.concatenate([acc[k]['p'] for k in ks])
        out[s] = {'acc': acc,
                  'thr': float(THR_GRID[int(np.argmax(
                      [matthews_corrcoef(vy, (vp >= t).astype(int)) for t in THR_GRID]))])}
    return out


def cat(pack_, seed, split):
    ks = sorted(k for k in pack_[seed]['acc'] if k.startswith(split + '/'))
    y = np.concatenate([pack_[seed]['acc'][k]['y'] for k in ks])
    p = np.concatenate([pack_[seed]['acc'][k]['p'] for k in ks])
    return y, p


def mcc(pack_, seed, split):
    y, p = cat(pack_, seed, split)
    return float(matthews_corrcoef(y, (p >= pack_[seed]['thr']).astype(int)))


def brier(pack_, seed, split):
    y, p = cat(pack_, seed, split)
    return float(np.mean((p - y.astype(float)) ** 2))


def per_chain_brier(pack_, seed, split):
    keys = sorted(k for k in pack_[seed]['acc'] if k.startswith(split + '/'))
    return np.array([np.mean((pack_[seed]['acc'][k]['p']
                              - pack_[seed]['acc'][k]['y'].astype(float)) ** 2) for k in keys])


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    graph = pack(*MEMBERS['graph'])
    seq = pack(*MEMBERS['sequence'])
    ens = ensemble_pack(graph, seq)

    res = {'protocol': 'five seeds, each at its own validation-selected threshold', 'seeds': SEEDS}


    rows = {}
    print('Ambiguity decomposition (Brier score, averaged over the four benchmarks):')
    for name, a, b, e in [('graph/sequence', graph, seq, ens)]:
        e_bar, e_ens, amb = [], [], []
        for s in SEEDS:
            mb = np.mean([(brier(a, s, sp) + brier(b, s, sp)) / 2 for sp in T])
            eb = np.mean([brier(e, s, sp) for sp in T])
            e_bar.append(mb)
            e_ens.append(eb)
            amb.append(mb - eb)
        rows[name] = {'mean_member_brier': float(np.mean(e_bar)),
                      'ensemble_brier': float(np.mean(e_ens)),
                      'ambiguity': float(np.mean(amb)),
                      'ambiguity_per_seed': [float(x) for x in amb]}
        print(f'  mean member Brier = {np.mean(e_bar):.4f}')
        print(f'  ensemble   Brier = {np.mean(e_ens):.4f}')
        print(f'  ambiguity  A     = {np.mean(amb):+.4f}  (per seed '
              f'{[round(x, 4) for x in amb]})')
        print(f'  -> the ensemble\'s Brier is lower than the members\' average in '
              f'{sum(1 for x in amb if x > 0)}/5 seeds (A > 0 is the theoretical guarantee)')
    res['ambiguity'] = rows


    print('\nDiversity of the two towers (residue level, each seed at its own threshold):')
    per_seed = {'pearson_brier': [], 'disagreement': [], 'q_statistic': []}
    for s in SEEDS:
        pb = []
        for sp in T:
            x, y = per_chain_brier(graph, s, sp), per_chain_brier(seq, s, sp)
            if len(x) > 2 and x.std() > 0 and y.std() > 0:
                pb.append(float(np.corrcoef(x, y)[0, 1]))
        per_seed['pearson_brier'].append(float(np.mean(pb)))
        dis, q = [], []
        for sp in T:
            y_, pg = cat(graph, s, sp)
            _, ps = cat(seq, s, sp)
            cg = (pg >= graph[s]['thr']).astype(int) == y_
            cs = (ps >= seq[s]['thr']).astype(int) == y_
            both_wrong = int(np.sum(~cg & ~cs))
            one_wrong = int(np.sum(cg ^ cs))
            dis.append(one_wrong / (one_wrong + both_wrong + 1e-9))
            a11 = int(np.sum(cg & cs))
            a00 = int(np.sum(~cg & ~cs))
            a10 = int(np.sum(cg & ~cs))
            a01 = int(np.sum(~cg & cs))
            if a11 * a00 + a10 * a01 > 0:
                q.append((a11 * a00 - a10 * a01) / (a11 * a00 + a10 * a01))
        per_seed['disagreement'].append(float(np.mean(dis)))
        per_seed['q_statistic'].append(float(np.mean(q)))
    for k, v in per_seed.items():
        print(f'  {k:<16} {np.mean(v):+.3f} ± {np.std(v, ddof=1):.3f}')
    res['diversity'] = {k: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)),
                            'per_seed': v} for k, v in per_seed.items()}


    print('\nError overlap (residues pooled over the four benchmarks, per seed):')
    tot = {'both_right': 0, 'graph_only_wrong': 0, 'sequence_only_wrong': 0, 'both_wrong': 0}
    rec = {'graph_only_wrong': [], 'sequence_only_wrong': []}
    for s in SEEDS:
        cnt = {k: 0 for k in tot}
        good = {k: 0 for k in rec}
        for sp in T:
            y_, pg = cat(graph, s, sp)
            _, ps = cat(seq, s, sp)
            _, pe = cat(ens, s, sp)
            cg = (pg >= graph[s]['thr']).astype(int) == y_
            cs = (ps >= seq[s]['thr']).astype(int) == y_
            ce = (pe >= ens[s]['thr']).astype(int) == y_
            cnt['both_right'] += int(np.sum(cg & cs))
            cnt['both_wrong'] += int(np.sum(~cg & ~cs))
            cnt['graph_only_wrong'] += int(np.sum(~cg & cs))
            cnt['sequence_only_wrong'] += int(np.sum(cg & ~cs))
            good['graph_only_wrong'] += int(np.sum((~cg & cs) & ce))
            good['sequence_only_wrong'] += int(np.sum((cg & ~cs) & ce))
        for k in tot:
            tot[k] += cnt[k]
        for k in rec:
            rec[k].append(good[k] / max(cnt[k], 1))
    n_tot = sum(tot.values())
    for k in tot:
        print(f'  {k:<20} {tot[k]:>7d}  ({100 * tot[k] / n_tot:4.1f} %)')
    for k, v in rec.items():
        print(f'  ensemble correct on "{k}": {np.mean(v) * 100:.1f} %')
    res['double_fault'] = {'counts': tot, 'residues': n_tot,
                           'recovery': {k: float(np.mean(v)) for k, v in rec.items()},
                           'recovery_per_seed': rec}


    print('\nEnsemble vs hindsight selection of the better member per benchmark:')
    ens_avg, orc_avg = [], []
    per_set = {}
    for s in SEEDS:
        e = np.mean([mcc(ens, s, sp) for sp in T])
        o = np.mean([max(mcc(graph, s, sp), mcc(seq, s, sp)) for sp in T])
        ens_avg.append(e)
        orc_avg.append(o)
        for sp in T:
            per_set.setdefault(sp, {'graph': [], 'sequence': [], 'ensemble': []})
            per_set[sp]['graph'].append(mcc(graph, s, sp))
            per_set[sp]['sequence'].append(mcc(seq, s, sp))
            per_set[sp]['ensemble'].append(mcc(ens, s, sp))
    print(f'  ensemble          {np.mean(ens_avg):.3f} ± {np.std(ens_avg, ddof=1):.3f}')
    print(f'  oracle selection  {np.mean(orc_avg):.3f} ± {np.std(orc_avg, ddof=1):.3f}')
    print(f'  difference        {np.mean(ens_avg) - np.mean(orc_avg):+.3f}  '
          f'({sum(1 for a, b in zip(ens_avg, orc_avg) if a > b)}/5 seeds)')
    for sp in T:
        print(f'    {sp:<6} graph={np.mean(per_set[sp]["graph"]):.3f} '
              f'sequence={np.mean(per_set[sp]["sequence"]):.3f} '
              f'ensemble={np.mean(per_set[sp]["ensemble"]):.3f}')
    res['oracle'] = {'ensemble': float(np.mean(ens_avg)),
                     'oracle_selection': float(np.mean(orc_avg)),
                     'delta': float(np.mean(ens_avg) - np.mean(orc_avg)),
                     'per_benchmark': {k: {kk: float(np.mean(vv)) for kk, vv in v.items()}
                                       for k, v in per_set.items()}}







    print('\nNegative control: replace the second member with a constant probability of 0.5')
    for name, member in [('graph tower', graph), ('sequence tower', seq)]:
        const = {s: {'acc': {k: {'p': np.full_like(member[s]['acc'][k]['p'], 0.5),
                                 'y': member[s]['acc'][k]['y']} for k in member[s]['acc']},
                     'thr': member[s]['thr']} for s in SEEDS}
        ctrl = ensemble_pack(member, const)
        ctrl_avg = float(np.mean([np.mean([mcc(ctrl, s, sp) for sp in T]) for s in SEEDS]))
        alone_avg = float(np.mean([np.mean([mcc(member, s, sp) for sp in T]) for s in SEEDS]))
        print(f'  {name:<15} alone = {alone_avg:.3f}   + uninformative member = {ctrl_avg:.3f}'
              f'   difference {ctrl_avg - alone_avg:+.4f}')
        res.setdefault('control', {})[name] = {'alone': alone_avg, 'with_constant': ctrl_avg}

    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\n[written] {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
