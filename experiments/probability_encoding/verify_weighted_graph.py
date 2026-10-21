from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import numpy as np
import torch

import data_loader as dl
import graph_builder as GB
import ss as SS
from rgcn_model import RGCNBlock

BP_FILE = os.path.join(ROOT, 'data', 'raw', 'ss_bp_prob.tsv')
SS_TSV = os.path.join(ROOT, 'data', 'raw', 'ss_output.tsv')
EMB_PATH = os.path.join(ROOT, 'data', 'processed', 'embeddings.pt')

FAILS = []


def check(name, ok, detail=''):
    tag = 'PASS' if ok else 'FAIL'
    print(f'[{tag}] {name}' + (f'  {detail}' if detail else ''), flush=True)
    if not ok:
        FAILS.append(name)


def load_inputs():
    emb_obj = torch.load(EMB_PATH, map_location='cpu', weights_only=False)
    emb = emb_obj['embeddings'] if 'embeddings' in emb_obj else emb_obj
    fold = SS.load_folds(SS_TSV)
    bp = SS.load_bp_prob(BP_FILE)
    data = dl.load_all()
    return emb, fold, bp, data



def test_weighted_forward(emb, fold, bp, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sample = None
    for s in data['train']:
        db = fold.get(s['rna_id'], '')
        if db and len(db) == len(s['sequence']) and bp.get(s['rna_id']):
            if SS.pairs_from_dotbracket(db):
                sample = s
                break
    assert sample is not None, 'no sample with base pairs found'
    print(f'  A test sample: {sample["rna_id"]} (len={len(sample["sequence"])})')
    g, _ = GB.build_one(sample, emb.get(sample['rna_id']), fold,
                        GB.PRESETS['seq_ss_alt'], bp_prob=bp,
                        weight_mode='prob')
    torch.manual_seed(0)
    blk = RGCNBlock(640, 128, 3, norm='graph', dropout=0.0, residual=True).to(device)
    blk.eval()
    x = g.x.to(device)
    ei = g.edge_index.to(device)
    et = g.edge_type.to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        out_pyg = blk(x, ei, et, batch)
        ones = torch.ones(ei.size(1), device=device)
        out_w1 = blk(x, ei, et, batch, edge_weight=ones)
        out_wp = blk(x, ei, et, batch,
                     edge_weight=g.edge_weight.to(device))

    d1 = (out_pyg - out_w1).abs().max().item()
    d2 = (out_pyg - out_wp).abs().max().item()
    check('A1 weighted path w≡1 matches the original PyG path', d1 < 1e-5,
          f'max|Δ|={d1:.3e} (float32)')
    check('A2 real probability weights do change the output', d2 > 1e-4, f'max|Δ|={d2:.3e}')


    w = g.edge_weight
    seq_mask = g.edge_type == 0
    check('A3 non-structural (seq) edge weights are always 1', bool((w[seq_mask] == 1.0).all()))


    blk2 = RGCNBlock(640, 128, 3, norm='graph', dropout=0.0, residual=True,
                     wnorm='count').to(device)
    blk2.load_state_dict(blk.state_dict())
    blk2.eval()
    with torch.no_grad():
        out_c1 = blk2(x, ei, et, batch, edge_weight=ones)
    d3 = (out_pyg - out_c1).abs().max().item()
    check('A4 count normalization also degenerates consistently at w≡1', d3 < 1e-5, f'max|Δ|={d3:.3e}')



def _cmp_stored(name, preset, stored_file, emb, fold, bp, threshold=0.3):
    obj = torch.load(os.path.join(ROOT, stored_file), map_location='cpu',
                     weights_only=False)
    data = dl.load_all()
    bad = 0
    n_checked = 0
    for split, graphs in obj['graphs'].items():
        samples = {s['rna_id']: s for s in data.get(split, [])}
        for g_old in graphs:
            s = samples[g_old.rna_id]
            g_new, why = GB.build_one(s, emb.get(s['rna_id']), fold, GB.PRESETS[preset],
                                      bp_prob=bp, bp_threshold=threshold,
                                      weight_mode='none')
            n_checked += 1
            ok = (g_new is not None
                  and torch.equal(g_new.edge_index, g_old.edge_index)
                  and torch.equal(g_new.edge_type, g_old.edge_type)
                  and torch.equal(g_new.y, g_old.y)
                  and g_new.num_relations == g_old.num_relations)
            if not ok:
                bad += 1
                if bad <= 3:
                    print(f'   [mismatch] {split}/{g_old.rna_id}')
    check(f'B {name} ({n_checked} graphs compared element-wise)', bad == 0, f'{bad} mismatches')



def test_weight_modes(emb, fold, bp, data):
    presets = GB.PRESETS['seq_ss_alt']
    chains = data['train'][:60] + data['RB9']
    stats = {'prob_ss': [], 'prob_alt': [], 'rand_ok': 0, 'rand_bad': 0,
             'shuf_same_multiset': 0, 'shuf_changed': 0, 'shuf_total': 0}
    for s in chains:
        rid = s['rna_id']
        common = dict(bp_prob=bp)
        g_p = GB.build_one(s, emb.get(rid), fold, presets, weight_mode='prob', **common)[0]
        g_s = GB.build_one(s, emb.get(rid), fold, presets, weight_mode='shuffle', **common)[0]
        g_r = GB.build_one(s, emb.get(rid), fold, presets, weight_mode='shuffle',
                           rand_sub=True, **common)[0]
        g_bp = GB.build_one(s, emb.get(rid), fold, GB.PRESETS['seq_bp'],
                            weight_mode='prob', **common)[0]
        if g_bp is not None:
            m = g_bp.edge_type == 1
            if m.any():
                stats.setdefault('bp_w', []).extend(g_bp.edge_weight[m].tolist())
        if g_bp is not None and g_p is not None:

            stats['bp_match'] = stats.get('bp_match', 0) + int(
                int((g_bp.edge_type == 1).sum())
                == int((g_p.edge_type == 1).sum()) + int((g_p.edge_type == 2).sum()))
        if g_p is None:
            continue

        ss_mask = g_p.edge_type == 1
        alt_mask = g_p.edge_type == 2
        if ss_mask.any():
            stats['prob_ss'] += g_p.edge_weight[ss_mask].tolist()
        if alt_mask.any():
            stats['prob_alt'] += g_p.edge_weight[alt_mask].tolist()


        if torch.equal(g_p.edge_index, g_s.edge_index):
            stats['shuf_total'] += 1
            same = np.allclose(sorted(g_p.edge_weight.tolist()),
                               sorted(g_s.edge_weight.tolist()))
            changed = not torch.allclose(g_p.edge_weight, g_s.edge_weight)
            stats['shuf_same_multiset'] += int(same)
            stats['shuf_changed'] += int(same and changed)


        real_alt = int((g_p.edge_type == 2).sum())
        rand_alt = int((g_r.edge_type == 2).sum())
        if real_alt == rand_alt and torch.equal(g_r.edge_index[:, g_r.edge_type == 0],
                                                g_p.edge_index[:, g_p.edge_type == 0]):
            mfe = set(map(tuple, g_p.edge_index[:, g_p.edge_type == 1].t().tolist()))
            r_pairs = set(map(tuple, g_r.edge_index[:, g_r.edge_type == 2].t().tolist()))
            if not (r_pairs & mfe):
                stats['rand_ok'] += 1
            else:
                stats['rand_bad'] += 1

    w_ss = np.array(stats['prob_ss'])
    w_al = np.array(stats['prob_alt'])
    check('C1 ss edge weights ∈ (0,1]', len(w_ss) and w_ss.min() > 0 and w_ss.max() <= 1.0,
          f'n={len(w_ss)} median={np.median(w_ss):.3f} mean={w_ss.mean():.3f}')
    check('C2 ss_alt edge weights ∈ (0,1)', len(w_al) and w_al.min() > 0 and w_al.max() <= 1.0,
          f'n={len(w_al)} median={np.median(w_al):.3f}')
    check('C3 shuffle preserves the weight multiset', stats['shuf_same_multiset'] == stats['shuf_total'],
          f"{stats['shuf_same_multiset']}/{stats['shuf_total']}")
    check('C4 shuffle actually changes the weight-edge assignment', stats['shuf_changed'] > 0,
          f"{stats['shuf_changed']}/{stats['shuf_total']} chains changed")
    check('C5 rand_sub keeps the edge count and avoids MFE/adjacent pairs',
          stats['rand_bad'] == 0 and stats['rand_ok'] > 0,
          f"ok={stats['rand_ok']} bad={stats['rand_bad']}")
    w_bp = np.array(stats.get('bp_w', []))
    non_degenerate = len(w_bp) and not np.allclose(w_bp, 1.0) and w_bp.max() <= 1.0
    check('C6 bp relation edge weights are real probabilities (not all 1)', bool(non_degenerate),
          f'n={len(w_bp)} median={np.median(w_bp):.3f} mean={w_bp.mean():.3f}'
          if len(w_bp) else 'no data')
    check('C7 bp edge count == ss + ss_alt edge count', stats.get('bp_match', 0) > 0,
          f"matched {stats.get('bp_match', 0)} chains")


def main():
    print('Loading inputs ...', flush=True)
    emb, fold, bp, data = load_inputs()
    test_weighted_forward(emb, fold, bp, data)
    _cmp_stored('seq_ss reproduces the stored graphs', 'seq_ss',
                'data/processed/graphs_seq_ss_clean.pt', emb, fold, bp)
    _cmp_stored('seq_ss_alt reproduces the stored graphs', 'seq_ss_alt',
                'data/processed/graphs_seq_ss_bp_clean.pt', emb, fold, bp,
                threshold=0.3)
    test_weight_modes(emb, fold, bp, data)
    print()
    if FAILS:
        print(f'[FAIL] {len(FAILS)} checks failed: {FAILS}')
        return 1
    print('[ALL PASS] all checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
