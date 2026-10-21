from __future__ import annotations

import os, sys, json, zlib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, HERE)

import ss as SS
import data_loader as dl




REL_SEQ = 0
REL_SS = 1
REL_SS_ALT = 2
REL_SIM = 3
REL_BP = 4

REL_NAMES = {REL_SEQ: 'seq', REL_SS: 'ss', REL_SS_ALT: 'ss_alt', REL_SIM: 'sim',
             REL_BP: 'bp'}


STRUCT_RELS = {REL_SS, REL_SS_ALT, REL_BP}


PRESETS = {
    'seq_only':       [REL_SEQ],
    'seq_ss':         [REL_SEQ, REL_SS],
    'seq_ss_alt':     [REL_SEQ, REL_SS, REL_SS_ALT],
    'seq_ss_sim':     [REL_SEQ, REL_SS, REL_SIM],
    'seq_ss_alt_sim': [REL_SEQ, REL_SS, REL_SS_ALT, REL_SIM],
    'seq_bp':         [REL_SEQ, REL_BP],
}

WEIGHT_MODES = ('none', 'ones', 'prob', 'shuffle')

DEFAULT_BP_THRESHOLD = 0.3
SIM_K = 4

GRAPH_OUT = os.path.join(ROOT, 'data', 'processed', 'graphs_seq_ss_bp.pt')





def _undirected(pairs, n):
    
    if not pairs:
        return torch.zeros(2, 0, dtype=torch.long)
    src = torch.tensor([p[0] for p in pairs] + [p[1] for p in pairs], dtype=torch.long)
    dst = torch.tensor([p[1] for p in pairs] + [p[0] for p in pairs], dtype=torch.long)
    return torch.stack([src, dst])


def sim_edges(emb, k: int = SIM_K):
    
    x = torch.nn.functional.normalize(emb.float(), dim=-1)
    n = x.size(0)
    if n < 2:
        return []
    sim = x @ x.t()
    sim.fill_diagonal_(-2.0)
    kk = min(k, n - 1)
    idx = torch.topk(sim, kk, dim=-1).indices.numpy()
    pairs = set()
    for i in range(n):
        for j in idx[i]:
            j = int(j)
            if abs(i - j) <= 1:
                continue
            pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)





def _stable_seed(rid: str) -> int:
    
    return (zlib.crc32(str(rid).encode('utf-8')) ^ 0x50524953) & 0xFFFFFFFF


NODE_FEAT_NAMES = ['prob_paired', 'max_prob', 'partner_entropy', 'n_partners', 'mfe_prob']


def prob_node_features(sample, probs, mfe_pairs):
    
    n = len(sample['sequence'])
    F = np.zeros((n, len(NODE_FEAT_NAMES)), dtype=np.float32)
    acc = {}
    for (i, j), p in probs.items():
        if j >= n or p <= 0:
            continue
        acc.setdefault(i, []).append(p)
        if j != i:
            acc.setdefault(j, []).append(p)
    for i in range(n):
        ps = acc.get(i, [])
        if ps:
            s = float(sum(ps))
            F[i, 0] = min(s, 1.0)
            F[i, 1] = float(max(ps))
            F[i, 2] = float(-sum((p / s) * np.log(p / s) for p in ps if p > 0))
            F[i, 3] = float(sum(1 for p in ps if p >= 0.1))
    for (i, j) in mfe_pairs:
        if j < n:
            v = float(probs.get((i, j), 1.0))
            F[i, 4] = v
            F[j, 4] = v
    return F


def _random_pairs(n, count, exclude, rng):
    
    if count <= 0 or n < 4:
        return []
    cand = [(i, j) for i in range(n) for j in range(i + 2, n) if (i, j) not in exclude]
    if not cand:
        return []
    if count <= len(cand):
        idx = rng.choice(len(cand), size=int(count), replace=False)
        return sorted(cand[int(k)] for k in idx)
    extra = rng.choice(len(cand), size=int(count) - len(cand), replace=True)
    return sorted(list(cand) + [cand[int(k)] for k in extra])


def build_one(sample, emb, fold, rels, bp_prob=None, bp_threshold=DEFAULT_BP_THRESHOLD,
              sim_k=SIM_K, weight_mode='none', rand_sub=False, verbose=False,
              node_feats=False, feat_stats=None, feat_shuffle=False,
              mfe_prob_min=None):
    
    n = len(sample['sequence'])
    if emb is None:
        return None, 'no_embedding'
    assert weight_mode in WEIGHT_MODES, f'unknown weight_mode: {weight_mode}'

    rel_map = {r: i for i, r in enumerate(rels)}

    x = torch.tensor(emb, dtype=torch.float)
    y = torch.tensor(sample['labels'], dtype=torch.float)

    rid = sample['rna_id']
    probs = (bp_prob or {}).get(rid, {})
    rng = np.random.default_rng(_stable_seed(rid))

    ei_parts, et_parts, ew_parts = [], [], []

    def add(pairs, rel_internal, weights=None):
        if not pairs:
            return
        ei_parts.append(_undirected(pairs, n))
        et_parts.append(np.full(2 * len(pairs), rel_map[rel_internal], dtype=np.int64))
        if weight_mode != 'none':
            if weights is None:
                weights = [1.0] * len(pairs)
            assert len(weights) == len(pairs), f'{rid}: number of weights {len(weights)} != number of edges {len(pairs)}'
            w = np.asarray(weights, dtype=np.float32)
            ew_parts.append(np.concatenate([w, w]))

    def struct_weights(pairs, prob_values=None):
        
        if weight_mode in ('none', 'ones'):
            return None if weight_mode == 'none' else [1.0] * len(pairs)
        w = [float(probs.get(p, 1.0)) for p in pairs] if prob_values is None \
            else [float(v) for v in prob_values]
        if weight_mode == 'shuffle':
            perm = rng.permutation(len(w))
            w = [w[int(k)] for k in perm]
        return w


    if REL_SEQ in rel_map:
        add([(i, i + 1) for i in range(n - 1)], REL_SEQ)


    db = fold.get(rid, '')
    mfe_pairs = SS.pairs_from_dotbracket(db) if (db and len(db) == n) else []
    mfe_set = set(mfe_pairs)


    if mfe_prob_min is not None:
        mfe_pairs = [(i, j) for (i, j) in mfe_pairs
                     if probs.get((i, j), 1.0) >= mfe_prob_min]


    if node_feats:
        F = prob_node_features(sample, probs, mfe_pairs)
        if feat_stats is not None:
            mu, sd = feat_stats
            F = (F - mu) / sd
        if feat_shuffle:
            F = F[rng.permutation(len(F))]
        x = torch.cat([x, torch.tensor(F, dtype=torch.float)], dim=1)


    need_sub = (REL_SS_ALT in rel_map) or (REL_BP in rel_map)
    sub_pairs, sub_w = [], None
    if need_sub:
        if bp_prob is None and verbose:
            print(f'[warn] {rid}: preset needs a probability matrix but none was supplied')
        real_sub = sorted([(i, j) for (i, j), pr in probs.items()
                           if pr >= bp_threshold and (i, j) not in mfe_set and j < n])
        if rand_sub and real_sub:

            sub_w = [probs[p] for p in real_sub]
            perm = rng.permutation(len(sub_w))
            sub_w = [sub_w[int(k)] for k in perm]
            sub_pairs = _random_pairs(n, len(real_sub), mfe_set, rng)
            if len(sub_pairs) != len(real_sub):
                sub_w = sub_w[:len(sub_pairs)]
        else:
            sub_pairs = real_sub


    if REL_SS in rel_map:
        add(mfe_pairs, REL_SS, struct_weights(mfe_pairs))


    if REL_SS_ALT in rel_map:
        add(sub_pairs, REL_SS_ALT, struct_weights(sub_pairs, sub_w))


    if REL_BP in rel_map:
        bp_pairs = list(mfe_pairs) + list(sub_pairs)
        bp_w = None
        if weight_mode != 'none':
            bp_w = struct_weights(mfe_pairs) + struct_weights(sub_pairs, sub_w)
        add(bp_pairs, REL_BP, bp_w)


    if REL_SIM in rel_map:
        add(sim_edges(emb, k=sim_k), REL_SIM)

    if not ei_parts:
        return None, 'no_edges'

    edge_index = torch.cat(ei_parts, dim=1)
    edge_type = torch.tensor(np.concatenate(et_parts), dtype=torch.long)

    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y,
                num_nodes=n, num_relations=len(rels), rna_id=rid)
    if weight_mode != 'none':
        ew = np.concatenate(ew_parts).astype(np.float32)
        assert len(ew) == edge_index.size(1), \
            f'{rid}: edge_weight length {len(ew)} != number of edges {edge_index.size(1)}'
        data.edge_weight = torch.tensor(ew, dtype=torch.float)
    return data, None





def build(preset='seq_ss_alt', out_path=GRAPH_OUT, verbose=True, bp_prob_file=None,
          remove_leaked=False, bp_threshold=DEFAULT_BP_THRESHOLD, sim_k=SIM_K,
          weight_mode='none', rand_sub=False, prob_min=0.1,
          node_feats=False, feat_shuffle=False, mfe_prob_min=None):
    rels = PRESETS.get(preset)
    if rels is None:
        raise ValueError(f'unknown preset: {preset}, expected one of {sorted(PRESETS)}')
    assert weight_mode in WEIGHT_MODES, f'unknown weight_mode: {weight_mode}'


    emb_path = os.path.join(ROOT, 'data', 'processed', 'embeddings.pt')
    emb_obj = torch.load(emb_path, map_location='cpu', weights_only=False)
    emb = emb_obj['embeddings'] if 'embeddings' in emb_obj else emb_obj


    ss_tsv = os.path.join(ROOT, 'data', 'raw', 'ss_output.tsv')
    fold = SS.load_folds(ss_tsv) if os.path.exists(ss_tsv) else {}


    need_prob = ((REL_SS_ALT in rels) or (REL_BP in rels)
                 or (weight_mode in ('prob', 'shuffle')) or rand_sub or node_feats
                 or (mfe_prob_min is not None))
    bp_prob = None
    if need_prob:
        if not (bp_prob_file and os.path.exists(bp_prob_file)):
            raise FileNotFoundError(
                f'preset={preset} weight_mode={weight_mode} needs RNAfold -p probability matrices, '
                f'but {bp_prob_file} was not found; run scripts/extract_bp_prob.py first')
        bp_prob = SS.load_bp_prob(bp_prob_file, prob_min=prob_min)
        if verbose:
            print(f'base-pair probabilities: {len(bp_prob)} RNAs'
                  f' (read cutoff P>={prob_min}, edge cutoff P>={bp_threshold})', flush=True)

    data = dl.load_all()

    if remove_leaked:
        leaked_path = os.path.join(ROOT, 'data', 'leaked_pairs.json')
        if os.path.exists(leaked_path):
            leaked = json.load(open(leaked_path, encoding='utf-8'))
            remove_set = set(leaked['remove_train_ids'])
            n_before = len(data.get('train', []))
            data['train'] = [s for s in data['train'] if s['rna_id'] not in remove_set]
            if verbose:
                print(f'[clean] removed {n_before - len(data["train"])} leaked training chains', flush=True)

    rel_names = [REL_NAMES[r] for r in rels]
    if verbose:
        print(f'preset={preset}  relations={rel_names}  num_relations={len(rels)}  '
              f'weight_mode={weight_mode}  rand_sub={rand_sub}', flush=True)


    feat_stats = None
    if node_feats:
        feats = []
        for s in data['train']:
            db = fold.get(s['rna_id'], '')
            mfe = (SS.pairs_from_dotbracket(db)
                   if (db and len(db) == len(s['sequence'])) else [])
            feats.append(prob_node_features(s, (bp_prob or {}).get(s['rna_id'], {}), mfe))
        F = np.concatenate(feats, axis=0)
        mu, sd = F.mean(axis=0), F.std(axis=0)
        sd[sd < 1e-6] = 1.0
        feat_stats = (mu.astype(np.float32), sd.astype(np.float32))
        if verbose:
            print('[node feats] training-set statistics (mean/std): '
                  + ' '.join(f'{n}={m:.3f}/{v:.3f}'
                             for n, m, v in zip(NODE_FEAT_NAMES, mu, sd)), flush=True)

    out, stats = {}, {}
    for split, samples in data.items():
        graphs, reasons = [], {}
        for s in samples:
            g, why = build_one(s, emb.get(s['rna_id']), fold, rels,
                               bp_prob=bp_prob, bp_threshold=bp_threshold, sim_k=sim_k,
                               weight_mode=weight_mode, rand_sub=rand_sub,
                               node_feats=node_feats, feat_stats=feat_stats,
                               feat_shuffle=feat_shuffle, mfe_prob_min=mfe_prob_min)
            if g is None:
                reasons[why] = reasons.get(why, 0) + 1
                continue
            graphs.append(g)
        out[split] = graphs
        stats[split] = {'graphs': len(graphs), 'dropped': reasons}
        if verbose:
            print(f'{split:<18s} graphs={len(graphs):4d}  dropped={reasons}', flush=True)

    total_nodes = sum(g.x.shape[0] for split in out for g in out[split])
    total_edges = sum(g.edge_index.shape[1] for split in out for g in out[split])


    rel_edge_counts = {}
    for split in ('train', 'val'):
        cnt = {name: 0 for name in rel_names}
        for g in out.get(split, []):
            for r, name in enumerate(rel_names):
                cnt[name] += int((g.edge_type == r).sum()) // 2
        rel_edge_counts[split] = cnt

    obj = {
        'graphs': out,
        'preset': preset,
        'num_relations': len(rels),
        'rel_names': rel_names,
        'in_dim': int(out['train'][0].x.shape[1]),
        'bp_prob': bool(bp_prob),
        'bp_threshold': float(bp_threshold) if bp_prob else None,
        'sim_k': int(sim_k) if REL_SIM in rels else None,
        'removed_leaked_train': bool(remove_leaked),
        'weight_mode': weight_mode,
        'rand_sub': bool(rand_sub),
        'has_edge_weight': weight_mode != 'none',
        'node_feats': bool(node_feats),
        'feat_shuffle': bool(feat_shuffle),
        'mfe_prob_min': float(mfe_prob_min) if mfe_prob_min is not None else None,
        'rel_edge_counts': rel_edge_counts,
        'stats': stats,
        'total_nodes': total_nodes,
        'total_edges': total_edges,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(obj, out_path)
    if verbose:
        print(f'saved {out_path}  preset={preset} rel={rel_names} '
              f'nodes={total_nodes} edges={total_edges}', flush=True)
    return obj





if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--preset', default='seq_ss_alt', choices=sorted(PRESETS))
    p.add_argument('--out', default=GRAPH_OUT)
    p.add_argument('--bp_prob_file', default=None, help='RNAfold -p base-pair probability TSV')
    p.add_argument('--bp_threshold', type=float, default=DEFAULT_BP_THRESHOLD,
                   help='probability cutoff for ss_alt edges (default 0.3)')
    p.add_argument('--sim_k', type=int, default=SIM_K, help='top-k for sim edges (default 4)')
    p.add_argument('--remove_leaked', action='store_true',
                   help='remove training chains whose sequence is identical (100%%) to the test set')
    p.add_argument('--weight_mode', default='none', choices=list(WEIGHT_MODES),
                   help='probability edge-weight mode (none/ones/prob/shuffle)')
    p.add_argument('--rand_sub', action='store_true',
                   help='replace suboptimal edges with the same number of random pairs (control condition)')
    p.add_argument('--prob_min', type=float, default=0.1,
                   help='lower cutoff when reading the probability matrices (lower it in step with a low-threshold TSV)')
    p.add_argument('--node_feats', action='store_true',
                   help='concatenate the 5 probability-derived node features (standardized with training-set statistics)')
    p.add_argument('--feat_shuffle', action='store_true',
                   help='shuffle node features within each row (control condition)')
    p.add_argument('--mfe_prob_min', type=float, default=None,
                   help='keep only MFE pairs with P ≥ this value (confidence pruning)')
    a = p.parse_args()

    bp_file = a.bp_prob_file or os.path.join(ROOT, 'data', 'raw', 'ss_bp_prob.tsv')
    if not os.path.exists(bp_file):
        bp_file = os.path.join(ROOT, 'data', 'raw', os.path.basename(bp_file))

    build(a.preset, a.out, bp_prob_file=bp_file, remove_leaked=a.remove_leaked,
          bp_threshold=a.bp_threshold, sim_k=a.sim_k, prob_min=a.prob_min,
          weight_mode=a.weight_mode, rand_sub=a.rand_sub,
          node_feats=a.node_feats, feat_shuffle=a.feat_shuffle,
          mfe_prob_min=a.mfe_prob_min)
