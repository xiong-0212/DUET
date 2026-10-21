from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from graph_builder import build_one
from rgcn_model import build_model
import ss as SS

DEV_SEEDS = [42, 0, 1, 2, 3]
CONF_SEEDS = [7, 11, 13, 17, 23]
SEEDS = DEV_SEEDS + CONF_SEEDS
MEMBERS = {s: (('clean_seq_ss', 'slim_tcn', 'slim_dense') if s in DEV_SEEDS
               else ('ns_graph', 'ns_tcn', 'ns_dense')) for s in SEEDS}
TRAIN_GRAPHS = {'graph': 'graphs_seq_ss_clean.pt', 'seq': 'graphs_dual_clean.pt'}
EXT_DIR = os.path.join(ROOT, 'data', 'external')
EXT_SETS = os.path.join(EXT_DIR, 'external_sets.json')
REL = [0, 1]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
THR_GRID = np.linspace(0.05, 0.95, 91)
METRICS = ['MCC', 'AUC', 'AUPRC', 'F1', 'Precision', 'Sensitivity', 'Specificity', 'GMean']
OUT = os.path.join(ROOT, 'results', 'external_eval.json')
OUT_MD = os.path.join(ROOT, 'docs', 'tables', 'external_validation_2026-10-16.md')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fold_external(sets):
    
    tsv = os.path.join(EXT_DIR, 'ss_external.tsv')
    if os.path.exists(tsv):
        return SS.load_folds(tsv)
    fa = os.path.join(EXT_DIR, 'ss_external.fa')
    with open(fa, 'w', encoding='utf-8') as f:
        for name, recs in sets.items():
            for cid, r in recs.items():
                f.write(f'>{cid}\n{r["sequence"]}\n')

    wsl_path = '/mnt/' + fa[0].lower() + fa[2:].replace('\\', '/')
    cmd = ['wsl', '-e', 'bash', '-c', f'RNAfold --noPS < "{wsl_path}"']
    print('[fold]', ' '.join(cmd), flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    lines = [l for l in res.stdout.splitlines() if l.strip()]
    out, rid, i = [], None, 0
    while i < len(lines):
        if lines[i].startswith('>'):
            rid = lines[i][1:].strip()
            i += 1
            if i < len(lines):
                i += 1
            if i < len(lines):
                db = lines[i].split()[0]
                out.append(f'{rid}\t{db}')
            i += 1
        else:
            i += 1
    open(tsv, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
    print(f'[fold] {len(out)} chains folded', flush=True)
    return SS.load_folds(tsv)


def embeddings(sets):
    
    f_fm = os.path.join(EXT_DIR, 'emb_rnafm.pt')
    f_ri = os.path.join(EXT_DIR, 'emb_rinalmo.pt')
    seqs = [(cid, r['sequence']) for name, recs in sets.items() for cid, r in recs.items()]
    if os.path.exists(f_fm) and os.path.exists(f_ri):
        a = torch.load(f_fm, map_location='cpu', weights_only=False)
        b = torch.load(f_ri, map_location='cpu', weights_only=False)
        return a, (b['embeddings'] if 'embeddings' in b else b)

    m1 = load_module(os.path.join(ROOT, 'scripts', '01_extract_rnafm_embeddings.py'), 'ext1')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, alphabet = m1.load_rna_fm(device=dev)
    embs = m1.embed_sequences(seqs, model, alphabet, device=dev)
    torch.save(embs, f_fm)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    m2 = load_module(os.path.join(ROOT, 'scripts', '02_extract_rinalmo_embeddings.py'), 'ext2')
    cfg = m2.model_config('giga')
    cfg.model.transformer.use_flash_attn = False
    model = m2.RiNALMo(cfg)
    model.load_state_dict(torch.load(m2.RINALMO_CKPT, map_location='cpu'), strict=True)
    model = model.to(dev).eval()
    alphabet2 = m2.Alphabet(**cfg['alphabet'])
    out = {}
    with torch.no_grad():
        for cid, seq in seqs:
            tokens = torch.tensor(alphabet2.batch_tokenize([seq]), dtype=torch.int64, device=dev)
            rep = model(tokens)['representation'].squeeze(0)
            out[cid] = rep[1:1 + len(seq)].half().cpu()
    torch.save({'embeddings': out, 'dim': 1280}, f_ri)
    return embs, out


def build_external_graphs(sets, folds, emb_fm, emb_ri):
    
    out = {}
    for name, dual in [('ss', False), ('dual', True)]:
        graphs = []
        for set_name, recs in sets.items():
            for cid, r in recs.items():
                sample = {'rna_id': cid, 'sequence': r['sequence'], 'labels': r['labels']}
                e_fm = emb_fm.get(cid)
                if e_fm is None:
                    print(f'  [skip] {cid}: no RNA-FM embedding')
                    continue
                e = e_fm
                if dual:
                    e_ri = emb_ri.get(cid)
                    if e_ri is None:
                        print(f'  [skip] {cid}: no RiNALMo embedding')
                        continue
                    e = torch.cat([e_fm.float(), e_ri.float()], dim=1)
                g, reason = build_one(sample, e.numpy(), folds, REL)
                if g is None:
                    print(f'  [skip] {cid}: {reason}')
                    continue
                g.set_name = set_name
                gsplits = getattr(g, 'split', None)
                graphs.append(g)
        path = os.path.join(EXT_DIR, f'graphs_ext_{name}.pt')
        obj = {'graphs': {'ext': graphs}, 'in_dim': int(graphs[0].x.size(1)) if graphs else None,
               'rel_names': ['seq', 'ss'], 'preset': 'seq_ss'}
        torch.save(obj, path)
        out[name] = path
        print(f'[graphs] {name}: {len(graphs)} chains -> {os.path.basename(path)}', flush=True)
    return out


def predict(model, graphs):
    model.eval()
    acc = {}
    with torch.no_grad():
        for g in graphs:
            z = model(g.x.to(DEV), g.edge_index.to(DEV), g.edge_type.to(DEV),
                      torch.zeros(g.x.size(0), dtype=torch.long, device=DEV))
            if isinstance(z, tuple):
                z = z[0]
            acc[g.rna_id] = {'p': torch.sigmoid(z).float().cpu().numpy(),
                             'y': g.y.numpy().astype(int),
                             'set': getattr(g, 'set_name', 'ext')}
    return acc


def load_model(tag, seed, graphs_file):
    gobj = torch.load(os.path.join(ROOT, 'data', 'processed', graphs_file),
                      map_location='cpu', weights_only=False)
    ck = torch.load(os.path.join(ROOT, 'checkpoints', 'controlled', f'{tag}_seed{seed}.pt'),
                    map_location='cpu', weights_only=False)
    cfg = dict(ck['config']['model'])
    cfg['num_relations'] = ck['num_relations']
    cfg['in_dim'] = gobj['in_dim']
    m = build_model(cfg, ck['num_relations']).to(DEV)
    m.load_state_dict(ck['state_dict'])
    return m, gobj, float(ck['threshold'])


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-embeddings', action='store_true')
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    sets = json.load(open(EXT_SETS, encoding='utf-8'))
    print('external sets:', {k: len(v) for k, v in sets.items()})
    folds = fold_external(sets)
    missing = [cid for recs in sets.values() for cid in recs if len(folds.get(cid, '')) != len(recs[cid]['sequence'])]
    if missing:
        print(f'[warn] {len(missing)} chains without a length-consistent fold: {missing[:5]}')
    emb_fm, emb_ri = embeddings(sets) if not a.skip_embeddings else (
        torch.load(os.path.join(EXT_DIR, 'emb_rnafm.pt'), map_location='cpu', weights_only=False),
        torch.load(os.path.join(EXT_DIR, 'emb_rinalmo.pt'), map_location='cpu',
                   weights_only=False)['embeddings'])
    paths = build_external_graphs(sets, folds, emb_fm, emb_ri)

    ext_ss = torch.load(paths['ss'], map_location='cpu', weights_only=False)['graphs']['ext']
    ext_dual = torch.load(paths['dual'], map_location='cpu', weights_only=False)['graphs']['ext']
    by_set = {}
    for g in ext_ss:
        by_set.setdefault(g.set_name, []).append(g.rna_id)
    print('per set:', {k: len(v) for k, v in by_set.items()})


    val_g = {}
    for which, f in TRAIN_GRAPHS.items():
        gobj = torch.load(os.path.join(ROOT, 'data', 'processed', f), map_location='cpu',
                          weights_only=False)
        leak = json.load(open(os.path.join(ROOT, 'data', 'leaked_pairs.json'), encoding='utf-8'))
        drop = set(leak['remove_val_ids'])
        val_g[which] = [g for g in gobj['graphs']['val'] if g.rna_id not in drop]

    per_seed = {name: {m: [] for m in METRICS} for name in by_set}
    for s in SEEDS:
        graph_tag, tcn_tag, dense_tag = MEMBERS[s]
        m_g, _, _ = load_model(graph_tag, s, TRAIN_GRAPHS['graph'])
        m_t, _, _ = load_model(tcn_tag, s, TRAIN_GRAPHS['seq'])
        m_d, _, _ = load_model(dense_tag, s, TRAIN_GRAPHS['seq'])
        members_val = [predict(m_g, val_g['graph']),
                       predict(m_t, val_g['seq']),
                       predict(m_d, val_g['seq'])]
        members_ext = [predict(m_g, ext_ss), predict(m_t, ext_dual), predict(m_d, ext_dual)]
        p_val = {k: {'p': np.mean([m[k]['p'] for m in members_val], axis=0),
                     'y': members_val[0][k]['y']} for k in members_val[0]}
        p_ext = {k: {'p': np.mean([m[k]['p'] for m in members_ext], axis=0),
                     'y': members_ext[0][k]['y'],
                     'set': members_ext[0][k]['set']} for k in members_ext[0]}
        vy = np.concatenate([v['y'] for v in p_val.values()])
        vp = np.concatenate([v['p'] for v in p_val.values()])
        thr = float(THR_GRID[int(np.argmax([matthews_corrcoef(vy, (vp >= t).astype(int))
                                            for t in THR_GRID]))])
        for set_name, ids in by_set.items():
            y = np.concatenate([p_ext[i]['y'] for i in ids])
            p = np.concatenate([p_ext[i]['p'] for i in ids])
            mm = met(y, p, thr)
            for k in METRICS:
                per_seed[set_name][k].append(mm[k])
        print(f"  seed {s}: threshold {thr:.2f}  "
              + '  '.join(f"{n}: MCC {np.mean([per_seed[n]['MCC'][-1]]):.3f}" for n in by_set),
              flush=True)

    res = {'per_set': {}, 'n_seeds': len(SEEDS)}
    for set_name in by_set:
        res['per_set'][set_name] = {m: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)),
                                        'per_seed': [float(x) for x in v]}
                                    for m, v in per_seed[set_name].items()}
        res['per_set'][set_name]['n_chains'] = len(by_set[set_name])
        res['per_set'][set_name]['n_residues'] = int(sum(len(p_ext[i]['y']) for i in by_set[set_name]))
    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    lines = ['# External validation (SMARTPocket external / time-dependent splits, 2026-10-16)', '',
             '> After de-leaking (`scripts/32_external_data.py`) + a 10-seed three-tower ensemble; each seed selects its threshold on **its own',
             '> training-validation set**, and the external data take no part in any tuning. Labels follow the SMARTPocket criterion (positive ratio',
             '> 0.11–0.20, sparser than our benchmarks), so columns compare models only, never absolute values across datasets.', '',
             '| External set | Chains | Residues | MCC | AUC | AUPRC | F1 | Precision | Sensitivity | Specificity |',
             '|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|']
    for set_name, d in res['per_set'].items():
        f = lambda m: f"{d[m]['mean']:.3f} ± {d[m]['sd']:.3f}"
        lines.append(f"| {set_name} | {d['n_chains']} | {d['n_residues']} | {f('MCC')} | {f('AUC')} | "
                     f"{f('AUPRC')} | {f('F1')} | {f('Precision')} | {f('Sensitivity')} | "
                     f"{f('Specificity')} |")
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print(f'\n[written] {OUT}\n[written] {OUT_MD}')
    for set_name, d in res['per_set'].items():
        print(f"  {set_name}: MCC {d['MCC']['mean']:.3f} ± {d['MCC']['sd']:.3f}  "
              f"AUC {d['AUC']['mean']:.3f}  ({d['n_chains']} chains)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
