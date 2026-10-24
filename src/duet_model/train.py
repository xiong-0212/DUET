from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'configs')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, HERE)



from utils import (compute_metrics, drop_edge, find_best_threshold,
                   save_json, set_seed)

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler

import losses as L
from rgcn_model import build_model


def load_cfg(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)



TEST_SPLITS = ['TE18', 'TL12', 'JL10', 'RB9', 'hariofficial_test']


def graphs_path(cfg, preset):
    return os.path.join(ROOT, cfg['data']['graphs_dir'], f'graphs_{preset}.pt')


def load_graphs(cfg, preset, graphs_file=None):
    p = graphs_file or graphs_path(cfg, preset)
    if not os.path.isabs(p):
        p = os.path.join(ROOT, p)
    if not os.path.exists(p):
        raise FileNotFoundError(f'graph file not found: {p}; run graph_builder.py --preset {preset} first')
    return torch.load(p, weights_only=False)


def to_device(data, device):
    d = data.to(device)
    return d


@torch.no_grad()
def collect_preds(model, loader, device, drop_edge_p=0.0):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        batch = to_device(batch, device)
        ew = batch.edge_weight if hasattr(batch, 'edge_weight') else None
        logit = model(batch.x, batch.edge_index, batch.edge_type,
                      getattr(batch, 'batch', None), edge_weight=ew)
        probs.append(torch.sigmoid(logit).float().cpu().numpy())
        labels.append(batch.y.cpu().numpy())
    return np.concatenate(labels), np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(ROOT, 'configs', 'default.yaml'))
    ap.add_argument('--tag', default='full')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--preset', default=None)
    ap.add_argument('--graphs', default=None,
                    help='specify the graph file directly (for variants such as the node-feature version); '
                         'takes precedence over --preset')
    ap.add_argument('--val_from', default=None,
                    help='Leave-one-benchmark-out: use one test set as the validation set (the other test sets '
                         'are still evaluated), to check whether the original validation set sharing '
                         'its source with the training set misleads model selection')
    ap.add_argument('--arch', default=None, choices=['rgcn', 'mlp', 'dense', 'tcn', 'transformer', 'fusion', 'fusion_avg', 'fusion_concat', 'dualview', 'dualview_nogate', 'dual2', 'dual2_avg', 'dual2_logit', 'dual2_graph', 'dual2_mlp', 'fusion_mamba', 'dualview_mamba', 'fusion_mamba_official', 'dualview_mamba_official'])
    ap.add_argument('--loss', default=None)
    ap.add_argument('--layers', type=int, default=None)
    ap.add_argument('--hidden', type=int, default=None)
    ap.add_argument('--head_hidden', type=int, default=None,
                    help='classification-head width; default hidden//2. Used to diagnose the dimensionality-reduction bottleneck')
    ap.add_argument('--input_skip', action='store_true',
                    help='concatenate the raw input features directly onto the classification head (so they are '
                         'not narrowed by the graph convolution)')
    ap.add_argument('--dropout', type=float, default=None)
    ap.add_argument('--drop_edge', type=float, default=None)
    ap.add_argument('--no_residual', action='store_true')
    ap.add_argument('--norm', default=None)
    ap.add_argument('--model_kv', default=None,
                    help='extra model keys, e.g. "ch=160" or "stem=160,growth=40"')
    ap.add_argument('--mlp_dropout', type=float, default=None,
                    help='dropout inside the sequence tower (dense layers / conv blocks)')
    ap.add_argument('--mlp_widths', default=None,
                    help='comma-separated hidden widths of the sequence-tower MLP '
                         '(e.g. "1325,256,128"); default 1024,256,128')
    ap.add_argument('--split_dim', type=int, default=None,
                    help='dual2 family: node-feature split point (the graph tower takes the first N dims, the MLP tower takes the rest)')
    ap.add_argument('--wnorm', default=None, choices=['wsum', 'count'],
                    help='how probability edge weights are normalized (only applies when the graphs carry edge_weight)')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--no-eval-test', dest='eval_test', action='store_false',
                    help='skip test-set evaluation (by default TE18/TL12/JL10/RB9 are all evaluated after training)')
    ap.add_argument('--val_subsample', type=int, default=None,
                    help='randomly subsample the validation set to N chains (controls the validation-set size '
                         'variable; pair with --val_from for the ablation)')
    ap.add_argument('--drop_val_leaked', action='store_true',
                    help='remove validation chains whose sequence is identical (100%%) to the test set (see '
                         'remove_val_ids in data/leaked_pairs.json; avoids early stopping / threshold '
                         'selection being contaminated by the test set)')
    ap.add_argument('--large_manifest', default=None,
                    help='large-protocol manifest name; select fold train/validation from its pool')
    ap.add_argument('--fold', type=int, default=None,
                    help='CV fold index from the large-protocol manifest')
    ap.add_argument('--all_pool', action='store_true',
                    help='use the whole large-protocol pool for final retraining')
    ap.add_argument('--fixed_epochs', type=int, default=None,
                    help='train this many epochs and retain the final state (final retraining mode)')
    ap.add_argument('--fixed_threshold', type=float, default=None,
                    help='threshold selected by CV; required with --fixed_epochs')
    ap.add_argument('--sample_weight_mode',
                    choices=['group', 'uniform', 'none'], default='group',
                    help='group: 1/cluster size; uniform: equal record weights; none: shuffle')
    ap.set_defaults(eval_test=True)
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    preset = a.preset or cfg['data']['preset']
    if a.arch: cfg['model']['arch'] = a.arch
    if a.loss: cfg['loss']['name'] = a.loss
    if a.layers: cfg['model']['num_layers'] = a.layers
    if a.hidden: cfg['model']['hidden'] = a.hidden
    if a.head_hidden: cfg['model']['head_hidden'] = a.head_hidden
    if a.input_skip: cfg['model']['input_skip'] = True
    if a.dropout is not None: cfg['model']['dropout'] = a.dropout
    if a.norm: cfg['model']['norm'] = a.norm
    if a.wnorm: cfg['model']['wnorm'] = a.wnorm
    if a.split_dim is not None: cfg['model']['split_dim'] = a.split_dim
    if a.model_kv:
        for kv in a.model_kv.split(','):
            k, _, v = kv.partition('=')
            cfg['model'][k.strip()] = int(v) if v.strip().lstrip('-').isdigit() else float(v)
    if a.mlp_dropout is not None: cfg['model']['mlp_dropout'] = a.mlp_dropout
    if a.mlp_widths:
        cfg['model']['mlp_widths'] = [int(x) for x in a.mlp_widths.split(',')]
    if a.no_residual: cfg['model']['residual'] = False
    if a.epochs: cfg['train']['epochs'] = a.epochs
    if a.batch_size: cfg['train']['batch_size'] = a.batch_size
    if a.lr: cfg['train']['lr'] = a.lr
    if a.drop_edge is not None: cfg['train']['drop_edge_p'] = a.drop_edge

    set_seed(a.seed)
    dev = cfg['train']['device']
    if dev == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)

    graphs_file = a.graphs or graphs_path(cfg, preset)
    if not os.path.isabs(graphs_file):
        graphs_file = os.path.join(ROOT, graphs_file)
    gobj = load_graphs(cfg, preset, a.graphs)
    graphs = gobj['graphs']

    sample_weights = None
    large_manifest = None
    if a.large_manifest:
        manifest_path = os.path.join(
            ROOT, 'data', 'clean', 'manifests', f'{a.large_manifest}.json')
        with open(manifest_path, encoding='utf-8') as f:
            large_manifest = json.load(f)
        pool = {g.rna_id: g for g in graphs.get('pool', [])}
        expected_pool = int(large_manifest['counts']['pool'])
        manifest_pool_ids = {
            record['rna_id']
            for record in large_manifest['splits']['pool']
        }
        if len(manifest_pool_ids) != expected_pool:
            raise ValueError(
                f'large manifest pool={expected_pool} '
                f'but lists {len(manifest_pool_ids)} records')
        missing_pool = sorted(manifest_pool_ids - set(pool))
        if missing_pool:
            raise ValueError(
                f'large manifest has {len(missing_pool)} records missing from graphs: '
                f'{missing_pool[:10]}')
        if a.all_pool:
            train_ids = sorted(manifest_pool_ids)
            val_ids = []
            if a.sample_weight_mode == 'group':
                sample_weights = {
                    rna_id: float(weight)
                    for cluster in large_manifest['clusters']
                    for rna_id in cluster['rna_ids']
                    for weight in [1.0 / int(cluster['cluster_size'])]
                }
            elif a.sample_weight_mode == 'uniform':
                sample_weights = {rna_id: 1.0 for rna_id in pool}
            else:
                sample_weights = None
        elif a.fold is not None:
            fold = next(
                (item for item in large_manifest['folds']
                 if int(item['fold']) == a.fold),
                None,
            )
            if fold is None:
                raise ValueError(f'unknown fold {a.fold}')
            train_ids = list(fold['training_ids'])
            val_ids = list(fold['validation_ids'])
            if a.sample_weight_mode == 'group':
                sample_weights = {
                    rna_id: float(weight)
                    for cluster in large_manifest['clusters']
                    for rna_id in cluster['rna_ids']
                    for weight in [1.0 / int(cluster['cluster_size'])]
                }
            elif a.sample_weight_mode == 'uniform':
                sample_weights = {rna_id: 1.0 for rna_id in pool}
            else:
                sample_weights = None
        else:
            raise ValueError('--large_manifest requires --fold or --all_pool')
        missing = [rna_id for rna_id in train_ids + val_ids if rna_id not in pool]
        if missing:
            raise KeyError(f'missing large-protocol graphs: {missing[:10]}')
        graphs = dict(graphs)
        graphs['train'] = [pool[rna_id] for rna_id in train_ids]
        graphs['val'] = [pool[rna_id] for rna_id in val_ids]
    if (a.fixed_epochs is None) != (a.fixed_threshold is None):
        raise ValueError('--fixed_epochs and --fixed_threshold must be used together')



    if a.val_from:
        assert a.val_from in TEST_SPLITS, f'--val_from must be one of {TEST_SPLITS}'
        val_g = graphs[a.val_from]
        eval_splits = [s for s in TEST_SPLITS if s != a.val_from]
    else:
        val_g = graphs['val']
        eval_splits = list(TEST_SPLITS)



    val_leaked_removed = []
    if a.drop_val_leaked:
        leaked_path = os.path.join(ROOT, 'data', 'leaked_pairs.json')
        remove_val = set()
        if os.path.exists(leaked_path):
            remove_val = set(json.load(open(leaked_path, encoding='utf-8'))
                             .get('remove_val_ids', []))
        n0 = len(val_g)
        val_leaked_removed = sorted(g.rna_id for g in val_g if g.rna_id in remove_val)
        val_g = [g for g in val_g if g.rna_id not in remove_val]
        if not a.quiet:
            print(f'[val cleanup] {n0} -> {len(val_g)}  removed: {val_leaked_removed}', flush=True)

    if a.val_subsample and len(val_g) > a.val_subsample:

        gen = torch.Generator().manual_seed(a.seed)
        idx = torch.randperm(len(val_g), generator=gen)[:a.val_subsample].tolist()
        val_g = [val_g[i] for i in sorted(idx)]
    train_g = graphs['train']

    num_relations = gobj['num_relations']
    assert all(g.num_relations == num_relations for g in train_g),\
        'the number of relations in the graphs does not match the preset'

    bs = cfg['train']['batch_size']

    tr_gen = torch.Generator().manual_seed(a.seed)
    if sample_weights is None:
        train_loader = DataLoader(
            train_g, batch_size=bs, shuffle=True, generator=tr_gen)
    else:
        sampler = WeightedRandomSampler(
            [sample_weights[graph.rna_id] for graph in train_g],
            num_samples=len(train_g),
            replacement=True,
            generator=tr_gen,
        )
        train_loader = DataLoader(train_g, batch_size=bs, sampler=sampler)
    val_loader = DataLoader(val_g, batch_size=bs, shuffle=False)

    mcfg = dict(cfg['model'])

    mcfg['in_dim'] = gobj.get('in_dim') or int(train_g[0].x.shape[1])
    mcfg['num_relations'] = num_relations
    model = build_model(mcfg, num_relations).to(device)


    n_pos = sum(int((g.y == 1).sum()) for g in train_g)
    n_neg = sum(int((g.y == 0).sum()) for g in train_g)
    pw, _, _ = L.compute_pos_weight([0] * n_neg + [1] * n_pos, cfg['loss']['pos_weight_floor'])
    lname = cfg['loss']['name']


    feat_dim = getattr(model, 'feat_dim', cfg['model']['hidden'])
    criterion = L.build_loss(lname, counts=(n_neg, n_pos), feat_dim=feat_dim,
                             focal_alpha=cfg['loss'].get('focal_alpha')).to(device)
    if lname == 'bce':
        criterion.set_pos_weight(pw)

    params = list(model.parameters())
    if hasattr(criterion, 'centers'):
        params += [criterion.centers]
    opt = torch.optim.AdamW(params, lr=cfg['train']['lr'],
                            weight_decay=cfg['train']['weight_decay'])

    sch_name = cfg['train']['scheduler']
    if sch_name == 'cosine_warm_restarts':
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=cfg['train']['T_0'], T_mult=cfg['train']['T_mult'],
            eta_min=cfg['train']['eta_min'])
    elif sch_name == 'plateau':
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5,
                                                        patience=5)
    else:
        sch = None

    if not a.quiet:
        print(f'[{a.tag}/seed{a.seed}] preset={preset} arch={mcfg["arch"]} '
              f'layers={cfg["model"]["num_layers"]} hidden={cfg["model"]["hidden"]} '
              f'loss={lname} rel={num_relations} device={dev}')
        print(f'  train graphs={len(train_g)} val graphs={len(val_g)}  '
              f'nodes={n_pos+n_neg} (pos={n_pos}, neg={n_neg}) pos_weight={pw.item():.2f} '
              f'model params={sum(p.numel() for p in model.parameters())}')

    best = {'mcc': -2.0, 'epoch': -1, 'state': None, 'threshold': 0.5}
    hist = []
    patience = 0

    g = torch.Generator(device=device)
    g.manual_seed(a.seed)

    training_epochs = a.fixed_epochs or cfg['train']['epochs']
    for epoch in range(1, training_epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in train_loader:
            batch = to_device(batch, device)
            ew = batch.edge_weight if hasattr(batch, 'edge_weight') else None
            if ew is None:
                ei, et = drop_edge(batch.edge_index, batch.edge_type,
                                   cfg['train']['drop_edge_p'], generator=g)
            else:
                ei, et, ew = drop_edge(batch.edge_index, batch.edge_type,
                                       cfg['train']['drop_edge_p'], generator=g,
                                       edge_weight=ew)
            out = model(batch.x, ei, et, getattr(batch, 'batch', None),
                        return_feat=hasattr(criterion, 'centers'), edge_weight=ew)
            y = batch.y.float()
            if isinstance(out, tuple):
                logit, feat = out
                loss, _ = criterion(logit, y, feat)
            else:
                logit = out
                loss = criterion(logit, y)
            opt.zero_grad()
            loss.backward()
            if cfg['train']['grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
            opt.step()
            tot += float(loss)
            nb += 1

        if a.fixed_epochs is None:
            yv, pv = collect_preds(model, val_loader, device)
            vm = compute_metrics(yv, pv, 0.5)
            vm_auc, vm_mcc, vm_ap = vm['AUC'], vm['MCC'], vm['AUPRC']
        else:
            vm_auc = vm_mcc = vm_ap = float('nan')

        if sch is not None:
            if sch_name == 'plateau' and a.fixed_epochs is None:
                sch.step(vm_mcc)
            else:
                sch.step()

        hist.append({'epoch': epoch, 'loss': tot / max(nb, 1), 'val_AUC': vm_auc,
                     'val_MCC': vm_mcc, 'val_AUPRC': vm_ap,
                     'lr': opt.param_groups[0]['lr']})

        if a.fixed_epochs is None and vm_mcc > best['mcc']:
            best.update(mcc=vm_mcc, epoch=epoch, threshold=0.5,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if hasattr(criterion, 'centers'):
                best['centers'] = criterion.centers.detach().cpu().clone()
            patience = 0
        else:
            patience += 1

        if not a.quiet and a.fixed_epochs is None and (epoch % 10 == 0 or epoch == 1):
            print(f'  ep{epoch:3d} loss={tot/max(nb,1):.4f} val_AUC={vm_auc:.4f} '
                  f'val_MCC={vm_mcc:.4f} val_AUPRC={vm_ap:.4f} '
                  f'lr={opt.param_groups[0]["lr"]:.2e} patience={patience}', flush=True)

        if (a.fixed_epochs is None
                and patience >= cfg['train']['early_stop_patience']):
            if not a.quiet:
                print(f'  early stopping at epoch {epoch} (best epoch={best["epoch"]}, val_MCC={best["mcc"]:.4f})')
            break


    if a.fixed_epochs is None:
        model.load_state_dict(best['state'])
        if hasattr(criterion, 'centers') and 'centers' in best:
            criterion.centers.data = best['centers'].to(device)
        yv, pv = collect_preds(model, val_loader, device)
    if a.fixed_threshold is not None:
        th = float(a.fixed_threshold)
        th_score = (compute_metrics(yv, pv, th)['MCC']
                    if a.fixed_epochs is None else None)
    elif cfg['eval']['threshold_search']:
        th, th_score = find_best_threshold(yv, pv, cfg['eval']['threshold_metric'])
    else:
        th, th_score = 0.5, compute_metrics(yv, pv, 0.5)['MCC']
    if a.fixed_epochs is None:
        val_metrics = compute_metrics(yv, pv, th)
        val_metrics_05 = compute_metrics(yv, pv, 0.5)
    else:
        val_metrics = None
        val_metrics_05 = None


    test_metrics = {}
    per_chain = {'val': {}}
    for g in val_g:
        with torch.no_grad():
            logit = model(g.x.to(device), g.edge_index.to(device), g.edge_type.to(device),
                          torch.zeros(g.x.size(0), dtype=torch.long, device=device),
                          edge_weight=(g.edge_weight.to(device)
                                       if getattr(g, 'edge_weight', None) is not None else None))
            probability = torch.sigmoid(logit).float().cpu().numpy()
        per_chain['val'][g.rna_id] = {
            'prob': probability.tolist(),
            'label': g.y.cpu().numpy().astype(int).tolist(),
        }
    if a.eval_test:
        for split in eval_splits:
            if split not in graphs or not graphs[split]:
                continue
            yt, pt = collect_preds(model, DataLoader(graphs[split], batch_size=bs,
                                                     shuffle=False), device)
            test_metrics[split] = {'at_threshold': compute_metrics(yt, pt, th),
                                   'at_0.5': compute_metrics(yt, pt, 0.5),
                                   'n_residues': int(len(yt))}
            per_chain[split] = {}
            for g in graphs[split]:
                with torch.no_grad():
                    logit = model(
                        g.x.to(device),
                        g.edge_index.to(device),
                        g.edge_type.to(device),
                        torch.zeros(g.x.size(0), dtype=torch.long, device=device),
                        edge_weight=(g.edge_weight.to(device)
                                     if getattr(g, 'edge_weight', None) is not None else None),
                    )
                    probability = torch.sigmoid(logit).float().cpu().numpy()
                per_chain[split][g.rna_id] = {
                    'prob': probability.tolist(),
                    'label': g.y.cpu().numpy().astype(int).tolist(),
                }

    ckpt = {'state_dict': best['state'], 'threshold': th, 'tag': a.tag, 'seed': a.seed,
            'preset': preset, 'config': cfg, 'num_relations': num_relations,
            'graphs_file': graphs_file, 'rel_names': gobj.get('rel_names'),
            'val_leaked_removed': val_leaked_removed,
            'val_metrics': val_metrics, 'val_metrics_at_0.5': val_metrics_05,
            'test_metrics': test_metrics,
            'per_chain': per_chain,
            'best_epoch': best['epoch'] if a.fixed_epochs is None else a.fixed_epochs,
            'best_val_mcc_at_0.5': best['mcc'] if a.fixed_epochs is None else None,
            'large_manifest': a.large_manifest, 'fold': a.fold,
            'all_pool': a.all_pool, 'sample_weight': (
                'record weight = 1 / homology-cluster size'
                if sample_weights is not None else None)}
    if a.fixed_epochs is not None:
        ckpt['state_dict'] = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    ck = os.path.join(ROOT, cfg['paths']['checkpoints'], f'{a.tag}_seed{a.seed}.pt')
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    torch.save(ckpt, ck)

    res = {'tag': a.tag, 'seed': a.seed, 'preset': preset, 'config': cfg,
           'num_relations': num_relations, 'n_train_graphs': len(train_g),
           'n_val_graphs': len(val_g), 'n_pos': n_pos, 'n_neg': n_neg,
           'best_epoch': best['epoch'] if a.fixed_epochs is None else a.fixed_epochs,
           'selected_epoch': (best['epoch'] if a.fixed_epochs is None
                              else a.fixed_epochs),
           'threshold': th,

           'graphs_file': graphs_file, 'rel_names': gobj.get('rel_names'),
           'bp_threshold': gobj.get('bp_threshold'), 'sim_k': gobj.get('sim_k'),
           'weight_mode': gobj.get('weight_mode'), 'rand_sub': gobj.get('rand_sub'),
           'wnorm': cfg['model'].get('wnorm'),
           'rel_edge_counts': gobj.get('rel_edge_counts'),
           'val_leaked_removed': val_leaked_removed,
           'val_metrics': val_metrics, 'val_metrics_at_0.5': val_metrics_05,
           'test_metrics': test_metrics, 'history': hist}
    res['large_manifest'] = a.large_manifest
    res['fold'] = a.fold
    res['all_pool'] = a.all_pool
    res['sample_weight'] = ckpt['sample_weight']
    res['fixed_epoch'] = a.fixed_epochs
    res['per_chain'] = per_chain
    rp = os.path.join(ROOT, cfg['paths']['results'], f'train_{a.tag}_seed{a.seed}.json')
    save_json(res, rp)

    if val_metrics is None:
        print(f'[{a.tag}/seed{a.seed}] fixed_ep={a.fixed_epochs} th={th:.2f}  '
              f'({time.strftime("%H:%M:%S")})')
    else:
        print(f'[{a.tag}/seed{a.seed}] best_ep={best["epoch"]} '
                  f'val_AUC={val_metrics["AUC"]:.4f} val_MCC={val_metrics["MCC"]:.4f} '
                  f'val_AUPRC={val_metrics["AUPRC"]:.4f} th={th:.2f}  '
                  f'({time.strftime("%H:%M:%S")})')
    if not a.quiet and test_metrics:
        parts = [f'{sp}:AUC={m["at_threshold"]["AUC"]:.4f}/MCC={m["at_threshold"]["MCC"]:.4f}'
                 for sp, m in test_metrics.items()]
        print('  test sets ' + '  '.join(parts))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
