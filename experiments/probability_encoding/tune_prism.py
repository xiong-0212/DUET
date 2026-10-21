import os, sys, json, time
import torch, numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
PRISM = os.path.join(_here, '..')
X = os.path.join(PRISM, '..', 'x')
sys.path.insert(0, os.path.join(PRISM, 'src'))
sys.path.insert(0, os.path.join(X, 'src'))

from utils import compute_metrics, save_json, set_seed
from rgcn_model import RGCNBindingSiteModel
from losses import build_loss
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

T = ['TE18', 'TL12', 'JL10', 'RB9']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'




print('Loading de-leaked graphs...', flush=True)
bp_graph = torch.load(os.path.join(PRISM, 'data', 'processed', 'graphs_seq_ss_bp_clean.pt'),
                      map_location='cpu', weights_only=False)
NUM_REL = bp_graph['num_relations']
IN_DIM = bp_graph['in_dim']





data = __import__('data_loader').load_all()
test_seqs = set()
for sp in T:
    for s in data[sp]:
        test_seqs.add(s['sequence'])

val_clean, val_leaked = [], []
for g in bp_graph['graphs']['val']:

    rid = g.rna_id
    seq = None
    for s in data['val']:
        if s['rna_id'] == rid:
            seq = s['sequence']
            break
    if seq and seq in test_seqs:
        val_leaked.append(rid)
    else:
        val_clean.append(g)

print(f'Val: {len(bp_graph["graphs"]["val"])} → {len(val_clean)} (removed {len(val_leaked)} leaked chains: {val_leaked})')


leaked_test = set()
train_seqs = {}
for s in data['train']:
    train_seqs.setdefault(s['sequence'], []).append(s['rna_id'])
for sp in T:
    for s in data[sp]:
        if s['sequence'] in train_seqs:
            leaked_test.add(s['rna_id'])
print(f'Leaked test chains: {len(leaked_test)}/49')




def make_loaders(val_graphs):
    train_loader = DataLoader(bp_graph['graphs']['train'], batch_size=8, shuffle=False)
    val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)
    test_loaders = {sp: DataLoader(bp_graph['graphs'][sp], batch_size=8, shuffle=False)
                    for sp in T}
    return train_loader, val_loader, test_loaders




def train_and_eval(dropout, lr, wd, seed=42, epochs=120, patience=20,
                   train_loader=None, val_loader=None, test_loaders=None):
    set_seed(seed)
    model = RGCNBindingSiteModel(in_dim=IN_DIM, hidden=128, num_relations=NUM_REL,
                                  num_layers=2, dropout=dropout, norm='graph',
                                  residual=True).to(DEVICE)

    n_pos = sum(g.y.sum().item() for g in bp_graph['graphs']['train'])
    n_neg = sum(g.y.shape[0] - g.y.sum().item() for g in bp_graph['graphs']['train'])
    pw = max(n_neg / max(n_pos, 1), 3.0)

    from losses import build_loss
    tcl = build_loss('tcl_focal', counts=(int(n_neg), int(n_pos)), feat_dim=64).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_mcc, best_state, best_ep = -2, None, -1
    bad = 0
    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            z = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
            feat = None
            loss = tcl.focal(z, batch.y.view(-1).float())
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        vy, vp = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                z = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                vy.append(batch.y.cpu().numpy())
                vp.append(torch.sigmoid(z).squeeze(-1).cpu().numpy())
        vy = np.concatenate(vy); vp = np.concatenate(vp)
        mcc = compute_metrics(vy, vp, 0.5)['MCC']
        if mcc > best_mcc:
            best_mcc, best_ep = mcc, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()

    results = {}
    for sp, loader in test_loaders.items():
        ys, ps = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                z = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                ys.append(batch.y.cpu().numpy())
                ps.append(torch.sigmoid(z).squeeze(-1).cpu().numpy())
        ys = np.concatenate(ys); ps = np.concatenate(ps)
        results[sp] = compute_metrics(ys, ps, 0.5)


    nl_ys, nl_ps = [], []
    offset = 0
    for sp, loader in test_loaders.items():
        for batch in loader:
            batch = batch.to(DEVICE)
            for i in range(batch.num_graphs):
                mask = batch.batch == i
                rid = None
            break


    avg_mcc = np.mean([results[sp]['MCC'] for sp in T])
    return {'best_val_mcc': best_mcc, 'best_epoch': best_ep,
            'test': results, 'avg_mcc': avg_mcc,
            'config': {'dropout': dropout, 'lr': lr, 'wd': wd}}


def eval_nonleaked(model, test_loaders, leaked_test):
    
    model.eval()
    nl_ys, nl_ps = [], []
    l_ys, l_ps = [], []
    with torch.no_grad():
        for sp, loader in test_loaders.items():
            for batch in loader:
                batch = batch.to(DEVICE)
                z = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                p = torch.sigmoid(z).squeeze(-1).cpu().numpy()
                y = batch.y.cpu().numpy()

                sizes = torch.bincount(batch.batch, minlength=batch.num_graphs)
                off = 0
                for gi in range(batch.num_graphs):
                    L = sizes[gi].item()

                    rid = None


                    off += L

                nl_ys.append(y); nl_ps.append(p)
    return None





GRID = []
for dropout in [0.2, 0.4, 0.6]:
    for lr in [3e-4, 5e-4, 1e-3]:
        GRID.append({'dropout': dropout, 'lr': lr, 'wd': 1e-4})

print(f'Grid search: {len(GRID)} configs', flush=True)
print(f'Device: {DEVICE}', flush=True)

train_loader, val_loader, test_loaders = make_loaders(val_clean)

all_results = []
t0 = time.time()
for ci, cfg in enumerate(GRID):
    print(f'\n[{ci+1}/{len(GRID)}] dropout={cfg["dropout"]} lr={cfg["lr"]} wd={cfg["wd"]}', flush=True)
    r = train_and_eval(cfg['dropout'], cfg['lr'], cfg['wd'],
                       train_loader=train_loader, val_loader=val_loader,
                       test_loaders=test_loaders)
    r['config'] = cfg
    all_results.append(r)
    vals = [f'{r["test"][sp]["MCC"]:.4f}' for sp in T]
    print(f'  val_MCC={r["best_val_mcc"]:.4f}  test_avg={r["avg_mcc"]:.4f}  [{"/".join(vals)}]', flush=True)

print(f'\nGrid search finished in {time.time()-t0:.0f}s')




best = max(all_results, key=lambda r: r['best_val_mcc'])
print(f'\nBest config (by de-leaked val MCC): dropout={best["config"]["dropout"]} lr={best["config"]["lr"]}')
print(f'  val_MCC={best["best_val_mcc"]:.4f}  test_avg_MCC={best["avg_mcc"]:.4f}')


print('\n' + '='*80)
print(f'{"dropout":>8s} {"lr":>8s} {"wd":>8s} {"val_MCC":>9s} {"test_avg":>9s} {"TE18":>8s} {"RB9":>8s}')
print('-'*80)
for r in sorted(all_results, key=lambda x: -x['best_val_mcc']):
    c = r['config']
    print(f'{c["dropout"]:>8.1f} {c["lr"]:>8.0e} {c["wd"]:>8.0e} '
          f'{r["best_val_mcc"]:>9.4f} {r["avg_mcc"]:>9.4f} '
          f'{r["test"]["TE18"]["MCC"]:>8.4f} {r["test"]["RB9"]["MCC"]:>8.4f}')


save_json({'grid_results': all_results, 'best': best},
          os.path.join(PRISM, 'results', 'tuning_cleaned.json'))
print(f'\nSaved PRISM/results/tuning_cleaned.json')
