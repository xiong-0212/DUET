import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TF'] = 'NONE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys, torch, numpy as np, json

_here = os.path.dirname(os.path.abspath(__file__))
PRISM = os.path.join(_here, '..')
X = os.path.join(PRISM, '..', 'x')
GRASSP = os.path.join(PRISM, '..', 'grassp', 'GRASSP-main')
for p in [os.path.join(PRISM, 'src'), os.path.join(X, 'src'), GRASSP]:
    sys.path.insert(0, p)

from sklearn.metrics import matthews_corrcoef, roc_auc_score
from utils import compute_metrics




import data_loader as dl
data = dl.load_all()
T = ['TE18', 'TL12', 'JL10', 'RB9']

train_seqs = {}
for s in data['train']:
    train_seqs.setdefault(s['sequence'], []).append(s['rna_id'])

leaked_ids = set()
for sp in T:
    for s in data[sp]:
        if s['sequence'] in train_seqs:
            leaked_ids.add(s['rna_id'])

print(f'leaked test chains: {len(leaked_ids)}/49')
for sp in T:
    leaked_in_sp = [s['rna_id'] for s in data[sp] if s['rna_id'] in leaked_ids]
    print(f'  {sp}: {len(leaked_in_sp)} leaked / {len(data[sp])} total')




print('\n' + '='*70)
print('PRISM (bp_prob, seed42)')
print('='*70)


from rgcn_model import RGCNBindingSiteModel
from losses import build_loss
bp_graph = torch.load(os.path.join(PRISM, 'data', 'processed', 'graphs_seq_ss_bp.pt'),
                      map_location='cpu', weights_only=False)
num_rel = bp_graph['num_relations']
in_dim = bp_graph['in_dim']
model_p = RGCNBindingSiteModel(in_dim=in_dim, hidden=128, num_relations=num_rel,
                                num_layers=2, dropout=0.4, norm='graph', residual=True)
ckpt = torch.load(os.path.join(PRISM, 'checkpoints', 'bp_prob_seed42.pt'),
                  map_location='cpu', weights_only=False)
model_p.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt['state_dict'])
model_p.eval()

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

prism_by_chain = {}
for sp in T:
    loader = DataLoader(bp_graph['graphs'][sp], batch_size=1, shuffle=False)
    for batch in loader:
        rid = batch.rna_id[0] if hasattr(batch, 'rna_id') and batch.rna_id else f'unknown'
        with torch.no_grad():
            z = model_p(batch.x, batch.edge_index, batch.edge_type, batch.batch)
            p = torch.sigmoid(z).squeeze(-1).cpu().numpy()
        y = batch.y.cpu().numpy()
        prism_by_chain[rid] = (y, p)


prism_leaked_y, prism_leaked_p = [], []
prism_clean_y, prism_clean_p = [], []
for sp in T:
    for s in data[sp]:
        rid = s['rna_id']
        if rid in prism_by_chain:
            y, p = prism_by_chain[rid]
            if rid in leaked_ids:
                prism_leaked_y.append(y); prism_leaked_p.append(p)
            else:
                prism_clean_y.append(y); prism_clean_p.append(p)

for label, ys, ps in [('leaked', prism_leaked_y, prism_leaked_p),
                       ('non-leaked', prism_clean_y, prism_clean_p)]:
    y = np.concatenate(ys); p = np.concatenate(ps)
    m = compute_metrics(y, p, 0.5)
    print(f'  {label}: MCC={m["MCC"]:.4f}  AUC={m["AUC"]:.4f}  (n={len(y)} residues, {len(y)//40} chains)')




print('\n' + '='*70)
print('GRASSP (TR60_SEED39.pt)')
print('='*70)
from model.RNABP import HybridRNABindingSiteModel
ckpt_g = torch.load(os.path.join(GRASSP, 'checkpoints', 'TR60_SEED39.pt'), map_location='cpu', weights_only=False)
model_g = HybridRNABindingSiteModel(rna_dim=768, ss_dim=4, hidden=86, dropout=0.6)
model_g.load_state_dict(ckpt_g['model_state_dict'])
model_g.eval()
grassp_th = ckpt_g.get('threshold', 0.7)
from feature_extraction.datasetnew import RNAGraphDatasetNew
grassp_by_chain = {}
for sp, root in [('TE18', os.path.join(GRASSP, 'data', 'processed', 'TE18NEW')), ('TL12', os.path.join(GRASSP, 'data', 'processed', 'CORBRA', 'TL12')), ('JL10', os.path.join(GRASSP, 'data', 'processed', 'CORBRA', 'JL10'))]:
    if not os.path.exists(os.path.join(root, 'processed')): continue
    try: dataset = RNAGraphDatasetNew(root=root)
    except Exception as e: print(f'  skip {sp}: {e}'); continue
    for ci, d in enumerate(dataset):
        name = str(d.pdb_name[0]) if hasattr(d, 'pdb_name') and d.pdb_name else f'{sp}_{ci}'
        cobra_id = name[:4].lower() + '_' + (name[4:] if len(name) > 4 else 'A')
        L = d.x.shape[0]
        with torch.no_grad():
            logits = model_g(d.x, d.ss_emb, d.edge_index, torch.zeros(L, dtype=torch.long))
            if logits.dim() == 2: logits = logits.squeeze(0)
            p = torch.sigmoid(logits).cpu().numpy()
        grassp_by_chain[cobra_id] = (d.y.view(-1).cpu().numpy(), p, name)
print(f'GRASSP per-chain: {len(grassp_by_chain)}')
gr_ly, gr_lp, gr_cy, gr_cp = [], [], [], []
n_found = 0
for sp in T:
    for s in data[sp]:
        rid = s['rna_id']
        if rid in grassp_by_chain:
            y, p, gn = grassp_by_chain[rid]
            if rid in leaked_ids: gr_ly.append(y); gr_lp.append(p); n_found += 1
            else: gr_cy.append(y); gr_cp.append(p)
print(f'  leaked found: {n_found}')
for label, ys, ps in [('leaked', gr_ly, gr_lp), ('non-leaked', gr_cy, gr_cp)]:
    if not ys: print(f'  {label}: no data'); continue
    y = np.concatenate(ys); p = np.concatenate(ps)
    mcc = matthews_corrcoef(y, (p > grassp_th).astype(int))
    auc = roc_auc_score(y, p)
    print(f'  {label}: MCC={mcc:.4f}  AUC={auc:.4f}  n={len(y)}')


print('\n' + '='*70)
print('Leakage sensitivity summary (training set unchanged; test chains split into leaked / non-leaked)')
print('='*70)
print(f'{"":<20s} {"leaked MCC":>12s} {"non-leaked MCC":>14s} {"delta":>8s}')
print('-'*60)
for model_name, ly, lp, cy, cp in [
    ('PRISM bp_prob', prism_leaked_y, prism_leaked_p, prism_clean_y, prism_clean_p),
    ('GRASSP', gr_leaked_y, gr_leaked_p, gr_clean_y, gr_clean_p)]:
    if not ly or not cy:
        print(f'{model_name:<20s} {"—":>12s} {"—":>14s}')
        continue
    yl = np.concatenate(ly); pl = np.concatenate(lp)
    yc = np.concatenate(cy); pc = np.concatenate(cp)
    mcc_l = matthews_corrcoef(yl, (pl > 0.5).astype(int))
    mcc_c = matthews_corrcoef(yc, (pc > 0.5).astype(int))
    print(f'{model_name:<20s} {mcc_l:>12.4f} {mcc_c:>14.4f} {mcc_l-mcc_c:>8.4f}')

print(f'\nLeaked chains: {len(leaked_ids)} | non-leaked chains: {49-len(leaked_ids)}')
print('→ a delta > 0 means the model does better on sequences it has seen (memorization effect)')
print('→ a delta ≈ 0 means the model generalizes well and its performance does not rely on sequence leakage')
