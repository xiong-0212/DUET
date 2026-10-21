import os, sys, torch, numpy as np, json

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TF'] = 'NONE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_here = os.path.dirname(os.path.abspath(__file__))
PRISM = os.path.join(_here, '..')
X = os.path.join(PRISM, '..', 'x')
GRASSP = os.path.join(PRISM, '..', 'grassp', 'GRASSP-main')
for p in [os.path.join(PRISM, 'src'), os.path.join(X, 'src'), GRASSP]:
    if p not in sys.path: sys.path.insert(0, p)

from sklearn.metrics import matthews_corrcoef, roc_auc_score
from utils import compute_metrics, set_seed
import data_loader as dl
import ss as SS

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
print(f'leaked chains: {len(leaked_ids)}/49')




print('\n--- CoBRA-equiv: train first, then go chain by chain ---')
from rgcn_model import MLPOnly
from losses import build_loss
import torch.nn as nn, torch.nn.functional as F

bp_graph = torch.load(os.path.join(PRISM, 'data', 'processed', 'graphs_seq_ss_bp.pt'),
                      map_location='cpu', weights_only=False)


set_seed(42)
model_c = MLPOnly(in_dim=640, hidden=128, dropout=0.4).to('cuda' if torch.cuda.is_available() else 'cpu')
DEV = next(model_c.parameters()).device
train_samples = []
for g in bp_graph['graphs']['train']:
    train_samples.append({'x': g.x.float(), 'y': g.y.float()})
n_pos = sum(s['y'].sum().item() for s in train_samples)
n_neg = sum(len(s['y']) - s['y'].sum().item() for s in train_samples)
tcl = build_loss('tcl_focal', counts=(int(n_neg), int(n_pos)), feat_dim=64).to(DEV)
opt = torch.optim.AdamW(model_c.parameters(), lr=5e-4, weight_decay=1e-4)

for ep in range(1, 101):
    model_c.train()
    perm = torch.randperm(len(train_samples)).tolist()
    for i in range(0, len(perm), 8):
        batch = [train_samples[j] for j in perm[i:i+8]]
        x = torch.cat([s['x'] for s in batch]).to(DEV)
        y = torch.cat([s['y'] for s in batch]).to(DEV)
        z, feat = model_c(x, return_feat=True)
        loss, _ = tcl(z, y, feat)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model_c.parameters(), 5.0)
        opt.step()
    if ep % 30 == 0: print(f'  ep{ep} loss={tot:.3f}' if 'tot' in dir() else f'  ep{ep}')
model_c.eval()
print(f'  CoBRA-equiv training done')


cobra_by_chain = {}
for sp in T:
    for g in bp_graph['graphs'][sp]:
        rid = g.rna_id
        with torch.no_grad():
            z = model_c(g.x.float().to(DEV))
            p = torch.sigmoid(z).squeeze(-1).cpu().numpy()
        y = g.y.cpu().numpy()
        cobra_by_chain[rid] = (y, p)
print(f'  CoBRA: {len(cobra_by_chain)} chains')




print('\n--- PRISM per-chain inference ---')
from rgcn_model import RGCNBindingSiteModel
model_p = RGCNBindingSiteModel(in_dim=640, hidden=128, num_relations=3,
                                num_layers=2, dropout=0.4, norm='graph', residual=True)
ckpt_p = torch.load(os.path.join(PRISM, 'checkpoints', 'bp_prob_seed42.pt'),
                    map_location='cpu', weights_only=False)
model_p.load_state_dict(ckpt_p.get('model_state_dict', ckpt_p.get('state_dict', ckpt_p)))
model_p.eval()

prism_by_chain = {}
for sp in T:
    for g in bp_graph['graphs'][sp]:
        rid = g.rna_id
        with torch.no_grad():
            z = model_p(g.x.float(), g.edge_index, g.edge_type, None)
            p = torch.sigmoid(z).squeeze(-1).cpu().numpy()
        y = g.y.cpu().numpy()
        prism_by_chain[rid] = (y, p)
print(f'  PRISM: {len(prism_by_chain)} chains')




print('\n--- GRASSP per-chain inference ---')
from model.RNABP import HybridRNABindingSiteModel
ckpt_g = torch.load(os.path.join(GRASSP, 'checkpoints', 'TR60_SEED39.pt'),
                    map_location='cpu', weights_only=False)
model_g = HybridRNABindingSiteModel(rna_dim=768, ss_dim=4, hidden=86, dropout=0.6)
model_g.load_state_dict(ckpt_g['model_state_dict'])
model_g.eval()
grassp_th = ckpt_g.get('threshold', 0.7)

from feature_extraction.datasetnew import RNAGraphDatasetNew
grassp_by_chain = {}
for sp, roots in [('TE18', [os.path.join(GRASSP, 'data', 'processed', 'TE18NEW')]),
                  ('TL12', [os.path.join(GRASSP, 'data', 'processed', 'CORBRA', 'TL12')]),
                  ('JL10', [os.path.join(GRASSP, 'data', 'processed', 'CORBRA', 'JL10')])]:
    for root in roots:
        if not os.path.exists(os.path.join(root, 'processed', 'processed_data_hariboss.pt')):
            print(f'  [skip] {sp}: processed file not found')
            continue
        try:
            dataset = RNAGraphDatasetNew(root=root)
            n = len(dataset)
            print(f'  {sp}: {n} chains')
            for ci in range(n):
                d = dataset[ci]

                if hasattr(d, 'pdb_name') and d.pdb_name:
                    pn = d.pdb_name
                    name = str(pn[0]) if isinstance(pn, (list, tuple)) else str(pn)
                else:
                    name = f'{sp}_chain{ci}'

                pdb4 = name[:4].lower()
                ch = name[4:] if len(name) > 4 else 'A'
                cobra_id = f'{pdb4}_{ch}'
                L = d.x.shape[0]
                with torch.no_grad():
                    logits = model_g(d.x, d.ss_emb, d.edge_index,
                                     torch.zeros(L, dtype=torch.long))
                    if logits.dim() == 2:
                        logits = logits.squeeze(0)
                    p = torch.sigmoid(logits).cpu().numpy()
                y = d.y.view(-1).cpu().numpy()
                grassp_by_chain[cobra_id] = (y, p, name)
        except Exception as e:
            print(f'  [error] {sp}: {type(e).__name__}: {e}')

print(f'  GRASSP: {len(grassp_by_chain)} chains')

if len(grassp_by_chain) < 20:
    for rid in sorted(grassp_by_chain.keys()):
        print(f'    {rid}')




print('\n' + '='*80)
print('Leaked / non-leaked stratified results')
print('='*80)

results = {}
for model_name, by_chain in [('CoBRA-equiv', cobra_by_chain),
                              ('PRISM bp_prob', prism_by_chain),
                              ('GRASSP', grassp_by_chain)]:
    ly, lp, cy, cp = [], [], [], []
    n_l = n_c = 0
    for sp in T:
        for s in data[sp]:
            rid = s['rna_id']
            if rid in by_chain:
                y, p = by_chain[rid][:2]
                if rid in leaked_ids:
                    ly.append(y); lp.append(p); n_l += 1
                else:
                    cy.append(y); cp.append(p); n_c += 1

    r = {'n_leaked': n_l, 'n_clean': n_c}
    for label, ys, ps in [('leaked', ly, lp), ('non_leaked', cy, cp)]:
        if not ys:
            r[label] = None
            continue
        y = np.concatenate(ys); p = np.concatenate(ps)
        mcc = matthews_corrcoef(y, (p > 0.5).astype(int))
        auc = roc_auc_score(y, p)
        r[label] = {'MCC': mcc, 'AUC': auc, 'n_res': len(y)}
        print(f'  {model_name:<16s} {label:<12s} MCC={mcc:.4f}  AUC={auc:.4f}  n={len(y)}')

    if r['leaked'] and r['non_leaked']:
        r['delta_mcc'] = r['leaked']['MCC'] - r['non_leaked']['MCC']
        print(f'  {model_name:<16s} {"ΔMCC":<12s} {r["delta_mcc"]:+.4f}')
    results[model_name] = r
    print()


save_json(results, os.path.join(PRISM, 'results', 'leak_sensitivity_3models.json'))
print('saved results/leak_sensitivity_3models.json')
