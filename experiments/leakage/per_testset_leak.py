import os, sys, torch, numpy as np, json

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TF'] = 'NONE'
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
set_seed(42)


train_seqs = {}
for s in data['train']:
    train_seqs.setdefault(s['sequence'], []).append(s['rna_id'])
leaked_ids = set()
for sp in T:
    for s in data[sp]:
        if s['sequence'] in train_seqs: leaked_ids.add(s['rna_id'])




from rgcn_model import MLPOnly
from losses import build_loss
import torch.nn as nn, torch.nn.functional as F

bp_graph = torch.load(os.path.join(PRISM, 'data', 'processed', 'graphs_seq_ss_bp.pt'),
                      map_location='cpu', weights_only=False)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_c = MLPOnly(in_dim=640, hidden=128, dropout=0.4).to(DEVICE)
train_s = [{'x': g.x.float(), 'y': g.y.float()} for g in bp_graph['graphs']['train']]
n_pos = sum(s['y'].sum().item() for s in train_s)
n_neg = sum(len(s['y']) - s['y'].sum().item() for s in train_s)
tcl = build_loss('tcl_focal', counts=(int(n_neg), int(n_pos)), feat_dim=64).to(DEVICE)
opt = torch.optim.AdamW(model_c.parameters(), lr=5e-4, weight_decay=1e-4)
for ep in range(1, 101):
    model_c.train()
    perm = torch.randperm(len(train_s)).tolist()
    for i in range(0, len(perm), 8):
        b = [train_s[j] for j in perm[i:i+8]]
        x = torch.cat([s['x'] for s in b]).to(DEVICE)
        y = torch.cat([s['y'] for s in b]).to(DEVICE)
        z, feat = model_c(x, return_feat=True)
        loss, _ = tcl(z, y, feat)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model_c.parameters(), 5.0); opt.step()
model_c.eval()
print('CoBRA-equiv training done', flush=True)

cobra_by_chain = {}
for sp in T:
    for g in bp_graph['graphs'][sp]:
        with torch.no_grad():
            p = torch.sigmoid(model_c(g.x.float().to(DEVICE)).squeeze(-1)).cpu().numpy()
        cobra_by_chain[g.rna_id] = (g.y.numpy(), p)




from rgcn_model import RGCNBindingSiteModel
model_p = RGCNBindingSiteModel(in_dim=640, hidden=128, num_relations=3,
                                num_layers=2, dropout=0.4, norm='graph', residual=True)
ck = torch.load(os.path.join(PRISM, 'checkpoints', 'bp_prob_seed42.pt'),
                map_location='cpu', weights_only=False)
model_p.load_state_dict(ck.get('model_state_dict', ck.get('state_dict', ck)))
model_p.eval()
prism_by_chain = {}
for sp in T:
    for g in bp_graph['graphs'][sp]:
        with torch.no_grad():
            z = model_p(g.x.float(), g.edge_index, g.edge_type, None)
            p = torch.sigmoid(z).squeeze(-1).cpu().numpy()
        prism_by_chain[g.rna_id] = (g.y.numpy(), p)
print('PRISM per-chain done', flush=True)




from model.RNABP import HybridRNABindingSiteModel
ckpt_g = torch.load(os.path.join(GRASSP, 'checkpoints', 'TR60_SEED39.pt'),
                    map_location='cpu', weights_only=False)
model_g = HybridRNABindingSiteModel(rna_dim=768, ss_dim=4, hidden=86, dropout=0.6)
model_g.load_state_dict(ckpt_g['model_state_dict'])
model_g.eval()
grassp_th = ckpt_g.get('threshold', 0.7)
from feature_extraction.datasetnew import RNAGraphDatasetNew
grassp_by_chain = {}
for sp, root in [('TE18', os.path.join(GRASSP, 'data', 'processed', 'TE18NEW')),
                  ('TL12', os.path.join(GRASSP, 'data', 'processed', 'CORBRA', 'TL12')),
                  ('JL10', os.path.join(GRASSP, 'data', 'processed', 'CORBRA', 'JL10'))]:
    if not os.path.exists(os.path.join(root, 'processed', 'processed_data_hariboss.pt')): continue
    try:
        dataset = RNAGraphDatasetNew(root=root)
        for ci in range(len(dataset)):
            d = dataset[ci]
            pn = d.pdb_name
            name = str(pn[0]) if isinstance(pn, (list, tuple)) else str(pn)
            cobra_id = name[:4].lower() + '_' + (name[4:] if len(name) > 4 else 'A')
            L = d.x.shape[0]
            with torch.no_grad():
                logits = model_g(d.x, d.ss_emb, d.edge_index, torch.zeros(L, dtype=torch.long))
                if logits.dim() == 2: logits = logits.squeeze(0)
                p = torch.sigmoid(logits).cpu().numpy()
            grassp_by_chain[cobra_id] = (d.y.view(-1).cpu().numpy(), p)
    except Exception as e:
        print(f'  GRASSP {sp} error: {e}')
print(f'GRASSP: {len(grassp_by_chain)} chains', flush=True)




models = [('CoBRA-equiv', cobra_by_chain), ('PRISM', prism_by_chain), ('GRASSP', grassp_by_chain)]

print('\n' + '='*110)
print('Per-test-set leaked / non-leaked MCC (seed42)')
print('='*110)
print(f"{'test set':<8s} {'model':<14s} {'leak MCC':>9s} {'clean MCC':>10s} {'all MCC':>9s} {'leak n':>6s} {'clean n':>7s} {'total n':>5s}")
print('-'*110)

for sp in T:
    leaked_in = [s['rna_id'] for s in data[sp] if s['rna_id'] in leaked_ids]
    clean_in = [s['rna_id'] for s in data[sp] if s['rna_id'] not in leaked_ids]
    for model_name, by_chain in models:
        ly, lp, cy, cp, ay, ap = [], [], [], [], [], []
        for s in data[sp]:
            rid = s['rna_id']
            if rid in by_chain:
                y, p = by_chain[rid][:2]
                ay.append(y); ap.append(p)
                if rid in leaked_ids: ly.append(y); lp.append(p)
                else: cy.append(y); cp.append(p)
        ay = np.concatenate(ay) if ay else np.array([])
        ap = np.concatenate(ap) if ap else np.array([])
        m_all = matthews_corrcoef(ay, (ap > 0.5).astype(int)) if len(set(ay)) > 1 else 0
        m_l = matthews_corrcoef(np.concatenate(ly), (np.concatenate(lp) > 0.5).astype(int)) if ly and len(set(np.concatenate(ly))) > 1 else float('nan')
        m_c = matthews_corrcoef(np.concatenate(cy), (np.concatenate(cp) > 0.5).astype(int)) if cy and len(set(np.concatenate(cy))) > 1 else float('nan')
        nl_s = f'{m_l:.4f}' if not np.isnan(m_l) else '—'
        nc_s = f'{m_c:.4f}' if not np.isnan(m_c) else '—'
        print(f'{sp:<8s} {model_name:<14s} {nl_s:>9s} {nc_s:>10s} {m_all:>9.4f} {len(ly):>6d} {len(cy):>7d} {len(ay):>5d}')
    print()


print('\n' + '='*110)
print('Mean (4 test sets)')
print('='*110)
out = {}
for model_name, by_chain in models:
    all_m, leak_ms, clean_ms = [], [], []
    per_set = {}
    for sp in T:
        ly, lp, cy, cp, ay, ap = [], [], [], [], [], []
        for s in data[sp]:
            rid = s['rna_id']
            if rid in by_chain:
                y, p = by_chain[rid][:2]
                ay.append(y); ap.append(p)
                if rid in leaked_ids: ly.append(y); lp.append(p)
                else: cy.append(y); cp.append(p)
        if not ay:
            continue
        ay = np.concatenate(ay); ap = np.concatenate(ap)
        m_all = float(matthews_corrcoef(ay, (ap > 0.5).astype(int)))
        all_m.append(m_all)
        rec = {'all': m_all, 'leaked': None, 'non_leaked': None,
               'n_leaked': len(ly), 'n_clean': len(cy)}
        if ly:
            m = float(matthews_corrcoef(np.concatenate(ly), (np.concatenate(lp) > 0.5).astype(int)))
            if not np.isnan(m):
                leak_ms.append(m); rec['leaked'] = m
        if cy:
            m = float(matthews_corrcoef(np.concatenate(cy), (np.concatenate(cp) > 0.5).astype(int)))
            if not np.isnan(m):
                clean_ms.append(m); rec['non_leaked'] = m
        per_set[sp] = rec
    lm = f'{np.mean(leak_ms):.4f}' if leak_ms else '—'
    cm = f'{np.mean(clean_ms):.4f}' if clean_ms else '—'
    print(f'  {model_name:<14s} leaked={lm}  non-leaked={cm}  all={np.mean(all_m):.4f}')
    out[model_name] = {
        'per_testset': per_set,
        'avg_of_4sets': {
            'all': float(np.mean(all_m)),
            'leaked': float(np.mean(leak_ms)) if leak_ms else None,
            'non_leaked': float(np.mean(clean_ms)) if clean_ms else None,
        },
    }

out['_meta'] = {
    'threshold': 0.5,
    'aggregation': 'per_testset_then_average (protocol B)',
    'source': 'PRISM = checkpoints/bp_prob_seed42.pt (seed42, trained on the original benchmark); '
              'CoBRA-equiv = trained in this script (100 epochs, no early stopping); GRASSP = official weights',
    'leak_rule': '100% sequence identity (compared against the 345 training chains)',
}
save_path = os.path.join(PRISM, 'results', 'leak_per_testset.json')
json.dump(out, open(save_path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'\nsaved {save_path}')
