import os, sys, torch, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE

while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)



GRASSP_ROOT = os.environ.get('GRASSP_ROOT', os.path.join(ROOT, 'third_party', 'GRASSP-main'))
X_ROOT = os.environ.get('X_ROOT', os.path.join(ROOT, '..', 'x'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(X_ROOT, 'src'))
sys.path.insert(0, GRASSP_ROOT)

import ss as SS
from ss import load_bp_prob


X_DATA = os.path.join(X_ROOT, 'data', 'raw')
PDB_DIR = os.path.join(X_DATA, 'pdb')




def read_fasta(path):
    seqs, rid, buf = {}, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith('>'):
            if rid: seqs[rid] = ''.join(buf)
            rid = line[1:].split()[0]
            buf = []
        elif line:
            buf.append(line)
    if rid: seqs[rid] = ''.join(buf)
    return seqs

def read_labels(path):
    
    labels = {}
    for line in open(path):
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            labels[parts[0]] = [int(c) for c in parts[1]]
    return labels

fa = os.path.join(X_DATA, 'ss_input.fa')
all_seqs = read_fasta(fa)
rb9_ids = ['1nem_A','1pbr_A','1q8n_A','1uui_B','1y26_X','2gdi_X','2tob_A','3d2x_A','3gx2_A']
rb9 = {rid: all_seqs[rid] for rid in rb9_ids if rid in all_seqs}
print(f'RB9: {len(rb9)} chains, lengths {[len(s) for s in rb9.values()]}')

folds = SS.load_folds(os.path.join(X_DATA, 'ss_output.tsv'))


import glob

def get_pdb_coords(rna_id):
    
    pdb_dir = PDB_DIR
    pdb_file = os.path.join(pdb_dir, f'{rna_id.split("_")[0]}.pdb')
    if not os.path.exists(pdb_file):
        return None
    coords = []
    chain_id = rna_id.split('_')[1] if '_' in rna_id else 'A'
    for line in open(pdb_file):
        if line.startswith('ATOM') and "' " in line:
            atom_name = line[12:16].strip()
            if atom_name in ("C4'", "C1'"):
                chain = line[21].strip()
                if chain == chain_id:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    coords.append((x, y, z))
    return np.array(coords) if coords else None

from scipy.spatial.distance import cdist

def knn_edges(coords, k=8, max_dist=20.0):
    
    if coords is None or len(coords) < 2:
        return []
    dist = cdist(coords, coords)
    edges = []
    for i in range(len(coords)):
        d = dist[i]
        d[i] = np.inf
        neighbors = np.argsort(d)[:k]
        for j in neighbors:
            if d[j] <= max_dist:
                edges.append((i, int(j)))
    return edges




from torch_geometric.data import Data

ernie_emb = torch.load(os.path.join(ROOT, 'data', 'ernie_rna', 'rb9_ernie_embeddings.pt'),
                        map_location='cpu', weights_only=False)

data_list = []
for rid, seq in rb9.items():
    L = len(seq)
    db = folds.get(rid, '')


    bs_file = os.path.join(X_DATA, 'cobra', 'RB9', 'RB9-bindingsite.txt')
    if '_labels_cache' not in globals():
        globals()['_labels_cache'] = {}
        for line in open(bs_file):
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                globals()['_labels_cache'][parts[0]] = [int(c) for c in parts[1] if c in '01']
    y_list = globals()['_labels_cache'].get(rid, [0] * L)
    y = torch.tensor(y_list[:L], dtype=torch.float)


    x = torch.tensor(ernie_emb[rid], dtype=torch.float)
    L = min(L, x.shape[0])
    x = x[:L]
    y = y[:L]
    paired = np.zeros(L)
    if db and len(db) >= L:
        for i, c in enumerate(db[:L]):
            if c in '()[]':
                paired[i] = 1.0
    ss_emb = torch.tensor(np.stack([
        paired,
        np.arange(L) / max(L - 1, 1),
        np.ones(L) if db else np.zeros(L),
        paired
    ], axis=1), dtype=torch.float)


    coords = get_pdb_coords(rid)
    if coords is not None and len(coords) != L:
        coords = None
    edges = knn_edges(coords) if coords is not None else []
    if edges:
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:

        ei = torch.tensor([[i for i in range(L-1)] + [i+1 for i in range(L-1)],
                           [i+1 for i in range(L-1)] + [i for i in range(L-1)]],
                          dtype=torch.long)

    batch = torch.zeros(L, dtype=torch.long)
    data_list.append(Data(x=x, edge_index=ei, y=y, ss_emb=ss_emb, num_nodes=L))

print(f'RB9: {len(data_list)} chains, total residues {sum(d.x.shape[0] for d in data_list)}')




from model.RNABP import HybridRNABindingSiteModel
from torch_geometric.loader import DataLoader
ckpt_path = os.path.join(GRASSP_ROOT, 'checkpoints', 'TR60_SEED39.pt')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

model = HybridRNABindingSiteModel(rna_dim=768, ss_dim=4, hidden=86, dropout=0.6)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
threshold = ckpt.get('threshold', 0.7)
print(f'GRASSP model loaded (hidden=86, threshold={threshold})')

loader = DataLoader(data_list, batch_size=1, shuffle=False)
probs_all, labels_all = [], []
with torch.no_grad():
    for batch in loader:
        emb = batch.x
        ss_e = batch.ss_emb
        edge = batch.edge_index
        b = batch.batch
        logits = model(emb, ss_e, edge, b)
        if logits.dim() == 2:
            logits = logits.squeeze(0)
        probs_all.append(torch.sigmoid(logits).cpu().numpy())
        labels_all.append(batch.y.cpu().numpy())

probs = np.concatenate(probs_all)
labels = np.concatenate(labels_all).astype(int)
print(f'total residues: {len(probs)}, positives: {labels.sum()} ({labels.mean()*100:.1f}%)')




from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, precision_score, recall_score, average_precision_score


preds = (probs > threshold).astype(int)
mcc = matthews_corrcoef(labels, preds)
auc = roc_auc_score(labels, probs)
f1 = f1_score(labels, preds, zero_division=0)
prec = precision_score(labels, preds, zero_division=0)
rec = recall_score(labels, preds, zero_division=0)
aupr = average_precision_score(labels, probs)


from utils import compute_metrics
m_best = compute_metrics(labels, probs, 0.5)

print(f'\n=== GRASSP results on RB9 ===')
print(f'fixed threshold {threshold}:')
print(f'  MCC = {mcc:.4f}')
print(f'  AUC = {auc:.4f}')
print(f'  F1  = {f1:.4f}')
print(f'  Precision = {prec:.4f}')
print(f'  Recall    = {rec:.4f}')
print(f'  AUPRC     = {aupr:.4f}')
print(f'\nthreshold 0.5 (our protocol):')
print(f'  MCC = {m_best["MCC"]:.4f}')
print(f'  AUC = {m_best["AUC"]:.4f}')


print(f'\n--- reference values ---')
print(f'  our PRISM RB9 MCC  = 0.637 (5 seeds)')
print(f'  CoBRA-M2 RB9 MCC   = 0.593 (paper)')
print(f'  CoBRA-M3 RB9 MCC   = 0.557 (paper)')
print(f'  GRASSP RB9 MCC     = {mcc:.4f} (fixed threshold) / {m_best["MCC"]:.4f} (0.5)')


torch.save({'probs': probs, 'labels': labels, 'threshold': threshold,
            'metrics_fixed': {'mcc': mcc, 'auc': auc, 'f1': f1},
            'metrics_05': {'mcc': m_best['MCC'], 'auc': m_best['AUC']}},
           'results/grassp_rb9_inference.pt')
print('\n[written] results/grassp_rb9_inference.pt')
