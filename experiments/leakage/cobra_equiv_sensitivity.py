import os, sys, torch, numpy as np, time, json

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from utils import compute_metrics, save_json, set_seed
from rgcn_model import MLPOnly
from losses import build_loss
import torch.nn as nn
import torch.nn.functional as F

T = ['TE18', 'TL12', 'JL10', 'RB9']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data(graph_file, remove_leaked=False):
    
    obj = torch.load(graph_file, map_location='cpu', weights_only=False)
    graphs = obj['graphs']

    remove_set = set()
    if remove_leaked:
        leaked = json.load(open(os.path.join(ROOT, 'data', 'leaked_pairs.json'),
                                encoding='utf-8'))
        remove_set = set(leaked['remove_train_ids'])

    result = {}
    for split, gs in graphs.items():
        samples = []
        for g in gs:
            if split == 'train' and g.rna_id in remove_set:
                continue
            samples.append({
                'rna_id': g.rna_id,
                'x': g.x.float(),
                'y': g.y.float(),
            })
        result[split] = samples
    return result, obj.get('num_relations', 2)


def train_eval(data, tag, seed=42, epochs=120, patience=20):
    set_seed(seed)
    model = MLPOnly(in_dim=640, hidden=128, dropout=0.4).to(DEVICE)

    train = data['train']

    n_pos = sum(s['y'].sum().item() for s in train)
    n_neg = sum(len(s['y']) - s['y'].sum().item() for s in train)
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1), 3.0)], device=DEVICE)

    crit = lambda z, y: F.binary_cross_entropy_with_logits(z, y, pos_weight=pos_weight)




    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)


    from losses import build_loss
    tcl_focal = build_loss('tcl_focal', counts=(int(n_neg), int(n_pos)), feat_dim=64).to(DEVICE)

    best = {'mcc': -2, 'state': None, 'epoch': -1}
    bad = 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(train)).tolist()
        tot = 0.0
        for i in range(0, len(perm), 8):
            batch = [train[j] for j in perm[i:i+8]]
            x = torch.cat([s['x'] for s in batch]).to(DEVICE)
            y = torch.cat([s['y'] for s in batch]).to(DEVICE)
            z, feat = model(x, return_feat=True)
            loss, _tcl = tcl_focal(z, y, feat)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss)


        model.eval()
        vy = torch.cat([s['y'] for s in data['val']]).numpy()
        with torch.no_grad():
            vp = np.concatenate([torch.sigmoid(model(s['x'].to(DEVICE)).squeeze(-1)).cpu().numpy()
                           for s in data['val']])
        mcc = compute_metrics(vy, vp, 0.5)['MCC']
        if mcc > best['mcc']:
            best = {'mcc': mcc, 'epoch': ep,
                    'state': {k: v.cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
        if ep % 20 == 0 or ep == 1:
            print(f'  ep{ep:3d} loss={tot:.4f} val_MCC={mcc:.4f}', flush=True)
        if bad >= patience:
            break

    model.load_state_dict(best['state'])
    model.eval()
    print(f'  best: ep{best["epoch"]} val_MCC={best["mcc"]:.4f}')


    results = {}
    with torch.no_grad():
        for sp in T:
            if sp not in data: continue
            ys, ps = [], []
            for s in data[sp]:
                ys.append(s['y'].numpy())
                ps.append(torch.sigmoid(model(s['x'].to(DEVICE)).squeeze(-1)).cpu().numpy())
            ys = np.concatenate(ys); ps = np.concatenate(ps)
            results[sp] = compute_metrics(ys, ps, 0.5)
            print(f'  {sp:5s} MCC={results[sp]["MCC"]:.4f} AUC={results[sp]["AUC"]:.4f}')

    avg = np.mean([results[sp]['MCC'] for sp in T if sp in results])
    print(f'  mean MCC = {avg:.4f}')
    return results, avg


if __name__ == '__main__':
    set_seed(42)
    print('Loading RNA-FM embeddings...', flush=True)
    data_orig, _ = load_data(os.path.join(ROOT, 'data', 'processed', 'graphs_seq_ss_bp.pt'),
                             remove_leaked=False)
    data_clean, _ = load_data(os.path.join(ROOT, 'data', 'processed', 'graphs_seq_ss_bp.pt'),
                              remove_leaked=True)
    print(f'original train: {len(data_orig["train"])}  cleaned: {len(data_clean["train"])}')

    all_results = {}
    for label, d in [('cobra_equiv_orig', data_orig), ('cobra_equiv_clean', data_clean)]:
        print(f'\n===== {label} ({len(d["train"])} chains) =====', flush=True)
        results, avg = train_eval(d, label, seed=42)
        all_results[label] = {'results': results, 'avg_mcc': avg}


    print('\n' + '='*70)
    print('CoBRA-equivalent model (RNA-FM + MLP + TCL-Focal) sensitivity experiment')
    print('='*70)
    for label in ['cobra_equiv_orig', 'cobra_equiv_clean']:
        r = all_results[label]
        vals = [r['results'][sp]['MCC'] for sp in T if sp in r['results']]
        print(f'  {label:<24s} ' + ''.join(f'{r["results"][sp]["MCC"]:>8.4f}' for sp in T if sp in r['results'])
              + f'  mean={np.mean(vals):.4f}')

    a = all_results['cobra_equiv_orig']['avg_mcc']
    b = all_results['cobra_equiv_clean']['avg_mcc']
    print(f'\n  Δ (leakage contribution) = {a - b:+.4f}')

    save_json(all_results, os.path.join(ROOT, 'results', 'cobra_equiv_sensitivity.json'))
    print(f'saved results/cobra_equiv_sensitivity.json')
