from __future__ import annotations

import json
import os
import random

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)


    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def gmean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    se = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    return float(np.sqrt(se * sp))


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        'AUC': float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) > 1 else float('nan'),
        'AUPRC': float(average_precision_score(y_true, y_prob)) if len(set(y_true.tolist())) > 1 else float('nan'),
        'MCC': float(matthews_corrcoef(y_true, y_pred)),
        'F1': float(f1_score(y_true, y_pred, zero_division=0)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'GMean': gmean(y_true, y_pred),
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
        'threshold': float(threshold),
    }
    return m


def find_best_threshold(y_true, y_prob, metric: str = 'mcc',
                        lo: float = 0.01, hi: float = 0.99, n: int = 99):
    
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    best_t, best_s = 0.5, -np.inf
    for t in np.linspace(lo, hi, n):
        pred = (y_prob >= t).astype(int)
        if metric == 'mcc':
            s = matthews_corrcoef(y_true, pred)
        elif metric == 'f1':
            s = f1_score(y_true, pred, zero_division=0)
        else:
            s = accuracy_score(y_true, pred)
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t, float(best_s)


def drop_edge(edge_index: torch.Tensor, edge_type: torch.Tensor, p: float,
              generator=None, edge_weight: torch.Tensor = None):
    
    if p <= 0:
        return (edge_index, edge_type) if edge_weight is None else \
            (edge_index, edge_type, edge_weight)
    e = edge_index.size(1)
    keep = torch.rand(e, device=edge_index.device, generator=generator) >= p
    if keep.sum() == 0:
        return (edge_index, edge_type) if edge_weight is None else \
            (edge_index, edge_type, edge_weight)
    out = (edge_index[:, keep], edge_type[keep])
    if edge_weight is not None:
        out = out + (edge_weight[keep],)
    return out


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=float)


def mean_std(values):
    v = [x for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return float('nan'), float('nan')
    a = np.asarray(v, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) if len(a) > 1 else 0.0)
