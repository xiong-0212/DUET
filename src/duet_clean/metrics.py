from __future__ import annotations

import os
import random
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def set_seed(seed: int) -> None:
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


def gmean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return float(np.sqrt(sensitivity * specificity))


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_two_classes = len(set(y_true.tolist())) > 1
    return {
        "AUC": float(roc_auc_score(y_true, y_prob)) if has_two_classes else float("nan"),
        "AUPRC": float(average_precision_score(y_true, y_prob))
        if has_two_classes
        else float("nan"),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "GMean": gmean(y_true, y_pred),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "threshold": float(threshold),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    lo: float = 0.05,
    hi: float = 0.95,
    n: int = 91,
) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    best_t, best_score = 0.5, -np.inf
    for threshold in np.linspace(lo, hi, n):
        pred = (y_prob >= threshold).astype(int)
        score = matthews_corrcoef(y_true, pred)
        if score > best_score:
            best_t, best_score = float(threshold), float(score)
    return best_t, best_score


def safe_torch_load(path: str, map_location: str | torch.device = "cpu") -> Any:
    return torch.load(os.path.abspath(path), map_location=map_location, weights_only=False)
