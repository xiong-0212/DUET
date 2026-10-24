from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch

from .metrics import compute_metrics, find_best_threshold


def weighted_order(
    rna_ids: list[str],
    sample_weights: dict[str, float],
    seed_or_generator: int | torch.Generator,
) -> list[int]:
    generator = (
        torch.Generator().manual_seed(seed_or_generator)
        if isinstance(seed_or_generator, int)
        else seed_or_generator
    )
    weights = torch.tensor(
        [float(sample_weights[rna_id]) for rna_id in rna_ids],
        dtype=torch.double,
    )
    if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise ValueError("sample weights must be finite and positive")
    return torch.multinomial(
        weights,
        num_samples=len(rna_ids),
        replacement=True,
        generator=generator,
    ).tolist()


def pooled_per_chain_metrics(
    per_chain: dict[str, dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    labels = np.concatenate(
        [np.asarray(value["label"], dtype=int) for value in per_chain.values()]
    )
    probabilities = np.concatenate(
        [np.asarray(value["prob"], dtype=float) for value in per_chain.values()]
    )
    return compute_metrics(labels, probabilities, threshold)


def select_cv_hyperparameters(
    folds: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    folds = list(folds)
    if not folds:
        raise ValueError("at least one CV fold is required")
    oof = {}
    for fold in folds:
        for rna_id, payload in fold["per_chain"]["val"].items():
            if rna_id in oof:
                raise ValueError(f"duplicate OOF chain {rna_id}")
            oof[rna_id] = payload
    threshold, score = find_best_threshold(
        np.concatenate([np.asarray(value["label"], dtype=int) for value in oof.values()]),
        np.concatenate([np.asarray(value["prob"], dtype=float) for value in oof.values()]),
    )
    epochs = [int(fold["selected_epoch"]) for fold in folds]
    selected_epoch = int(np.median(epochs))
    return {
        "selected_epoch": selected_epoch,
        "fold_best_epochs": epochs,
        "epoch_rule": "median of fold-best epochs selected at threshold 0.5",
        "threshold": float(threshold),
        "oof_threshold_score": float(score),
        "oof_metrics_at_selected_threshold": pooled_per_chain_metrics(oof, threshold),
        "oof_metrics_at_0.5": pooled_per_chain_metrics(oof, 0.5),
        "n_oof_chains": len(oof),
    }
