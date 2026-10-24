from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import compute_metrics, find_best_threshold, set_seed
from .models import Record


class CoBRAHead(nn.Module):
    def __init__(self, in_dim: int = 1280, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.mlp(x))


@dataclass(frozen=True)
class EmbeddedRecord:
    rna_id: str
    split: str
    x: torch.Tensor
    y: torch.Tensor


def attach_embeddings(
    splits: dict[str, list[Record]],
    embeddings: dict[str, torch.Tensor],
) -> dict[str, list[EmbeddedRecord]]:
    out: dict[str, list[EmbeddedRecord]] = {}
    missing: list[str] = []
    for split, records in splits.items():
        out[split] = []
        for record in records:
            embedding = embeddings.get(record.rna_id)
            if embedding is None:
                missing.append(record.rna_id)
                continue
            if embedding.shape[0] != len(record.sequence):
                raise ValueError(
                    f"{record.rna_id}: embedding length {embedding.shape[0]} "
                    f"!= sequence length {len(record.sequence)}"
                )
            out[split].append(
                EmbeddedRecord(
                    rna_id=record.rna_id,
                    split=split,
                    x=embedding.float(),
                    y=torch.tensor([int(value) for value in record.labels], dtype=torch.float),
                )
            )
    if missing:
        raise KeyError(f"missing embeddings for {len(missing)} records: {sorted(missing)[:20]}")
    return out


def _batches(
    records: list[EmbeddedRecord],
    batch_size: int,
    generator: torch.Generator | None = None,
    sample_weights: dict[str, float] | None = None,
) -> Iterable[list[EmbeddedRecord]]:
    if generator is None:
        order = list(range(len(records)))
    elif sample_weights is not None:
        weights = torch.tensor(
            [sample_weights[record.rna_id] for record in records],
            dtype=torch.double,
        )
        order = torch.multinomial(
            weights,
            num_samples=len(records),
            replacement=True,
            generator=generator,
        ).tolist()
    else:
        order = torch.randperm(len(records), generator=generator).tolist()
    for start in range(0, len(order), batch_size):
        yield [records[index] for index in order[start : start + batch_size]]


@torch.no_grad()
def predict(
    model: nn.Module,
    records: list[EmbeddedRecord],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for record in records:
        logits = model(record.x.to(device))
        probabilities.append(torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy())
        labels.append(record.y.numpy().astype(int))
    return np.concatenate(labels), np.concatenate(probabilities)


@torch.no_grad()
def predict_per_chain(
    model: nn.Module,
    records: list[EmbeddedRecord],
    device: torch.device,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    model.eval()
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        logits = model(record.x.to(device))
        probability = torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy()
        labels = record.y.numpy().astype(int)
        out[record.rna_id] = {
            "prob": probability.tolist(),
            "label": labels.tolist(),
            "metrics": compute_metrics(labels, probability, threshold),
        }
    return out


def evaluate(
    model: nn.Module,
    data: dict[str, list[EmbeddedRecord]],
    device: torch.device,
    threshold: float,
    split_names: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    metrics: dict[str, Any] = {}
    per_chain: dict[str, dict[str, dict[str, Any]]] = {}
    for split in split_names:
        records = data.get(split, [])
        if not records:
            continue
        labels, probabilities = predict(model, records, device)
        metrics[split] = compute_metrics(labels, probabilities, threshold)
        per_chain[split] = predict_per_chain(model, records, device, threshold)
    return metrics, per_chain


def train_one(
    data: dict[str, list[EmbeddedRecord]],
    test_splits: list[str],
    seed: int,
    protocol: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int | None,
    sample_weights: dict[str, float] | None = None,
    fixed_epoch: int | None = None,
    threshold_override: float | None = None,
) -> tuple[CoBRAHead, dict[str, Any]]:
    set_seed(seed)
    train_records = data["train"]
    val_records = data["val"]
    model = CoBRAHead().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    if protocol == "paper":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )

    generator = torch.Generator().manual_seed(seed)
    best = {"mcc": -2.0, "epoch": -1, "state": None}
    stale = 0
    history: list[dict[str, Any]] = []

    training_epochs = int(fixed_epoch) if fixed_epoch is not None else epochs
    for epoch in range(1, training_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in _batches(
            train_records,
            batch_size,
            generator,
            sample_weights,
        ):
            x = torch.cat([record.x for record in batch]).to(device)
            y = torch.cat([record.y for record in batch]).to(device).long()
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            n_batches += 1
        scheduler.step()

        if fixed_epoch is None:
            val_y, val_p = predict(model, val_records, device)
            val_mcc = compute_metrics(val_y, val_p, 0.5)["MCC"]
            history.append(
                {
                    "epoch": epoch,
                    "loss": total_loss / max(n_batches, 1),
                    "val_mcc_at_0.5": val_mcc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            if val_mcc > best["mcc"]:
                best = {
                    "mcc": val_mcc,
                    "epoch": epoch,
                    "state": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                }
                stale = 0
            else:
                stale += 1
            if patience is not None and stale >= patience:
                break
        else:
            history.append(
                {
                    "epoch": epoch,
                    "loss": total_loss / max(n_batches, 1),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

    if fixed_epoch is not None:
        selected_epoch = int(fixed_epoch)
        final_mcc = None
    elif protocol == "paper":
        final_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        model.load_state_dict(final_state)
        selected_epoch = epochs
        final_mcc = None
    else:
        model.load_state_dict(best["state"])
        selected_epoch = int(best["epoch"])
        final_mcc = float(best["mcc"])

    if threshold_override is not None:
        threshold = float(threshold_override)
        threshold_score = None
        if val_records:
            val_y, val_p = predict(model, val_records, device)
            threshold_score = float(
                compute_metrics(val_y, val_p, threshold)["MCC"]
            )
    elif protocol == "paper":
        threshold = 0.5
        val_y, val_p = predict(model, val_records, device)
        threshold_score = compute_metrics(val_y, val_p, threshold)["MCC"]
    else:
        val_y, val_p = predict(model, val_records, device)
        threshold, threshold_score = find_best_threshold(val_y, val_p)

    test_metrics, per_chain = evaluate(
        model,
        data,
        device,
        threshold,
        test_splits,
    )
    if val_records:
        per_chain = {
            "val": predict_per_chain(
                model,
                val_records,
                device,
                threshold,
            ),
            **per_chain,
        }
    result = {
        "seed": seed,
        "protocol": protocol,
        "selected_epoch": selected_epoch,
        "best_validation_epoch_at_0.5": int(best["epoch"]),
        "best_validation_mcc_at_0.5": (
            float(best["mcc"]) if best["mcc"] > -2.0 else None
        ),
        "threshold": float(threshold),
        "validation_threshold_score": (
            float(threshold_score) if threshold_score is not None else None
        ),
        "n_train": len(train_records),
        "n_val": len(val_records),
        "sample_weight": (
            "record weight = 1 / homology-cluster size"
            if sample_weights is not None
            else None
        ),
        "fixed_epoch": fixed_epoch,
        "test": test_metrics,
        "per_chain": per_chain,
        "history": history,
    }
    return model, result
