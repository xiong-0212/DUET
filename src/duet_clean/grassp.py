from __future__ import annotations

import copy
import os
from pathlib import Path
import random
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from .io import project_path
from .metrics import compute_metrics, find_best_threshold


SHARED_TESTS = ["TE18", "TL12", "JL10"]
PROCESSED_PATHS = {
    "train": "data/grassp/processed/HARI60NEW_TRAIN/processed/processed_data_hariboss.pt",
    "val": "data/grassp/processed/HARI60NEW_VAL/processed/processed_data_hariboss.pt",
    "TE18": "data/grassp/processed/CORBRA/TE18/processed/processed_data_hariboss.pt",
    "TL12": "data/grassp/processed/CORBRA/TL12/processed/processed_data_hariboss.pt",
    "JL10": "data/grassp/processed/CORBRA/JL10/processed/processed_data_hariboss.pt",
}


def to_grassp_candidates(rna_id: str) -> list[str]:
    pdb, _, chain = rna_id.partition("_")
    candidates = [f"{pdb}{chain}".upper()]
    if chain.upper() == "A":
        candidates.append(pdb.upper())
    return candidates


def load_processed_chains(path: str | Path) -> dict[str, Data]:
    data, slices = torch.load(path, map_location="cpu", weights_only=False)
    names = list(data.pdb_name)
    xs = slices["x"]
    edges = slices["edge_index"]
    out = {}
    for index, name in enumerate(names):
        n0, n1 = int(xs[index]), int(xs[index + 1])
        e0, e1 = int(edges[index]), int(edges[index + 1])
        out[str(name)] = Data(
            x=data.x[n0:n1],
            edge_index=data.edge_index[:, e0:e1],
            y=data.y[n0:n1],
            ss_emb=data.ss_emb[n0:n1],
        )
    return out


def select_chains(
    records,
    chains: dict[str, Data],
) -> list[tuple[str, Data]]:
    selected = []
    for record in records:
        graph = None
        for candidate in to_grassp_candidates(record.rna_id):
            if candidate in chains:
                graph = chains[candidate]
                break
        if graph is None:
            raise KeyError(f"GRASSP data missing for {record.rna_id}")
        if graph.x.size(0) != len(record.sequence):
            raise ValueError(f"{record.rna_id}: GRASSP length mismatch")
        expected = torch.tensor(
            [int(value) for value in record.labels],
            dtype=graph.y.dtype,
        )
        if not torch.equal(graph.y.view(-1).cpu(), expected):
            raise ValueError(f"{record.rna_id}: GRASSP labels do not match manifest")
        selected.append((record.rna_id, graph))
    return selected


def load_grassp_splits(
    manifest_splits: dict[str, list],
    feature_source: str = "ernie",
) -> dict[str, list[tuple[str, Data]]]:
    splits = {
        split: select_chains(
            manifest_splits[split],
            load_processed_chains(path),
        )
        for split, path in PROCESSED_PATHS.items()
    }
    if feature_source == "ernie":
        return splits
    if feature_source != "rnafm":
        raise ValueError(f"unknown feature source: {feature_source}")

    source = torch.load(
        project_path("data/processed/graphs_graph_homology_p80.pt"),
        map_location="cpu",
        weights_only=False,
    )
    rnafm = {
        graph.rna_id: graph.x
        for split in ["TE18", "TL12", "JL10", "val", "train"]
        for graph in source["graphs"][split]
    }
    for split, chains in splits.items():
        replaced = []
        for rna_id, graph in chains:
            if rna_id not in rnafm:
                raise KeyError(f"missing RNA-FM features for {rna_id}")
            replaced.append(
                (
                    rna_id,
                    Data(
                        x=rnafm[rna_id],
                        edge_index=graph.edge_index,
                        y=graph.y,
                        ss_emb=graph.ss_emb,
                    ),
                )
            )
        splits[split] = replaced
    return splits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict(
    model: nn.Module,
    chains: list[tuple[str, Data]],
    device: torch.device,
    batch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    model.eval()
    probabilities = []
    labels = []
    for start in range(0, len(chains), batch_size):
        chunk = chains[start : start + batch_size]
        if batch_size == 1:
            _, graph = chunk[0]
            graph = graph.to(device)
            batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
        else:
            graph = Batch.from_data_list([item[1] for item in chunk]).to(device)
            batch = graph.batch
        logits = model(graph.x, graph.ss_emb, graph.edge_index, batch)
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(graph.y.cpu().numpy().astype(int))
    return probabilities, labels


def _pooled(probabilities: list[np.ndarray], labels: list[np.ndarray], threshold: float):
    return compute_metrics(np.concatenate(labels), np.concatenate(probabilities), threshold)


def _per_chain(
    probabilities: list[np.ndarray],
    labels: list[np.ndarray],
    chains: list[tuple[str, Data]],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    flat_probability = np.concatenate(probabilities) if probabilities else np.array([])
    flat_label = np.concatenate(labels) if labels else np.array([], dtype=int)
    if len(flat_probability) != len(flat_label):
        raise ValueError("prediction and label lengths differ")
    out = {}
    offset = 0
    for rna_id, graph in chains:
        length = int(graph.y.numel())
        end = offset + length
        probability = flat_probability[offset:end]
        label = flat_label[offset:end]
        if len(probability) != length or len(label) != length:
            raise ValueError(f"{rna_id}: per-chain prediction length mismatch")
        out[rna_id] = {
            "prob": probability.tolist(),
            "label": label.tolist(),
            "metrics": compute_metrics(label, probability, threshold),
        }
        offset = end
    if offset != len(flat_probability):
        raise ValueError("per-chain predictions did not cover the full batch pool")
    return out


def train_one(
    splits: dict[str, list[tuple[str, Data]]],
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    hidden: int,
    dropout: float,
    batch_size: int,
    device: torch.device,
    rna_dim: int = 768,
    sample_weights: dict[str, float] | None = None,
    fixed_epoch: int | None = None,
    threshold_override: float | None = None,
    evaluation_splits: list[str] | None = None,
    patience: int | None = 12,
) -> tuple[nn.Module, dict[str, Any]]:
    from grassp_model.RNABP import HybridRNABindingSiteModel

    set_seed(seed)
    train_chains = splits["train"]
    val_chains = splits["val"]
    model = HybridRNABindingSiteModel(
        rna_dim=rna_dim,
        ss_dim=4,
        hidden=hidden,
        dropout=dropout,
    ).to(device)
    n_pos = int(sum(int(graph.y.sum()) for _, graph in train_chains))
    n_neg = int(sum(graph.y.numel() - int(graph.y.sum()) for _, graph in train_chains))
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1), 3)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6,
    )

    best = {"mcc": -1.0, "epoch": -1, "state": None}
    stale = 0
    history = []
    generator = random.Random(seed)
    evaluation_splits = evaluation_splits or SHARED_TESTS
    training_epochs = int(fixed_epoch) if fixed_epoch is not None else epochs
    for epoch in range(1, training_epochs + 1):
        model.train()
        if sample_weights is None:
            order = list(range(len(train_chains)))
            generator.shuffle(order)
        else:
            weights = torch.tensor(
                [sample_weights[rna_id] for rna_id, _ in train_chains],
                dtype=torch.double,
            )
            order = torch.multinomial(
                weights,
                num_samples=len(train_chains),
                replacement=True,
                generator=torch.Generator().manual_seed(seed + epoch),
            ).tolist()
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(order), batch_size):
            chunk = [train_chains[index] for index in order[start : start + batch_size]]
            graph = Batch.from_data_list([item[1] for item in chunk]).to(device)
            logits = model(graph.x, graph.ss_emb, graph.edge_index, graph.batch)
            loss = criterion(logits, graph.y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            n_batches += 1
        scheduler.step()
        if fixed_epoch is None:
            val_p, val_y = predict(model, val_chains, device, batch_size)
            val_mcc = _pooled(val_p, val_y, 0.5)["MCC"]
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
                    "state": copy.deepcopy(model.state_dict()),
                }
                stale = 0
            else:
                stale += 1
            if patience is not None and stale >= patience:
                history[-1]["early_stopped"] = True
                break
        else:
            history.append(
                {
                    "epoch": epoch,
                    "loss": total_loss / max(n_batches, 1),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

    if fixed_epoch is None:
        model.load_state_dict(best["state"])
        val_p, val_y = predict(model, val_chains, device, batch_size)
        val_labels = np.concatenate(val_y)
        val_prob = np.concatenate(val_p)
        if threshold_override is None:
            threshold, threshold_score = find_best_threshold(val_labels, val_prob)
        else:
            threshold = float(threshold_override)
            threshold_score = _pooled(val_p, val_y, threshold)["MCC"]
        selected_epoch = int(best["epoch"])
    else:
        if threshold_override is None:
            raise ValueError("fixed_epoch requires threshold_override")
        threshold = float(threshold_override)
        threshold_score = None
        selected_epoch = int(fixed_epoch)
        if val_chains:
            val_p, val_y = predict(
                model,
                val_chains,
                device,
                batch_size,
            )
        else:
            val_p, val_y = [], []

    metrics = {}
    metrics_05 = {}
    per_chain = {
        "val": (
            _per_chain(val_p, val_y, val_chains, threshold)
            if val_chains
            else {}
        )
    }
    for split in evaluation_splits:
        chain_data = splits[split]
        p, y = predict(model, chain_data, device, batch_size)
        metrics[split] = _pooled(p, y, threshold)
        metrics_05[split] = _pooled(p, y, 0.5)
        per_chain[split] = _per_chain(p, y, chain_data, threshold)

    result = {
        "seed": seed,
        "best_validation_epoch_at_0.5": best["epoch"],
        "best_validation_mcc_at_0.5": best["mcc"] if best["epoch"] > 0 else None,
        "selected_epoch": selected_epoch,
        "threshold": threshold,
        "validation_threshold_score": threshold_score,
        "n_train": len(train_chains),
        "n_val": len(val_chains),
        "sample_weight": (
            "record weight = 1 / homology-cluster size"
            if sample_weights is not None
            else None
        ),
        "fixed_epoch": fixed_epoch,
        "test": metrics,
        "test_at_0.5": metrics_05,
        "per_chain": per_chain,
        "history": history,
    }
    return model, result
