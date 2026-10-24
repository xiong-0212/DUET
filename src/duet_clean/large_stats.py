from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .comparison import holm_adjust, paired_test
from .io import project_path, write_json
from .metrics import compute_metrics


FOUR_SPLITS = ["TE18", "TL12", "JL10", "RB9"]
SHARED_SPLITS = ["TE18", "TL12", "JL10"]
INDEPENDENT_EXTERNAL = ["external_jiang", "time_dependent"]
EXTERNAL_SPLITS = [*INDEPENDENT_EXTERNAL, "hariboss_test"]


def load_final(method: str, seed: int) -> dict[str, Any]:
    path = project_path(
        f"results/large_protocol/final/{method}/seed{seed}.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def load_external(method: str, seed: int) -> dict[str, Any]:
    path = project_path(
        f"results/large_protocol/external/{method}_seed{seed}.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def average_mcc(result: dict[str, Any], splits: list[str]) -> float:
    return float(np.mean([result["test"][split]["MCC"] for split in splits]))


def summarize(
    runs: list[dict[str, Any]],
    splits: list[str],
) -> dict[str, Any]:
    per_split = {}
    for split in splits:
        values = np.asarray(
            [run["test"][split]["MCC"] for run in runs],
            dtype=float,
        )
        per_split[split] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "values": values.tolist(),
        }
    averages = np.asarray([average_mcc(run, splits) for run in runs])
    return {
        "n_seeds": len(runs),
        "splits": splits,
        "average": {
            "mean": float(averages.mean()),
            "sd": float(averages.std(ddof=1)),
            "values": averages.tolist(),
        },
        "per_split": per_split,
    }


def compare_runs(
    first_runs: list[dict[str, Any]],
    second_runs: list[dict[str, Any]],
    splits: list[str],
    metric_key: str = "test",
) -> dict[str, Any]:
    first = np.asarray(
        [
            np.mean([run[metric_key][split]["MCC"] for split in splits])
            for run in first_runs
        ]
    )
    second = np.asarray(
        [
            np.mean([run[metric_key][split]["MCC"] for split in splits])
            for run in second_runs
        ]
    )
    result = paired_test(first, second)
    result["splits"] = splits
    result["first_mean"] = float(first.mean())
    result["second_mean"] = float(second.mean())
    return result


def apply_holm(comparisons: dict[str, dict[str, Any]]) -> None:
    keys = list(comparisons)
    adjusted = holm_adjust([float(comparisons[key]["p"]) for key in keys])
    for key, p_value in zip(keys, adjusted):
        comparisons[key]["p_holm"] = float(p_value)


def _fast_mcc(labels: np.ndarray, predictions: np.ndarray) -> float:
    tp = float(np.logical_and(predictions == 1, labels == 1).sum())
    tn = float(np.logical_and(predictions == 0, labels == 0).sum())
    fp = float(np.logical_and(predictions == 1, labels == 0).sum())
    fn = float(np.logical_and(predictions == 0, labels == 1).sum())
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denominator) if denominator else 0.0


def _pack_chains(
    per_chain: dict[str, dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    ids = sorted(per_chain)
    lengths = np.asarray(
        [len(per_chain[rna_id]["label"]) for rna_id in ids],
        dtype=int,
    )
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    labels = np.concatenate(
        [np.asarray(per_chain[rna_id]["label"], dtype=int) for rna_id in ids]
    )
    probabilities = np.concatenate(
        [np.asarray(per_chain[rna_id]["prob"], dtype=float) for rna_id in ids]
    )
    return {
        "ids": ids,
        "offsets": offsets,
        "labels": labels,
        "predictions": (probabilities >= threshold).astype(np.int8),
    }


def chain_bootstrap_compare(
    first_runs: list[dict[str, Any]],
    second_runs: list[dict[str, Any]],
    splits: list[str],
    n_resamples: int = 5000,
    seed: int = 20261024,
    fixed_threshold: bool = False,
) -> dict[str, Any]:
    if len(first_runs) != len(second_runs):
        raise ValueError("paired methods must have the same number of seeds")
    pairs = []
    observed = []
    for first, second in zip(first_runs, second_runs):
        if int(first["seed"]) != int(second["seed"]):
            raise ValueError("seed ordering differs")
        split_pairs = {}
        split_deltas = []
        for split in splits:
            left_threshold = 0.5 if fixed_threshold else float(first["threshold"])
            right_threshold = 0.5 if fixed_threshold else float(second["threshold"])
            left = _pack_chains(first["per_chain"][split], left_threshold)
            right = _pack_chains(second["per_chain"][split], right_threshold)
            if left["ids"] != right["ids"]:
                raise ValueError(f"{split}: chain IDs differ")
            split_pairs[split] = (left, right)
            split_deltas.append(
                _fast_mcc(left["labels"], left["predictions"])
                - _fast_mcc(right["labels"], right["predictions"])
            )
        pairs.append(split_pairs)
        observed.append(float(np.mean(split_deltas)))

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_resamples):
        deltas = []
        for split in splits:
            reference = pairs[0][split][0]
            offsets = reference["offsets"]
            sampled = rng.integers(0, len(offsets) - 1, size=len(offsets) - 1)
            flat = np.concatenate(
                [np.arange(offsets[index], offsets[index + 1]) for index in sampled]
            )
            for split_pairs in pairs:
                left, right = split_pairs[split]
                deltas.append(
                    _fast_mcc(left["labels"][flat], left["predictions"][flat])
                    - _fast_mcc(right["labels"][flat], right["predictions"][flat])
                )
        boot.append(float(np.mean(deltas)))
    values = np.asarray(boot)
    low = int((values <= 0).sum())
    high = int((values >= 0).sum())
    return {
        "observed_delta": float(np.mean(observed)),
        "ci95": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
        "p_two_sided": float(
            2 * min((low + 1) / (n_resamples + 1), (high + 1) / (n_resamples + 1))
        ),
        "n_resamples": n_resamples,
        "bootstrap_seed": seed,
        "n_seeds": len(pairs),
        "splits": splits,
        "fixed_threshold": fixed_threshold,
    }


def merge_ensemble(
    member_results: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    split_names = sorted(member_results[0]["per_chain"])
    merged_per_chain = {}
    for split in split_names:
        ids = sorted(member_results[0]["per_chain"][split])
        merged = {}
        for rna_id in ids:
            labels = np.asarray(
                member_results[0]["per_chain"][split][rna_id]["label"],
                dtype=int,
            )
            probabilities = []
            for result in member_results:
                payload = result["per_chain"][split][rna_id]
                if labels.tolist() != payload["label"]:
                    raise ValueError(f"{split}/{rna_id}: label mismatch")
                probabilities.append(np.asarray(payload["prob"], dtype=float))
            merged[rna_id] = {
                "label": labels.tolist(),
                "prob": np.mean(probabilities, axis=0).tolist(),
            }
        merged_per_chain[split] = merged
    metrics = {}
    metrics_05 = {}
    for split in split_names:
        labels = np.concatenate(
            [
                np.asarray(value["label"], dtype=int)
                for value in merged_per_chain[split].values()
            ]
        )
        probabilities = np.concatenate(
            [
                np.asarray(value["prob"], dtype=float)
                for value in merged_per_chain[split].values()
            ]
        )
        metrics[split] = compute_metrics(labels, probabilities, threshold)
        metrics_05[split] = compute_metrics(labels, probabilities, 0.5)
    return {
        "seed": int(member_results[0]["seed"]),
        "threshold": float(threshold),
        "test": metrics,
        "test_at_0.5": metrics_05,
        "per_chain": merged_per_chain,
    }


def write_json_result(path: str | Path, value: Any) -> None:
    write_json(path, value)
