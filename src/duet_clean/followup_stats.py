from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .comparison import holm_adjust, paired_test


METRICS = [
    "MCC",
    "AUC",
    "AUPRC",
    "F1",
    "Precision",
    "Recall",
    "GMean",
]


def _mean_metric(
    run: dict[str, Any],
    splits: list[str],
    metric: str,
    fixed_threshold: bool = False,
) -> float:
    metric_key = "test_at_0.5" if fixed_threshold else "test"
    return float(np.mean([run[metric_key][split][metric] for split in splits]))


def summarize_metrics(
    runs: list[dict[str, Any]],
    splits: list[str],
    fixed_threshold: bool = False,
) -> dict[str, Any]:
    output = {}
    for metric in METRICS:
        values = np.asarray(
            [
                _mean_metric(run, splits, metric, fixed_threshold)
                for run in runs
            ],
            dtype=float,
        )
        output[metric] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "values": values.tolist(),
        }
    output["n_seeds"] = len(runs)
    output["splits"] = splits
    return output


def compare_metric_runs(
    first_runs: list[dict[str, Any]],
    second_runs: list[dict[str, Any]],
    splits: list[str],
    metric: str,
    fixed_threshold: bool = False,
) -> dict[str, Any]:
    first = np.asarray(
        [
            _mean_metric(run, splits, metric, fixed_threshold)
            for run in first_runs
        ],
        dtype=float,
    )
    second = np.asarray(
        [
            _mean_metric(run, splits, metric, fixed_threshold)
            for run in second_runs
        ],
        dtype=float,
    )
    result = paired_test(first, second)
    result["metric"] = metric
    result["splits"] = splits
    result["fixed_threshold"] = fixed_threshold
    result["first_mean"] = float(first.mean())
    result["second_mean"] = float(second.mean())
    return result


def apply_metric_holm(
    family: dict[str, dict[str, Any]],
    p_key: str = "p",
) -> None:
    keys = list(family)
    adjusted = holm_adjust([float(family[key][p_key]) for key in keys])
    output_key = "p_holm" if p_key == "p" else "p_bootstrap_holm"
    for key, p_value in zip(keys, adjusted):
        family[key][output_key] = float(p_value)


def apply_bootstrap_family_holm(
    family: dict[str, dict[str, Any]],
) -> None:
    for metric in METRICS:
        apply_metric_holm(
            {
                name: result["metrics"][metric]
                for name, result in family.items()
            },
            p_key="p_two_sided",
        )


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
        "probabilities": probabilities,
    }


def _metric_values(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    predictions = (probabilities >= threshold).astype(np.int8)
    tp = int(np.logical_and(predictions == 1, labels == 1).sum())
    tn = int(np.logical_and(predictions == 0, labels == 0).sum())
    fp = int(np.logical_and(predictions == 1, labels == 0).sum())
    fn = int(np.logical_and(predictions == 0, labels == 1).sum())
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    has_two_classes = len(np.unique(labels)) > 1
    return {
        "MCC": (
            float((tp * tn - fp * fn) / denominator)
            if denominator
            else 0.0
        ),
        "AUC": (
            float(roc_auc_score(labels, probabilities))
            if has_two_classes
            else None
        ),
        "AUPRC": (
            float(average_precision_score(labels, probabilities))
            if has_two_classes
            else None
        ),
        "F1": (
            float(2 * tp / (2 * tp + fp + fn))
            if (2 * tp + fp + fn)
            else 0.0
        ),
        "Precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "Recall": recall,
        "GMean": float(np.sqrt(recall * specificity)),
    }


def bootstrap_metric_compare(
    first_runs: list[dict[str, Any]],
    second_runs: list[dict[str, Any]],
    splits: list[str],
    n_resamples: int = 2000,
    seed: int = 20261024,
    fixed_threshold: bool = False,
) -> dict[str, Any]:
    if len(first_runs) != len(second_runs):
        raise ValueError("paired methods must have the same number of seeds")
    if any(
        int(left["seed"]) != int(right["seed"])
        for left, right in zip(first_runs, second_runs)
    ):
        raise ValueError("seed ordering differs")

    packed = {}
    reference_offsets = {}
    observed_rows = {metric: [] for metric in METRICS}
    seeds = [int(run["seed"]) for run in first_runs]
    for split in splits:
        reference_offsets[split] = None
        for left, right in zip(first_runs, second_runs):
            left_threshold = 0.5 if fixed_threshold else float(left["threshold"])
            right_threshold = 0.5 if fixed_threshold else float(right["threshold"])
            left_pack = _pack_chains(left["per_chain"][split], left_threshold)
            right_pack = _pack_chains(right["per_chain"][split], right_threshold)
            if left_pack["ids"] != right_pack["ids"]:
                raise ValueError(f"{split}: chain IDs differ")
            if reference_offsets[split] is None:
                reference_offsets[split] = left_pack["offsets"]
            elif not np.array_equal(reference_offsets[split], left_pack["offsets"]):
                raise ValueError(f"{split}: chain lengths differ across seeds")
            packed[(split, int(left["seed"]), "first")] = left_pack
            packed[(split, int(left["seed"]), "second")] = right_pack

    for split in splits:
        for seed_value in seeds:
            left = packed[(split, seed_value, "first")]
            right = packed[(split, seed_value, "second")]
            left_metrics = _metric_values(
                left["labels"],
                left["probabilities"],
                float(
                    (
                        0.5
                        if fixed_threshold
                        else next(
                            run["threshold"]
                            for run in first_runs
                            if int(run["seed"]) == seed_value
                        )
                    )
                ),
            )
            right_metrics = _metric_values(
                right["labels"],
                right["probabilities"],
                float(
                    (
                        0.5
                        if fixed_threshold
                        else next(
                            run["threshold"]
                            for run in second_runs
                            if int(run["seed"]) == seed_value
                        )
                    )
                ),
            )
            for metric in METRICS:
                if (
                    left_metrics[metric] is not None
                    and right_metrics[metric] is not None
                ):
                    observed_rows[metric].append(
                        float(left_metrics[metric] - right_metrics[metric])
                    )

    observed = {
        metric: float(np.mean(values))
        for metric, values in observed_rows.items()
    }
    rng = np.random.default_rng(seed)
    boot = {metric: [] for metric in METRICS}
    invalid = {metric: 0 for metric in METRICS}
    for _ in range(n_resamples):
        flat_by_split = {}
        for split in splits:
            offsets = reference_offsets[split]
            sampled = rng.integers(0, len(offsets) - 1, size=len(offsets) - 1)
            flat_by_split[split] = np.concatenate(
                [np.arange(offsets[index], offsets[index + 1]) for index in sampled]
            )
        deltas = {metric: [] for metric in METRICS}
        for split in splits:
            flat = flat_by_split[split]
            for seed_value in seeds:
                left = packed[(split, seed_value, "first")]
                right = packed[(split, seed_value, "second")]
                left_metrics = _metric_values(
                    left["labels"][flat],
                    left["probabilities"][flat],
                    float(
                        (
                            0.5
                            if fixed_threshold
                            else next(
                                run["threshold"]
                                for run in first_runs
                                if int(run["seed"]) == seed_value
                            )
                        )
                    ),
                )
                right_metrics = _metric_values(
                    right["labels"][flat],
                    right["probabilities"][flat],
                    float(
                        (
                            0.5
                            if fixed_threshold
                            else next(
                                run["threshold"]
                                for run in second_runs
                                if int(run["seed"]) == seed_value
                            )
                        )
                    ),
                )
                for metric in METRICS:
                    if (
                        left_metrics[metric] is not None
                        and right_metrics[metric] is not None
                    ):
                        deltas[metric].append(
                            float(left_metrics[metric] - right_metrics[metric])
                        )
        for metric in METRICS:
            expected = len(splits) * len(seeds)
            if len(deltas[metric]) != expected:
                invalid[metric] += 1
                continue
            boot[metric].append(float(np.mean(deltas[metric])))

    output = {}
    for metric in METRICS:
        values = np.asarray(boot[metric], dtype=float)
        low = int((values <= 0).sum())
        high = int((values >= 0).sum())
        output[metric] = {
            "observed_delta": observed[metric],
            "ci95": [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ],
            "p_two_sided": float(
                2
                * min(
                    (low + 1) / (len(values) + 1),
                    (high + 1) / (len(values) + 1),
                )
            ),
            "valid_resamples": int(len(values)),
            "invalid_resamples": int(invalid[metric]),
        }
    return {
        "splits": splits,
        "fixed_threshold": fixed_threshold,
        "n_resamples_requested": n_resamples,
        "bootstrap_seed": seed,
        "n_seeds": len(seeds),
        "metrics": output,
    }
