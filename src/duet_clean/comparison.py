from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from .metrics import compute_metrics, find_best_threshold


EXTERNAL_TESTS = ["TE18", "TL12", "JL10", "RB9"]
SEEDS = [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]


def load_result(path: str | Path) -> dict[str, Any]:
    import json

    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def pooled_from_per_chain(
    chains: dict[str, dict[str, Any]],
    threshold: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for rna_id in sorted(chains):
        payload = chains[rna_id]
        labels.append(np.asarray(payload["label"], dtype=int))
        probabilities.append(np.asarray(payload["prob"], dtype=float))
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    return compute_metrics(y_true, y_prob, threshold), chains


def merge_member_probabilities(
    member_runs: list[dict[str, Any]],
    split: str,
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    reference = member_runs[0]["per_chain"][split]
    for rna_id in sorted(reference):
        labels = np.asarray(reference[rna_id]["label"], dtype=int)
        probabilities = []
        for run in member_runs:
            payload = run["per_chain"][split][rna_id]
            if labels.tolist() != payload["label"]:
                raise ValueError(f"{split}/{rna_id}: label mismatch across DUET members")
            probabilities.append(np.asarray(payload["prob"], dtype=float))
        merged[rna_id] = {
            "label": labels.tolist(),
            "prob": np.mean(probabilities, axis=0).tolist(),
        }
    return merged


def ensemble_run(
    member_runs: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    val_chains = merge_member_probabilities(member_runs, "val")
    val_labels = np.concatenate(
        [np.asarray(value["label"], dtype=int) for value in val_chains.values()]
    )
    val_prob = np.concatenate(
        [np.asarray(value["prob"], dtype=float) for value in val_chains.values()]
    )
    threshold, threshold_score = find_best_threshold(val_labels, val_prob)

    metrics: dict[str, Any] = {}
    metrics_05: dict[str, Any] = {}
    per_chain: dict[str, dict[str, dict[str, Any]]] = {"val": val_chains}
    for split in EXTERNAL_TESTS:
        chains = merge_member_probabilities(member_runs, split)
        per_chain[split] = chains
        metrics[split], _ = pooled_from_per_chain(chains, threshold)
        metrics_05[split], _ = pooled_from_per_chain(chains, 0.5)

    average_mcc = float(np.mean([metrics[split]["MCC"] for split in EXTERNAL_TESTS]))
    average_mcc_05 = float(
        np.mean([metrics_05[split]["MCC"] for split in EXTERNAL_TESTS])
    )
    return {
        "seed": seed,
        "threshold": threshold,
        "validation_threshold_score": threshold_score,
        "metrics": metrics,
        "metrics_at_0.5": metrics_05,
        "average_mcc": average_mcc,
        "average_mcc_at_0.5": average_mcc_05,
        "per_chain": per_chain,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = sorted(range(n), key=lambda index: p_values[index])
    adjusted = [1.0] * n
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (n - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_test(
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, Any]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    delta = first - second
    n = len(delta)
    mean = float(delta.mean())
    sd = float(delta.std(ddof=1))
    se = sd / np.sqrt(n)
    t_critical = float(stats.t.ppf(0.975, n - 1))
    t_result = stats.ttest_rel(first, second)
    wilcoxon = stats.wilcoxon(first, second) if n >= 5 else None
    return {
        "n": n,
        "mean_difference": mean,
        "difference_sd": sd,
        "ci95": [mean - t_critical * se, mean + t_critical * se],
        "t": float(t_result.statistic),
        "p": float(t_result.pvalue),
        "wilcoxon_p": float(wilcoxon.pvalue) if wilcoxon is not None else None,
        "seeds_first_better": int((delta > 0).sum()),
    }


def summarize_method(
    runs: list[dict[str, Any]],
    metric_getter,
) -> dict[str, Any]:
    per_split: dict[str, Any] = {}
    for split in EXTERNAL_TESTS:
        values = np.asarray([metric_getter(run, split) for run in runs], dtype=float)
        per_split[split] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "values": values.tolist(),
        }
    averages = np.asarray(
        [
            np.mean([metric_getter(run, split) for split in EXTERNAL_TESTS])
            for run in runs
        ]
    )
    return {
        "average": {
            "mean": float(averages.mean()),
            "sd": float(averages.std(ddof=1)),
            "values": averages.tolist(),
        },
        "per_split": per_split,
    }


def hierarchical_bootstrap_delta(
    duet_runs: list[dict[str, Any]],
    cobra_runs: dict[int, dict[str, Any]],
    n_resamples: int = 2000,
    seed: int = 20261023,
) -> dict[str, Any]:
    """Resample seeds and test chains within each benchmark."""

    def fast_mcc(labels: np.ndarray, predictions: np.ndarray) -> float:
        tp = float(np.logical_and(predictions == 1, labels == 1).sum())
        tn = float(np.logical_and(predictions == 0, labels == 0).sum())
        fp = float(np.logical_and(predictions == 1, labels == 0).sum())
        fn = float(np.logical_and(predictions == 0, labels == 1).sum())
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return float((tp * tn - fp * fn) / denominator) if denominator else 0.0

    def pack(per_chain: dict[str, dict[str, Any]], threshold: float) -> dict[str, Any]:
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

    packed_duet = {}
    packed_cobra = {}
    observed_by_seed = {}
    for run in duet_runs:
        seed_value = int(run["seed"])
        cobra_run = cobra_runs[seed_value]
        split_deltas = []
        for split in EXTERNAL_TESTS:
            duet_pack = pack(run["per_chain"][split], run["threshold"])
            cobra_pack = pack(cobra_run["per_chain"][split], float(cobra_run["threshold"]))
            if duet_pack["ids"] != cobra_pack["ids"]:
                raise ValueError(f"{split}/{seed_value}: chain sets differ")
            packed_duet[(seed_value, split)] = duet_pack
            packed_cobra[(seed_value, split)] = cobra_pack
            split_deltas.append(
                fast_mcc(duet_pack["labels"], duet_pack["predictions"])
                - fast_mcc(cobra_pack["labels"], cobra_pack["predictions"])
            )
        observed_by_seed[seed_value] = float(np.mean(split_deltas))
    observed_delta = float(np.mean(list(observed_by_seed.values())))

    rng = np.random.default_rng(seed)
    seeds = [int(run["seed"]) for run in duet_runs]
    boot = []
    for _ in range(n_resamples):
        seed_value = seeds[int(rng.integers(0, len(seeds)))]
        split_deltas = []
        for split in EXTERNAL_TESTS:
            duet_pack = packed_duet[(seed_value, split)]
            cobra_pack = packed_cobra[(seed_value, split)]
            offsets = duet_pack["offsets"]
            sampled = rng.integers(0, len(offsets) - 1, size=len(offsets) - 1)
            flat_indices = np.concatenate(
                [np.arange(offsets[index], offsets[index + 1]) for index in sampled]
            )
            split_deltas.append(
                fast_mcc(
                    duet_pack["labels"][flat_indices],
                    duet_pack["predictions"][flat_indices],
                )
                - fast_mcc(
                    cobra_pack["labels"][flat_indices],
                    cobra_pack["predictions"][flat_indices],
                )
            )
        boot.append(float(np.mean(split_deltas)))

    values = np.asarray(boot)
    return {
        "observed_delta": observed_delta,
        "ci95": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
        "p_two_sided": float(2 * min((values <= 0).mean(), (values >= 0).mean())),
        "n_resamples": n_resamples,
        "bootstrap_seed": seed,
    }


def chain_bootstrap_delta(
    duet_runs: list[dict[str, Any]],
    cobra_runs: dict[int, dict[str, Any]],
    n_resamples: int = 2000,
    seed: int = 20261023,
) -> dict[str, Any]:
    """Bootstrap test chains while keeping the ten paired seeds fixed."""

    def fast_mcc(labels: np.ndarray, predictions: np.ndarray) -> float:
        tp = float(np.logical_and(predictions == 1, labels == 1).sum())
        tn = float(np.logical_and(predictions == 0, labels == 0).sum())
        fp = float(np.logical_and(predictions == 1, labels == 0).sum())
        fn = float(np.logical_and(predictions == 0, labels == 1).sum())
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return float((tp * tn - fp * fn) / denominator) if denominator else 0.0

    def pack(per_chain: dict[str, dict[str, Any]], threshold: float) -> dict[str, Any]:
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

    packed: dict[tuple[int, str, str], dict[str, Any]] = {}
    observed = []
    for run in duet_runs:
        seed_value = int(run["seed"])
        split_deltas = []
        for split in EXTERNAL_TESTS:
            duet_pack = pack(run["per_chain"][split], run["threshold"])
            cobra_pack = pack(
                cobra_runs[seed_value]["per_chain"][split],
                float(cobra_runs[seed_value]["threshold"]),
            )
            packed[(seed_value, split, "duet")] = duet_pack
            packed[(seed_value, split, "cobra")] = cobra_pack
            split_deltas.append(
                fast_mcc(duet_pack["labels"], duet_pack["predictions"])
                - fast_mcc(cobra_pack["labels"], cobra_pack["predictions"])
            )
        observed.append(float(np.mean(split_deltas)))
    observed_delta = float(np.mean(observed))

    rng = np.random.default_rng(seed)
    seeds = [int(run["seed"]) for run in duet_runs]
    boot = []
    for _ in range(n_resamples):
        all_deltas = []
        for split in EXTERNAL_TESTS:
            reference = packed[(seeds[0], split, "duet")]
            offsets = reference["offsets"]
            sampled = rng.integers(0, len(offsets) - 1, size=len(offsets) - 1)
            flat_indices = np.concatenate(
                [np.arange(offsets[index], offsets[index + 1]) for index in sampled]
            )
            for seed_value in seeds:
                duet_pack = packed[(seed_value, split, "duet")]
                cobra_pack = packed[(seed_value, split, "cobra")]
                all_deltas.append(
                    fast_mcc(
                        duet_pack["labels"][flat_indices],
                        duet_pack["predictions"][flat_indices],
                    )
                    - fast_mcc(
                        cobra_pack["labels"][flat_indices],
                        cobra_pack["predictions"][flat_indices],
                    )
                )
        boot.append(float(np.mean(all_deltas)))

    values = np.asarray(boot)
    return {
        "observed_delta": observed_delta,
        "ci95": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
        "p_two_sided": float(2 * min((values <= 0).mean(), (values >= 0).mean())),
        "n_resamples": n_resamples,
        "bootstrap_seed": seed,
    }
