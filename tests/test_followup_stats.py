from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.followup_stats import (
    apply_bootstrap_family_holm,
    bootstrap_metric_compare,
    compare_metric_runs,
    summarize_metrics,
)


def run(seed: int, offset: float, probabilities: list[float] | None = None) -> dict:
    metrics = {
        "MCC": 0.5 + offset,
        "AUC": 0.7 + offset,
        "AUPRC": 0.6 + offset,
        "F1": 0.55 + offset,
        "Precision": 0.58 + offset,
        "Recall": 0.52 + offset,
        "GMean": 0.54 + offset,
    }
    return {
        "seed": seed,
        "threshold": 0.5,
        "test": {"TE18": metrics},
        "test_at_0.5": metrics,
        "per_chain": {
            "TE18": {
                "a": {
                    "label": [0, 1, 0, 1],
                    "prob": probabilities or [0.1, 0.8, 0.2, 0.7],
                },
                "b": {"label": [0, 1, 1, 0], "prob": [0.2, 0.7, 0.6, 0.4]},
            }
        },
    }


class FollowupStatsTests(unittest.TestCase):
    def test_summary_contains_all_metrics(self) -> None:
        summary = summarize_metrics([run(42, 0.0), run(0, 0.1)], ["TE18"])
        for metric in ["MCC", "AUC", "AUPRC", "F1", "Precision", "Recall", "GMean"]:
            self.assertIn(metric, summary)
            self.assertEqual(len(summary[metric]["values"]), 2)

    def test_paired_metric_comparison(self) -> None:
        first = [run(42, 0.2), run(0, 0.3)]
        second = [run(42, 0.0), run(0, 0.0)]
        result = compare_metric_runs(first, second, ["TE18"], "MCC")
        self.assertAlmostEqual(result["mean_difference"], 0.25)

    def test_bootstrap_all_metrics(self) -> None:
        first = [
            run(seed, 0.1, [0.05, 0.95, 0.05, 0.95])
            for seed in [42, 0, 1]
        ]
        second = [
            run(seed, 0.0, [0.60, 0.40, 0.60, 0.40])
            for seed in [42, 0, 1]
        ]
        result = bootstrap_metric_compare(
            first,
            second,
            ["TE18"],
            n_resamples=20,
        )
        self.assertEqual(set(result["metrics"]), {
            "MCC",
            "AUC",
            "AUPRC",
            "F1",
            "Precision",
            "Recall",
            "GMean",
        })
        self.assertGreater(result["metrics"]["MCC"]["observed_delta"], 0)

    def test_bootstrap_holm_is_within_metric_across_family(self) -> None:
        metrics = {
            metric: {"p_two_sided": p_value}
            for metric in [
                "MCC",
                "AUC",
                "AUPRC",
                "F1",
                "Precision",
                "Recall",
                "GMean",
            ]
            for p_value in [0.01]
        }
        family = {
            "first": {"metrics": metrics},
            "second": {
                "metrics": {
                    metric: {"p_two_sided": 0.04}
                    for metric in metrics
                }
            },
        }
        apply_bootstrap_family_holm(family)
        self.assertAlmostEqual(
            family["first"]["metrics"]["MCC"]["p_bootstrap_holm"],
            0.02,
        )
        self.assertAlmostEqual(
            family["second"]["metrics"]["MCC"]["p_bootstrap_holm"],
            0.04,
        )


if __name__ == "__main__":
    unittest.main()
