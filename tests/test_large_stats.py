from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.large_stats import (
    chain_bootstrap_compare,
    compare_runs,
    merge_ensemble,
)


def result(seed: int, probabilities: list[float]) -> dict:
    labels = [0, 1, 0, 1]
    return {
        "seed": seed,
        "threshold": 0.5,
        "test": {"TE18": {"MCC": 0.5}},
        "per_chain": {
            "TE18": {
                "chain": {
                    "label": labels,
                    "prob": probabilities,
                }
            }
        },
    }


class LargeStatsTests(unittest.TestCase):
    def test_ensemble_average_is_probability_mean(self) -> None:
        first = result(42, [0.1, 0.8, 0.2, 0.7])
        second = result(42, [0.3, 0.6, 0.4, 0.9])
        merged = merge_ensemble([first, second], 0.5)
        actual = merged["per_chain"]["TE18"]["chain"]["prob"]
        for value, expected in zip(actual, [0.2, 0.7, 0.3, 0.8]):
            self.assertAlmostEqual(value, expected)

    def test_identical_methods_have_zero_bootstrap_delta(self) -> None:
        runs = [result(seed, [0.1, 0.8, 0.2, 0.7]) for seed in [42, 0, 1]]
        comparison = chain_bootstrap_compare(
            runs,
            runs,
            ["TE18"],
            n_resamples=100,
        )
        self.assertEqual(comparison["observed_delta"], 0.0)
        self.assertEqual(comparison["ci95"], [0.0, 0.0])

    def test_paired_comparison_uses_seed_order(self) -> None:
        first = [
            {**result(42, [0.1, 0.8, 0.2, 0.7]), "test": {"TE18": {"MCC": 0.6}}},
            {**result(0, [0.1, 0.8, 0.2, 0.7]), "test": {"TE18": {"MCC": 0.5}}},
        ]
        second = [
            {**result(42, [0.1, 0.8, 0.2, 0.7]), "test": {"TE18": {"MCC": 0.4}}},
            {**result(0, [0.1, 0.8, 0.2, 0.7]), "test": {"TE18": {"MCC": 0.3}}},
        ]
        comparison = compare_runs(first, second, ["TE18"])
        self.assertAlmostEqual(comparison["mean_difference"], 0.2)
        self.assertEqual(comparison["seeds_first_better"], 2)


if __name__ == "__main__":
    unittest.main()
