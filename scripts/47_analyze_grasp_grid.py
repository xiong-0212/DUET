from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.followup_stats import (
    METRICS,
    apply_bootstrap_family_holm,
    apply_metric_holm,
    bootstrap_metric_compare,
    compare_metric_runs,
    summarize_metrics,
)
from duet_clean.large_stats import SHARED_SPLITS, load_final


SEEDS = [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]
FEATURES = ["ernie", "rnafm"]


def load_best(feature: str, seed: int) -> dict:
    return json.loads(
        (
            ROOT
            / "results"
            / "followup"
            / "grasp_grid"
            / "best"
            / feature
            / "final"
            / f"grassp_{feature}"
            / f"seed{seed}.json"
        ).read_text(encoding="utf-8")
    )


def main() -> int:
    selections = {}
    best_runs = {}
    for feature in FEATURES:
        selection_path = (
            ROOT
            / "results"
            / "followup"
            / "grasp_grid"
            / feature
            / "grid_selection.json"
        )
        selections[feature] = json.loads(
            selection_path.read_text(encoding="utf-8")
        )
        best_runs[feature] = [load_best(feature, seed) for seed in SEEDS]
    duet_runs = [load_final("duet_ensemble", seed) for seed in SEEDS]
    summaries = {
        feature: summarize_metrics(best_runs[feature], SHARED_SPLITS)
        for feature in FEATURES
    }
    comparisons = {}
    for metric in METRICS:
        rows = {
            "grassp_ernie_vs_duet": compare_metric_runs(
                best_runs["ernie"],
                duet_runs,
                SHARED_SPLITS,
                metric,
            ),
            "grassp_rnafm_vs_duet": compare_metric_runs(
                best_runs["rnafm"],
                duet_runs,
                SHARED_SPLITS,
                metric,
            ),
            "grassp_ernie_vs_rnafm": compare_metric_runs(
                best_runs["ernie"],
                best_runs["rnafm"],
                SHARED_SPLITS,
                metric,
            ),
        }
        apply_metric_holm(rows)
        comparisons[metric] = rows
    bootstrap = {}
    for name, first, second in [
        ("grassp_ernie_vs_duet", "ernie", "duet"),
        ("grassp_rnafm_vs_duet", "rnafm", "duet"),
        ("grassp_ernie_vs_rnafm", "ernie", "rnafm"),
    ]:
        runs = {
            "ernie": best_runs["ernie"],
            "rnafm": best_runs["rnafm"],
            "duet": duet_runs,
        }
        result = bootstrap_metric_compare(
            runs[first],
            runs[second],
            SHARED_SPLITS,
            n_resamples=1000,
            seed=20261024,
        )
        bootstrap[name] = result
    apply_bootstrap_family_holm(bootstrap)
    report = {
        "protocol": {
            "seeds": SEEDS,
            "splits": SHARED_SPLITS,
            "grid": {
                "learning_rate": [1e-4, 1e-3, 1e-2],
                "batch_size": [1, 16],
                "hidden": [86, 128],
                "dropout": [0.3, 0.6],
            },
            "selection_metric": "pooled out-of-fold MCC",
        },
        "selections": selections,
        "summaries": summaries,
        "comparisons": comparisons,
        "chain_bootstrap": bootstrap,
    }
    root = ROOT / "results" / "followup" / "grasp_grid"
    output = root / "analysis.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# GRASSP hyperparameter sensitivity",
        "",
        "The full 24-point author grid was evaluated on the fixed large-data "
        "fold 0 for each encoder with seed 42. The top three configurations "
        "were then confirmed on folds 1-4. Selection used pooled five-fold OOF "
        "MCC; the selected configuration was retrained with ten paired seeds.",
        "",
        "| Encoder | Selected configuration | OOF MCC | OOF AUC | OOF AUPRC |",
        "|:--|:--|--:|--:|--:|",
    ]
    for feature in FEATURES:
        best = selections[feature]["best"]
        metrics = best["oof_metrics_at_selected_threshold"]
        lines.append(
            f"| {feature} | {best['name']} | {metrics['MCC']:.4f} | "
            f"{metrics['AUC']:.4f} | {metrics['AUPRC']:.4f} |"
        )
    lines += [
        "",
        "## Final ten-seed summaries",
        "",
        "| Model | Metric | Mean | SD |",
        "|:--|:--|--:|--:|",
    ]
    best_ensemble_summed = {
        "DUET ensemble": summarize_metrics(duet_runs, SHARED_SPLITS),
        **summaries,
    }
    for model, summary in best_ensemble_summed.items():
        for metric in METRICS:
            lines.append(
                f"| {model} | {metric} | {summary[metric]['mean']:.4f} | "
                f"{summary[metric]['sd']:.4f} |"
            )
    lines += [
        "",
        "## Final ten-seed comparisons",
        "",
        "| Metric | Comparison | Delta | 95% CI | raw p | Holm p |",
        "|:--|:--|--:|:--|--:|--:|",
    ]
    for metric in METRICS:
        for name, value in comparisons[metric].items():
            lines.append(
                f"| {metric} | {name} | {value['mean_difference']:+.4f} | "
                f"{value['ci95'][0]:+.4f} to {value['ci95'][1]:+.4f} | "
                f"{value['p']:.4g} | {value['p_holm']:.4g} |"
            )
    (root / "GRASP_GRID.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
