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
from duet_clean.large_stats import FOUR_SPLITS, SHARED_SPLITS


SEEDS = [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]
METHODS = ["duet_graph", "grassp_rnafm"]
VARIANTS = ["group", "uniform", "dedup"]
PAIRS = {
    "uniform_vs_group": ("uniform", "group"),
    "dedup_vs_group": ("dedup", "group"),
    "dedup_vs_uniform": ("dedup", "uniform"),
}


def load(variant: str, method: str, seed: int) -> dict:
    return json.loads(
        (
            ROOT
            / "results"
            / "followup"
            / "duplicate_weighting"
            / variant
            / "final"
            / method
            / f"seed{seed}.json"
        ).read_text(encoding="utf-8")
    )


def main() -> int:
    summaries = {}
    comparisons = {}
    bootstrap = {}
    for method in METHODS:
        splits = FOUR_SPLITS if method == "duet_graph" else SHARED_SPLITS
        runs = {
            variant: [load(variant, method, seed) for seed in SEEDS]
            for variant in VARIANTS
        }
        summaries[method] = {
            variant: summarize_metrics(method_runs, splits)
            for variant, method_runs in runs.items()
        }
        comparisons[method] = {}
        bootstrap[method] = {}
        for metric in METRICS:
            rows = {
                name: compare_metric_runs(
                    runs[first],
                    runs[second],
                    splits,
                    metric,
                )
                for name, (first, second) in PAIRS.items()
            }
            apply_metric_holm(rows)
            comparisons[method][metric] = rows
        for name, (first, second) in PAIRS.items():
            result = bootstrap_metric_compare(
                runs[first],
                runs[second],
                splits,
                n_resamples=2000,
                seed=20261024,
            )
            bootstrap[method][name] = result
        apply_bootstrap_family_holm(bootstrap[method])
    report = {
        "protocol": {
            "seeds": SEEDS,
            "variants": VARIANTS,
            "dedup_manifest": "followup_dedup_290_cv5",
            "group_results_source": "results/large_protocol/final",
            "same_epochs_and_thresholds": True,
        },
        "summaries": summaries,
        "comparisons": comparisons,
        "chain_bootstrap": bootstrap,
    }
    root = ROOT / "results" / "followup" / "duplicate_weighting"
    output = root / "analysis.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Exact-duplicate weighting ablation",
        "",
        "All three strategies use the same folds, ten seeds, selected epochs and "
        "thresholds. The deduplicated manifest removes exact duplicate records; "
        "label-conflicting exact families are dropped entirely.",
        "",
        "## Metric summaries",
        "",
        "| Model | Strategy | Metric | Mean | SD |",
        "|:--|:--|:--|--:|--:|",
    ]
    for method in METHODS:
        for variant in VARIANTS:
            for metric in METRICS:
                value = summaries[method][variant][metric]
                lines.append(
                    f"| {method} | {variant} | {metric} | "
                    f"{value['mean']:.4f} | {value['sd']:.4f} |"
                )
    lines += [
        "",
        "## Paired comparisons",
        "",
        "| Model | Metric | Comparison | Delta | 95% CI | raw p | Holm p |",
        "|:--|:--|:--|--:|:--|--:|--:|",
    ]
    for method in METHODS:
        for metric in METRICS:
            for name, value in comparisons[method][metric].items():
                lines.append(
                    f"| {method} | {metric} | {name} | "
                    f"{value['mean_difference']:+.4f} | "
                    f"{value['ci95'][0]:+.4f} to {value['ci95'][1]:+.4f} | "
                    f"{value['p']:.4g} | {value['p_holm']:.4g} |"
                )
    (root / "DUPLICATE_WEIGHTING.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
