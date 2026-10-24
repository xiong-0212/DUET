from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

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


REPEATS = [101, 102, 103]
SEEDS = [42, 0, 1, 2, 3]
METHODS = [
    "cobra",
    "duet_graph",
    "duet_tcn",
    "duet_dense",
    "duet_ensemble",
    "graph_seq3d",
    "graph_seq3d_ensemble",
    "grassp_rnafm",
]
FOUR_SPLIT_METHODS = {
    "cobra",
    "duet_graph",
    "duet_tcn",
    "duet_dense",
    "duet_ensemble",
}
PAIRS = {
    "duet_ensemble_vs_cobra": ("duet_ensemble", "cobra"),
    "grassp_rnafm_vs_duet_ensemble": ("grassp_rnafm", "duet_ensemble"),
    "graph_seq3d_vs_duet_graph": ("graph_seq3d", "duet_graph"),
}


def root_for(repeat: int) -> Path:
    return (
        ROOT
        / "results"
        / "followup"
        / "repeated_cv"
        / f"cvrep_{repeat}"
    )


def load(run_root: Path, method: str, seed: int) -> dict:
    return json.loads(
        (run_root / "final" / method / f"seed{seed}.json").read_text(
            encoding="utf-8"
        )
    )


def write_markdown(report: dict) -> Path:
    lines = [
        "# Repeated cluster-stratified CV",
        "",
        "Three independent five-fold cluster assignments were generated. "
        "Each assignment keeps whole 80% identity components together and "
        "balances chain and positive-residue counts. Final models use the same "
        "five paired seeds for every partition. The reduced seed count is "
        "explicit because each partition contains five CV fits and five final "
        "fits for each method.",
        "",
        "## Partition balance",
        "",
        "| Assignment | Fold sizes | Positive residues |",
        "|--:|:--|:--|",
    ]
    for repeat, audit in report["partitions"].items():
        lines.append(
            f"| {repeat} | {audit['fold_sizes']} | "
            f"{audit['fold_positive_residues']} |"
        )
    lines += [
        "",
        "## Selected hyperparameters",
        "",
        "| Assignment | Method | Epoch | Threshold |",
        "|--:|:--|--:|--:|",
    ]
    for repeat, methods in report["selected"].items():
        for method, value in methods.items():
            lines.append(
                f"| {repeat} | {method} | {value['selected_epoch']} | "
                f"{value['threshold']:.4f} |"
            )
    lines += [
        "",
        "## Final-test comparisons",
        "",
        "| Assignment | Metric | Comparison | Delta | 95% CI | raw p | Holm p |",
        "|--:|:--|:--|--:|:--|--:|--:|",
    ]
    for repeat, metrics in report["comparisons"].items():
        for metric, rows in metrics.items():
            for name, value in rows.items():
                lines.append(
                    f"| {repeat} | {metric} | {name} | "
                    f"{value['mean_difference']:+.4f} | "
                    f"{value['ci95'][0]:+.4f} to {value['ci95'][1]:+.4f} | "
                    f"{value['p']:.4g} | {value['p_holm']:.4g} |"
                )
    lines += [
        "",
        "## All-metric method summaries",
        "",
        "| Assignment | Method | Metric | Mean | SD |",
        "|--:|:--|:--|--:|--:|",
    ]
    for repeat, methods in report["summaries"].items():
        for method, summary in methods.items():
            for metric in METRICS:
                lines.append(
                    f"| {repeat} | {method} | {metric} | "
                    f"{summary[metric]['mean']:.4f} | "
                    f"{summary[metric]['sd']:.4f} |"
                )
    path = ROOT / "results" / "followup" / "repeated_cv" / "REPEATED_CV.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    protocol_audit = json.loads(
        (
            ROOT
            / "results"
            / "followup"
            / "protocol"
            / "manifest_audit.json"
        ).read_text(encoding="utf-8")
    )["repeated_cv"]
    partitions = {}
    selected = {}
    summaries = {}
    comparisons = {}
    chain_bootstrap = {}
    for repeat in REPEATS:
        run_root = root_for(repeat)
        selected[repeat] = {}
        summaries[repeat] = {}
        for method in METHODS:
            selected_path = run_root / "cv" / method / "selected.json"
            selected[repeat][method] = json.loads(
                selected_path.read_text(encoding="utf-8")
            )
            runs = [load(run_root, method, seed) for seed in SEEDS]
            summaries[repeat][method] = summarize_metrics(
                runs,
                FOUR_SPLITS if method in FOUR_SPLIT_METHODS else SHARED_SPLITS,
            )
        partitions[repeat] = protocol_audit[str(repeat)]
        comparisons[repeat] = {}
        chain_bootstrap[repeat] = {}
        for metric in METRICS:
            rows = {}
            for name, (first, second) in PAIRS.items():
                splits = (
                    FOUR_SPLITS
                    if first in FOUR_SPLIT_METHODS
                    else SHARED_SPLITS
                )
                first_runs = [load(run_root, first, seed) for seed in SEEDS]
                second_runs = [load(run_root, second, seed) for seed in SEEDS]
                rows[name] = compare_metric_runs(
                    first_runs,
                    second_runs,
                    splits,
                    metric,
                )
            apply_metric_holm(rows)
            comparisons[repeat][metric] = rows
        for name, (first, second) in PAIRS.items():
            splits = (
                FOUR_SPLITS
                if first in FOUR_SPLIT_METHODS
                else SHARED_SPLITS
            )
            result = bootstrap_metric_compare(
                [load(run_root, first, seed) for seed in SEEDS],
                [load(run_root, second, seed) for seed in SEEDS],
                splits,
                n_resamples=1000,
                seed=20261024 + repeat,
            )
            chain_bootstrap[repeat][name] = result
        apply_bootstrap_family_holm(chain_bootstrap[repeat])
    report = {
        "protocol": {
            "repeats": REPEATS,
            "seeds": SEEDS,
            "n_folds": 5,
            "bootstrap_resamples": 1000,
        },
        "partitions": partitions,
        "selected": selected,
        "summaries": summaries,
        "comparisons": comparisons,
        "chain_bootstrap": chain_bootstrap,
    }
    output = ROOT / "results" / "followup" / "repeated_cv" / "analysis.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = write_markdown(report)
    print(output)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
