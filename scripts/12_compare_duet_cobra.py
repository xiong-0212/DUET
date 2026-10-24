from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.comparison import (
    EXTERNAL_TESTS,
    SEEDS,
    ensemble_run,
    chain_bootstrap_delta,
    hierarchical_bootstrap_delta,
    holm_adjust,
    load_result,
    paired_test,
    summarize_method,
)
from duet_clean.io import write_json


PAPER_MCC = {"TE18": 0.190, "TL12": 0.546, "JL10": 0.293, "RB9": 0.557}
PAPER_AVG = 0.397


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        choices=["exact_sequence_unique", "homology_p80"],
        default="exact_sequence_unique",
    )
    args = parser.parse_args()
    if args.manifest == "exact_sequence_unique":
        prefix = "exact"
        cobra_file = "cobra_exact_clean.json"
    else:
        prefix = "p80"
        cobra_file = "cobra_homology_p80.json"
    towers = [f"{prefix}_graph", f"{prefix}_tcn", f"{prefix}_dense"]

    cobra = load_result(ROOT / "results" / "training" / cobra_file)
    cobra_runs = {int(run["seed"]): run for run in cobra["runs"]}
    member_runs: dict[str, dict[int, dict]] = {}
    for tower in towers:
        member_runs[tower] = {}
        for seed in SEEDS:
            path = (
                ROOT
                / "results"
                / "training"
                / "duet_runs"
                / f"train_{tower}_seed{seed}.json"
            )
            member_runs[tower][seed] = load_result(path)

    ensemble_runs = []
    for seed in SEEDS:
        ensemble_runs.append(
            ensemble_run(
                [member_runs[tower][seed] for tower in towers],
                seed,
            )
        )
    write_json(
        f"results/training/duet_ensemble_{prefix}.json",
        {"seeds": ensemble_runs},
    )

    def ensemble_metric(run, split):
        return run["metrics"][split]["MCC"]

    def cobra_metric(run, split):
        return run["test"][split]["MCC"]

    def tower_metric(tower):
        return lambda run, split: run["test_metrics"][split]["at_threshold"]["MCC"]

    summaries = {
        "CoBRA-M3": summarize_method(
            [cobra_runs[seed] for seed in SEEDS],
            cobra_metric,
        ),
        "DUET ensemble": summarize_method(ensemble_runs, ensemble_metric),
    }
    for tower in towers:
        summaries[tower] = summarize_method(
            [member_runs[tower][seed] for seed in SEEDS],
            tower_metric(tower),
        )

    ensemble_values = np.asarray(
        [
            np.mean([ensemble_metric(run, split) for split in EXTERNAL_TESTS])
            for run in ensemble_runs
        ]
    )
    cobra_values = np.asarray(
        [
            np.mean([cobra_metric(cobra_runs[seed], split) for split in EXTERNAL_TESTS])
            for seed in SEEDS
        ]
    )
    primary = paired_test(ensemble_values, cobra_values)
    component_tests = {}
    for tower in towers:
        tower_values = np.asarray(
            [
                np.mean(
                    [
                        member_runs[tower][seed]["test_metrics"][split]["at_threshold"]["MCC"]
                        for split in EXTERNAL_TESTS
                    ]
                )
                for seed in SEEDS
            ]
        )
        component_tests[f"{tower}_vs_cobra"] = paired_test(tower_values, cobra_values)
    for tower in towers:
        tower_values = np.asarray(
            [
                np.mean(
                    [
                        member_runs[tower][seed]["test_metrics"][split]["at_threshold"]["MCC"]
                        for split in EXTERNAL_TESTS
                    ]
                )
                for seed in SEEDS
            ]
        )
        component_tests[f"ensemble_vs_{tower}"] = paired_test(
            ensemble_values,
            tower_values,
        )

    per_split = {}
    raw_p_values = []
    for split in EXTERNAL_TESTS:
        first = np.asarray([ensemble_metric(run, split) for run in ensemble_runs])
        second = np.asarray([cobra_metric(cobra_runs[seed], split) for seed in SEEDS])
        result = paired_test(first, second)
        per_split[split] = result
        raw_p_values.append(result["p"])
    adjusted = holm_adjust(raw_p_values)
    for split, adjusted_p in zip(EXTERNAL_TESTS, adjusted):
        per_split[split]["p_holm"] = adjusted_p

    fixed_threshold = {"DUET ensemble": {}, "CoBRA-M3": {}}
    duet_average_05 = []
    cobra_average_05 = []
    for split in EXTERNAL_TESTS:
        duet_values = np.asarray(
            [run["metrics_at_0.5"][split]["MCC"] for run in ensemble_runs]
        )
        cobra_values_split = np.asarray(
            [
                compute_at_05(cobra_runs[seed], split)
                for seed in SEEDS
            ]
        )
        fixed_threshold["DUET ensemble"][split] = {
            "mean": float(duet_values.mean()),
            "sd": float(duet_values.std(ddof=1)),
        }
        fixed_threshold["CoBRA-M3"][split] = {
            "mean": float(cobra_values_split.mean()),
            "sd": float(cobra_values_split.std(ddof=1)),
        }
        fixed_threshold.setdefault("paired", {})[split] = paired_test(
            duet_values, cobra_values_split
        )
        if split == EXTERNAL_TESTS[0]:
            duet_average_05 = duet_values.copy()
            cobra_average_05 = cobra_values_split.copy()
        else:
            duet_average_05 = np.column_stack((duet_average_05, duet_values))
            cobra_average_05 = np.column_stack((cobra_average_05, cobra_values_split))
    fixed_threshold["average"] = paired_test(
        duet_average_05.mean(axis=1),
        cobra_average_05.mean(axis=1),
    )

    paper_run = load_result(ROOT / "results" / "training" / "cobra_paper.json")["runs"][0]
    released = load_result(
        ROOT / "results" / "training" / "cobra_released_eval.json"
    )["variants"]
    report = {
        "protocol": {
            "split": args.manifest,
            "seeds": SEEDS,
            "threshold": "selected on validation MCC for controlled runs",
            "primary_test": "same-seed paired t-test on four-benchmark average MCC",
            "multiplicity": "Holm adjustment across four benchmark tests",
        },
        "summaries": summaries,
        "primary_duet_vs_cobra": primary,
        "per_benchmark_duet_vs_cobra": per_split,
        "component_tests": component_tests,
        "hierarchical_bootstrap": hierarchical_bootstrap_delta(
            ensemble_runs,
            cobra_runs,
        ),
        "chain_bootstrap_fixed_seeds": chain_bootstrap_delta(
            ensemble_runs,
            cobra_runs,
        ),
        "fixed_threshold_0.5": fixed_threshold,
        "cobra_reproduction": {
            "paper_external_MCC": PAPER_MCC,
            "paper_average_MCC": PAPER_AVG,
            "retrained_seed42_external_MCC": {
                split: paper_run["test"][split]["MCC"] for split in EXTERNAL_TESTS
            },
            "retrained_seed42_external_average": float(
                np.mean([paper_run["test"][split]["MCC"] for split in EXTERNAL_TESTS])
            ),
            "retrained_seed42_internal_MCC": paper_run["test"]["internal_test"]["MCC"],
            "released_checkpoint": {
                name: {
                    "average_external_MCC": value["average_external_MCC"],
                    "per_split_MCC": {
                        split: value["metrics"][split]["MCC"]
                        for split in EXTERNAL_TESTS
                        if split in value["metrics"]
                    },
                }
                for name, value in released.items()
            },
        },
    }
    suffix = "exact" if prefix == "exact" else "p80"
    write_json(f"results/training/duet_vs_cobra_{suffix}.json", report)
    write_markdown(report, args.manifest, suffix)
    print(
        json.dumps(
            {
                "cobra_mean": summaries["CoBRA-M3"]["average"]["mean"],
                "duet_mean": summaries["DUET ensemble"]["average"]["mean"],
                "delta": primary["mean_difference"],
                "p": primary["p"],
                "ci95": primary["ci95"],
                "seeds_duet_better": primary["seeds_first_better"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def compute_at_05(run: dict, split: str) -> float:
    from duet_clean.comparison import pooled_from_per_chain

    metrics, _ = pooled_from_per_chain(run["per_chain"][split], 0.5)
    return metrics["MCC"]


def write_markdown(report: dict, manifest: str, suffix: str) -> None:
    lines = [
        f"# DUET versus CoBRA on the `{manifest}` split",
        "",
        "Generated by `scripts/12_compare_duet_cobra.py`.",
        "",
        "## Four-benchmark MCC",
        "",
        "| Method | Mean | SD | TE18 | TL12 | JL10 | RB9 |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for name, summary in report["summaries"].items():
        per = summary["per_split"]
        lines.append(
            f"| {name} | {summary['average']['mean']:.3f} | "
            f"{summary['average']['sd']:.3f} | "
            + " | ".join(f"{per[split]['mean']:.3f}" for split in EXTERNAL_TESTS)
            + " |"
        )
    primary = report["primary_duet_vs_cobra"]
    lines += [
        "",
        "## Paired significance",
        "",
        f"DUET minus CoBRA: **{primary['mean_difference']:+.3f} MCC** "
        f"(95% CI {primary['ci95'][0]:+.3f} to {primary['ci95'][1]:+.3f}), "
        f"paired t-test p = {primary['p']:.4g}; "
        f"Wilcoxon p = {primary['wilcoxon_p']:.4g}; "
        f"DUET better in {primary['seeds_first_better']}/{primary['n']} seeds.",
        f"At a fixed 0.5 threshold the average difference is "
        f"**{report['fixed_threshold_0.5']['average']['mean_difference']:+.3f} MCC** "
        f"(p = {report['fixed_threshold_0.5']['average']['p']:.4g}).",
        "Paired chain bootstrap (seeds fixed): "
        f"95% CI {report['chain_bootstrap_fixed_seeds']['ci95'][0]:+.3f} to "
        f"{report['chain_bootstrap_fixed_seeds']['ci95'][1]:+.3f}, "
        f"p = {report['chain_bootstrap_fixed_seeds']['p_two_sided']:.4g}.",
        "Seed-resampling hierarchical bootstrap (secondary, conservative): "
        f"95% CI {report['hierarchical_bootstrap']['ci95'][0]:+.3f} to "
        f"{report['hierarchical_bootstrap']['ci95'][1]:+.3f}, "
        f"p = {report['hierarchical_bootstrap']['p_two_sided']:.4g}.",
        "",
        "| Benchmark | DUET | CoBRA | Delta | p (raw) | p (Holm) |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for split in EXTERNAL_TESTS:
        duet = report["summaries"]["DUET ensemble"]["per_split"][split]["mean"]
        cobra = report["summaries"]["CoBRA-M3"]["per_split"][split]["mean"]
        row = report["per_benchmark_duet_vs_cobra"][split]
        lines.append(
            f"| {split} | {duet:.3f} | {cobra:.3f} | "
            f"{row['mean_difference']:+.3f} | {row['p']:.4g} | {row['p_holm']:.4g} |"
        )
    lines += [
        "",
        "## Component comparisons",
        "",
        "| Comparison | Delta | 95% CI | p | Better |",
        "|:--|--:|:--|--:|--:|",
    ]
    for name, result in report["component_tests"].items():
        lines.append(
            f"| {name} | {result['mean_difference']:+.3f} | "
            f"{result['ci95'][0]:+.3f} to {result['ci95'][1]:+.3f} | "
            f"{result['p']:.4g} | {result['seeds_first_better']}/{result['n']} |"
        )
    reproduction = report["cobra_reproduction"]
    lines += [
        "",
        "## CoBRA paper reproduction",
        "",
        "| Source | TE18 | TL12 | JL10 | RB9 | Average |",
        "|:--|--:|--:|--:|--:|--:|",
        "| Paper Table 3 | "
        + " | ".join(f"{reproduction['paper_external_MCC'][sf]:.3f}" for sf in EXTERNAL_TESTS)
        + f" | {reproduction['paper_average_MCC']:.3f} |",
        "| Released checkpoint, unmodified tests | "
        + " | ".join(
            f"{reproduction['released_checkpoint']['paper_released']['per_split_MCC'][sf]:.3f}"
            for sf in EXTERNAL_TESTS
        )
        + f" | {reproduction['released_checkpoint']['paper_released']['average_external_MCC']:.3f} |",
        "| Retrained seed 42 | "
        + " | ".join(
            f"{reproduction['retrained_seed42_external_MCC'][sf]:.3f}"
            for sf in EXTERNAL_TESTS
        )
        + f" | {reproduction['retrained_seed42_external_average']:.3f} |",
        "",
        "The retrained seed-42 internal-test MCC is "
        f"**{reproduction['retrained_seed42_internal_MCC']:.3f}**, "
        "against 0.629 in Table 2 of the CoBRA paper.",
        "",
    ]
    path = ROOT / "results" / "training" / f"DUET_VS_COBRA_{suffix.upper()}.md"
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
