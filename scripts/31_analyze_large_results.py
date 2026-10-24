from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.large_stats import (
    EXTERNAL_SPLITS,
    FOUR_SPLITS,
    INDEPENDENT_EXTERNAL,
    SHARED_SPLITS,
    apply_holm,
    chain_bootstrap_compare,
    compare_runs,
    load_external,
    load_final,
    merge_ensemble,
    summarize,
    write_json_result,
)


SEEDS = [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]
METHODS = [
    "cobra",
    "duet_graph",
    "duet_tcn",
    "duet_dense",
    "duet_ensemble",
    "graph_seq3d",
    "graph_seqss3d",
    "grassp_ernie",
    "grassp_rnafm",
    "rgcn_ernie",
]
EXTERNAL_METHODS = [
    "cobra",
    "duet_graph",
    "duet_tcn",
    "duet_dense",
    "duet_ensemble",
    "graph_seq3d",
    "graph_seqss3d",
    "grassp_ernie",
    "grassp_rnafm",
    "rgcn_ernie",
]
PAIRS = {
    "duet_ensemble_vs_cobra": ("duet_ensemble", "cobra"),
    "duet_graph_vs_cobra": ("duet_graph", "cobra"),
    "duet_tcn_vs_cobra": ("duet_tcn", "cobra"),
    "duet_dense_vs_cobra": ("duet_dense", "cobra"),
    "duet_ensemble_vs_duet_graph": ("duet_ensemble", "duet_graph"),
    "grassp_ernie_vs_duet_ensemble": ("grassp_ernie", "duet_ensemble"),
    "grassp_rnafm_vs_duet_ensemble": ("grassp_rnafm", "duet_ensemble"),
    "grassp_ernie_vs_graph_seqss3d": ("grassp_ernie", "graph_seqss3d"),
    "graph_seq3d_vs_duet_graph": ("graph_seq3d", "duet_graph"),
    "graph_seqss3d_vs_graph_seq3d": ("graph_seqss3d", "graph_seq3d"),
    "grassp_ernie_vs_cobra": ("grassp_ernie", "cobra"),
    "grassp_ernie_vs_grassp_rnafm": ("grassp_ernie", "grassp_rnafm"),
    "rgcn_ernie_vs_graph_seq3d": ("rgcn_ernie", "graph_seq3d"),
    "grassp_rnafm_vs_graph_seq3d": ("grassp_rnafm", "graph_seq3d"),
    "grassp_ernie_vs_rgcn_ernie": ("grassp_ernie", "rgcn_ernie"),
    "grassp_rnafm_vs_cobra": ("grassp_rnafm", "cobra"),
}


def add_ensemble_compare(
    runs: dict[str, list[dict[str, Any]]],
    comparisons: dict[str, dict[str, Any]],
    name: str,
    first: str,
    second: str,
    splits: list[str],
    metric_key: str = "test",
) -> None:
    comparisons[name] = compare_runs(
        runs[first],
        runs[second],
        splits,
        metric_key=metric_key,
    )


def load_internal() -> dict[str, list[dict[str, Any]]]:
    runs = {}
    for method in METHODS:
        paths = [
            ROOT
            / "results"
            / "large_protocol"
            / "final"
            / method
            / f"seed{seed}.json"
            for seed in SEEDS
        ]
        missing = [path.name for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{method}: missing {missing[:3]}")
        runs[method] = [load_final(method, seed) for seed in SEEDS]
    return runs


def build_external_ensemble(seed: int) -> dict[str, Any]:
    selected_path = (
        ROOT
        / "results"
        / "large_protocol"
        / "cv"
        / "duet_ensemble"
        / "selected.json"
    )
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    members = [
        load_external(method, seed)
        for method in ["duet_graph", "duet_tcn", "duet_dense"]
    ]
    ensemble = merge_ensemble(members, float(selected["threshold"]))
    ensemble["method"] = "duet_ensemble"
    path = (
        ROOT
        / "results"
        / "large_protocol"
        / "external"
        / f"duet_ensemble_seed{seed}.json"
    )
    path.write_text(
        json.dumps(ensemble, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ensemble


def external_results() -> dict[str, list[dict[str, Any]]]:
    runs = {
        method: [load_external(method, seed) for seed in SEEDS]
        for method in EXTERNAL_METHODS
        if method != "duet_ensemble"
    }
    runs["duet_ensemble"] = [build_external_ensemble(seed) for seed in SEEDS]
    return runs


def bootstrap_family(
    runs: dict[str, list[dict[str, Any]]],
    families: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    output = {}
    fixed = {"primary_4split_fixed_0.5", "structure_3split_fixed_0.5"}
    for family, comparisons in families.items():
        for name in comparisons:
            first, second = PAIRS[name]
            output[f"{family}/{name}"] = chain_bootstrap_compare(
                runs[first],
                runs[second],
                comparisons[name]["splits"],
                n_resamples=2000,
                seed=20261024,
                fixed_threshold=family in fixed,
            )
    return output


def per_benchmark_family(
    runs: dict[str, list[dict[str, Any]]],
    comparisons: dict[str, dict[str, Any]],
    metric_key: str = "test",
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped = {}
    flat = {}
    for name, average in comparisons.items():
        first, second = PAIRS[name]
        grouped[name] = {}
        for split in average["splits"]:
            comparison = compare_runs(
                runs[first],
                runs[second],
                [split],
                metric_key=metric_key,
            )
            grouped[name][split] = comparison
            flat[f"{name}/{split}"] = comparison
    apply_holm(flat)
    for name in grouped:
        for split in grouped[name]:
            grouped[name][split]["p_holm"] = flat[f"{name}/{split}"]["p_holm"]
    return grouped


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Large-data protocol results",
        "",
        "Generated by `scripts/31_analyze_large_results.py`.",
        "",
        "The independence unit is the benchmark chain. Training seeds are paired "
        "blocks for same-seed inference; chain bootstrap intervals resample "
        "benchmark chains while keeping all ten seeds paired.",
        "",
        "## Internal test MCC",
        "",
        "| Method | Mean | SD | TE18 | TL12 | JL10 | RB9 |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for method, summary in report["internal"]["summaries"].items():
        per = summary["per_split"]
        lines.append(
            f"| {method} | {summary['average']['mean']:.3f} | "
            f"{summary['average']['sd']:.3f} | "
            + " | ".join(
                f"{per[split]['mean']:.3f}"
                if split in per
                else "-"
                for split in FOUR_SPLITS
            )
            + " |"
        )
    lines += [
        "",
        "## Same-seed comparisons",
        "",
        "| Family | Comparison | Delta | 95% CI | p | Holm p | Seeds better |",
        "|:--|:--|--:|:--|--:|--:|--:|",
    ]
    for family, comparisons in report["internal"]["families"].items():
        for name, value in comparisons.items():
            lines.append(
                f"| {family} | {name} | {value['mean_difference']:+.3f} | "
                f"{value['ci95'][0]:+.3f} to {value['ci95'][1]:+.3f} | "
                f"{value['p']:.4g} | {value['p_holm']:.4g} | "
                f"{value['seeds_first_better']}/{value['n']} |"
            )
    lines += [
        "",
        "## Chain bootstrap",
        "",
        "| Comparison | Splits | Delta | 95% CI | p |",
        "|:--|:--|--:|:--|--:|",
    ]
    for name, value in report["internal"]["chain_bootstrap"].items():
        lines.append(
            f"| {name} | {', '.join(value['splits'])} | "
            f"{value['observed_delta']:+.3f} | "
            f"{value['ci95'][0]:+.3f} to {value['ci95'][1]:+.3f} | "
            f"{value['p_two_sided']:.4g} |"
        )
    lines += [
        "",
        "## Per-benchmark tests",
        "",
        "Holm adjustment for these rows is applied across every comparison and "
        "benchmark within the same family.",
        "",
        "| Family | Comparison | Benchmark | Delta | raw p | Holm p |",
        "|:--|:--|:--|--:|--:|--:|",
    ]
    for family, methods in report["internal"]["per_benchmark"].items():
        for name, splits in methods.items():
            for split, value in splits.items():
                lines.append(
                    f"| {family} | {name} | {split} | "
                    f"{value['mean_difference']:+.3f} | "
                    f"{value['p']:.4g} | {value['p_holm']:.4g} |"
                )
    lines += [
        "",
        "## SMARTPocket external validation",
        "",
        "`external_jiang` and `time_dependent` are independent external sets. "
        "`hariboss_test` shares the training source and is retained only as a "
        "source-matched control.",
        "",
        "| Method | Jiang MCC | Time-dependent MCC | Independent mean |",
        "|:--|--:|--:|--:|",
    ]
    for method, summary in report["external"]["summaries"].items():
        lines.append(
            f"| {method} | {summary['per_split']['external_jiang']['mean']:.3f} | "
            f"{summary['per_split']['time_dependent']['mean']:.3f} | "
            f"{summary['average']['mean']:.3f} |"
        )
    lines += [
        "",
        "| External comparison | Delta | 95% CI | p | Holm p |",
        "|:--|--:|:--|--:|--:|",
    ]
    for name, value in report["external"]["comparisons"].items():
        lines.append(
            f"| {name} | {value['mean_difference']:+.3f} | "
            f"{value['ci95'][0]:+.3f} to {value['ci95'][1]:+.3f} | "
            f"{value['p']:.4g} | {value['p_holm']:.4g} |"
        )
    lines += [
        "",
        "### External chain bootstrap",
        "",
        "| Comparison | Delta | 95% CI | p |",
        "|:--|--:|:--|--:|",
    ]
    for name, value in report["external"]["chain_bootstrap"].items():
        lines.append(
            f"| {name} | {value['observed_delta']:+.3f} | "
            f"{value['ci95'][0]:+.3f} to {value['ci95'][1]:+.3f} | "
            f"{value['p_two_sided']:.4g} |"
        )
    lines += [
        "",
        "| External comparison | Split | Delta | raw p | Holm p |",
        "|:--|:--|--:|--:|--:|",
    ]
    for name, splits in report["external"]["per_benchmark"].items():
        for split, value in splits.items():
            lines.append(
                f"| {name} | {split} | {value['mean_difference']:+.3f} | "
                f"{value['p']:.4g} | {value['p_holm']:.4g} |"
            )
    lines += [""]
    path = ROOT / "results" / "large_protocol" / "LARGE_PROTOCOL_RESULTS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    internal_runs = load_internal()
    summaries = {
        method: summarize(
            runs,
            FOUR_SPLITS if "RB9" in runs[0]["test"] else SHARED_SPLITS,
        )
        for method, runs in internal_runs.items()
    }
    primary = {}
    add_ensemble_compare(
        internal_runs,
        primary,
        "duet_ensemble_vs_cobra",
        "duet_ensemble",
        "cobra",
        FOUR_SPLITS,
    )
    add_ensemble_compare(
        internal_runs,
        primary,
        "duet_graph_vs_cobra",
        "duet_graph",
        "cobra",
        FOUR_SPLITS,
    )
    add_ensemble_compare(
        internal_runs,
        primary,
        "duet_tcn_vs_cobra",
        "duet_tcn",
        "cobra",
        FOUR_SPLITS,
    )
    add_ensemble_compare(
        internal_runs,
        primary,
        "duet_dense_vs_cobra",
        "duet_dense",
        "cobra",
        FOUR_SPLITS,
    )
    add_ensemble_compare(
        internal_runs,
        primary,
        "duet_ensemble_vs_duet_graph",
        "duet_ensemble",
        "duet_graph",
        FOUR_SPLITS,
    )
    apply_holm(primary)

    primary_fixed = {}
    for name, first, second in [
        ("duet_ensemble_vs_cobra", "duet_ensemble", "cobra"),
        ("duet_graph_vs_cobra", "duet_graph", "cobra"),
        ("duet_tcn_vs_cobra", "duet_tcn", "cobra"),
        ("duet_dense_vs_cobra", "duet_dense", "cobra"),
        ("duet_ensemble_vs_duet_graph", "duet_ensemble", "duet_graph"),
    ]:
        add_ensemble_compare(
            internal_runs,
            primary_fixed,
            name,
            first,
            second,
            FOUR_SPLITS,
            metric_key="test_at_0.5",
        )
    apply_holm(primary_fixed)

    structure = {}
    add_ensemble_compare(
        internal_runs,
        structure,
        "grassp_ernie_vs_duet_ensemble",
        "grassp_ernie",
        "duet_ensemble",
        SHARED_SPLITS,
    )
    add_ensemble_compare(
        internal_runs,
        structure,
        "grassp_rnafm_vs_duet_ensemble",
        "grassp_rnafm",
        "duet_ensemble",
        SHARED_SPLITS,
    )
    add_ensemble_compare(
        internal_runs,
        structure,
        "grassp_ernie_vs_graph_seqss3d",
        "grassp_ernie",
        "graph_seqss3d",
        SHARED_SPLITS,
    )
    add_ensemble_compare(
        internal_runs,
        structure,
        "graph_seq3d_vs_duet_graph",
        "graph_seq3d",
        "duet_graph",
        SHARED_SPLITS,
    )
    add_ensemble_compare(
        internal_runs,
        structure,
        "graph_seqss3d_vs_graph_seq3d",
        "graph_seqss3d",
        "graph_seq3d",
        SHARED_SPLITS,
    )
    add_ensemble_compare(
        internal_runs,
        structure,
        "grassp_ernie_vs_cobra",
        "grassp_ernie",
        "cobra",
        SHARED_SPLITS,
    )
    apply_holm(structure)

    structure_fixed = {}
    for name, first, second in [
        ("grassp_ernie_vs_duet_ensemble", "grassp_ernie", "duet_ensemble"),
        ("grassp_rnafm_vs_duet_ensemble", "grassp_rnafm", "duet_ensemble"),
        ("grassp_ernie_vs_graph_seqss3d", "grassp_ernie", "graph_seqss3d"),
        ("graph_seq3d_vs_duet_graph", "graph_seq3d", "duet_graph"),
        ("graph_seqss3d_vs_graph_seq3d", "graph_seqss3d", "graph_seq3d"),
        ("grassp_ernie_vs_cobra", "grassp_ernie", "cobra"),
    ]:
        add_ensemble_compare(
            internal_runs,
            structure_fixed,
            name,
            first,
            second,
            SHARED_SPLITS,
            metric_key="test_at_0.5",
        )
    apply_holm(structure_fixed)

    crossed = {}
    add_ensemble_compare(
        internal_runs,
        crossed,
        "grassp_ernie_vs_grassp_rnafm",
        "grassp_ernie",
        "grassp_rnafm",
        SHARED_SPLITS,
    )
    add_ensemble_compare(
        internal_runs,
        crossed,
        "rgcn_ernie_vs_graph_seq3d",
        "rgcn_ernie",
        "graph_seq3d",
        SHARED_SPLITS,
    )
    add_ensemble_compare(
        internal_runs,
        crossed,
        "grassp_rnafm_vs_graph_seq3d",
        "grassp_rnafm",
        "graph_seq3d",
        SHARED_SPLITS,
    )
    add_ensemble_compare(
        internal_runs,
        crossed,
        "grassp_ernie_vs_rgcn_ernie",
        "grassp_ernie",
        "rgcn_ernie",
        SHARED_SPLITS,
    )
    apply_holm(crossed)

    bootstrap = bootstrap_family(
        internal_runs,
        {
            "primary_4split": primary,
            "primary_4split_fixed_0.5": primary_fixed,
            "structure_3split": structure,
            "structure_3split_fixed_0.5": structure_fixed,
            "crossed_controls_3split": crossed,
        },
    )

    external_runs = external_results()
    external_summaries = {
        method: summarize(runs, INDEPENDENT_EXTERNAL)
        for method, runs in external_runs.items()
    }
    external_comparisons = {}
    add_ensemble_compare(
        external_runs,
        external_comparisons,
        "duet_ensemble_vs_cobra",
        "duet_ensemble",
        "cobra",
        INDEPENDENT_EXTERNAL,
    )
    add_ensemble_compare(
        external_runs,
        external_comparisons,
        "grassp_ernie_vs_duet_ensemble",
        "grassp_ernie",
        "duet_ensemble",
        INDEPENDENT_EXTERNAL,
    )
    add_ensemble_compare(
        external_runs,
        external_comparisons,
        "grassp_rnafm_vs_duet_ensemble",
        "grassp_rnafm",
        "duet_ensemble",
        INDEPENDENT_EXTERNAL,
    )
    add_ensemble_compare(
        external_runs,
        external_comparisons,
        "grassp_rnafm_vs_cobra",
        "grassp_rnafm",
        "cobra",
        INDEPENDENT_EXTERNAL,
    )
    add_ensemble_compare(
        external_runs,
        external_comparisons,
        "graph_seqss3d_vs_graph_seq3d",
        "graph_seqss3d",
        "graph_seq3d",
        INDEPENDENT_EXTERNAL,
    )
    apply_holm(external_comparisons)
    external_bootstrap = bootstrap_family(
        external_runs,
        {"external_independent_2split": external_comparisons},
    )

    report = {
        "protocol": {
            "manifest": "large_p80_cv5",
            "seeds": SEEDS,
            "cv_seed": 42,
            "paired_unit": "same training seed",
            "resampling_unit": "benchmark chain",
            "holm_families": {
                "primary_4split": list(primary),
                "primary_4split_fixed_0.5": list(primary_fixed),
                "structure_3split": list(structure),
                "structure_3split_fixed_0.5": list(structure_fixed),
                "crossed_controls_3split": list(crossed),
                "external_independent_2split": list(external_comparisons),
            },
        },
        "internal": {
            "summaries": summaries,
            "families": {
                "primary_4split": primary,
                "primary_4split_fixed_0.5": primary_fixed,
                "structure_3split": structure,
                "structure_3split_fixed_0.5": structure_fixed,
                "crossed_controls_3split": crossed,
            },
            "chain_bootstrap": bootstrap,
            "per_benchmark": {
                "primary_4split": per_benchmark_family(
                    internal_runs,
                    primary,
                ),
                "primary_4split_fixed_0.5": per_benchmark_family(
                    internal_runs,
                    primary_fixed,
                    metric_key="test_at_0.5",
                ),
                "structure_3split": per_benchmark_family(
                    internal_runs,
                    structure,
                ),
                "structure_3split_fixed_0.5": per_benchmark_family(
                    internal_runs,
                    structure_fixed,
                    metric_key="test_at_0.5",
                ),
                "crossed_controls_3split": per_benchmark_family(
                    internal_runs,
                    crossed,
                ),
            },
        },
        "external": {
            "independent_splits": INDEPENDENT_EXTERNAL,
            "control_split": ["hariboss_test"],
            "summaries": external_summaries,
            "comparisons": external_comparisons,
            "chain_bootstrap": external_bootstrap,
            "per_benchmark": per_benchmark_family(
                external_runs,
                external_comparisons,
            ),
        },
    }
    write_json_result("results/large_protocol/analysis.json", report)
    write_markdown(report)
    print(
        json.dumps(
            {
                "primary": {
                    key: {
                        "delta": value["mean_difference"],
                        "p": value["p"],
                        "p_holm": value["p_holm"],
                    }
                    for key, value in primary.items()
                },
                "crossed": {
                    key: {
                        "delta": value["mean_difference"],
                        "p": value["p"],
                        "p_holm": value["p_holm"],
                    }
                    for key, value in crossed.items()
                },
                "external": {
                    key: {
                        "delta": value["mean_difference"],
                        "p": value["p"],
                        "p_holm": value["p_holm"],
                    }
                    for key, value in external_comparisons.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
