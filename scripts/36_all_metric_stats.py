from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
from duet_clean.io import write_json
from duet_clean.large_stats import (
    EXTERNAL_SPLITS,
    FOUR_SPLITS,
    INDEPENDENT_EXTERNAL,
    SHARED_SPLITS,
    load_external,
    load_final,
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
    "graph_seq3d_ensemble",
    "graph_seqss3d_ensemble",
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
    "graph_seqss3d_vs_graph_seq3d_external": (
        "graph_seqss3d",
        "graph_seq3d",
    ),
}


def load_runs(methods: list[str], external: bool = False) -> dict[str, list[dict]]:
    loader = load_external if external else load_final
    return {
        method: [loader(method, seed) for seed in SEEDS]
        for method in methods
    }


def comparison_family(
    runs: dict[str, list[dict[str, Any]]],
    names: list[str],
    splits: list[str],
    fixed_threshold: bool = False,
) -> dict[str, dict[str, dict[str, Any]]]:
    output = {}
    for metric in METRICS:
        metric_rows = {}
        for name in names:
            first, second = PAIRS[name]
            metric_rows[name] = compare_metric_runs(
                runs[first],
                runs[second],
                splits,
                metric,
                fixed_threshold=fixed_threshold,
            )
        apply_metric_holm(metric_rows)
        output[metric] = metric_rows
    return output


def bootstrap_family(
    runs: dict[str, list[dict[str, Any]]],
    families: dict[str, dict[str, dict[str, Any]]],
    n_resamples: int,
) -> dict[str, dict[str, Any]]:
    output = {}
    fixed = {
        "primary_4split_fixed_0.5",
        "structure_3split_fixed_0.5",
    }
    for family_name, metric_rows in families.items():
        comparisons = list(next(iter(metric_rows.values())))
        for name in comparisons:
            first, second = PAIRS[name]
            splits = next(iter(metric_rows.values()))[name]["splits"]
            result = bootstrap_metric_compare(
                runs[first],
                runs[second],
                splits,
                n_resamples=n_resamples,
                seed=20261024,
                fixed_threshold=family_name in fixed,
            )
            for metric in METRICS:
                result["metrics"][metric]["comparison"] = name
                result["metrics"][metric]["family"] = family_name
            output[f"{family_name}/{name}"] = result
    apply_bootstrap_family_holm(output)
    return output


def context_summaries(
    runs: dict[str, list[dict[str, Any]]],
    fixed_threshold: bool = False,
) -> dict[str, Any]:
    output = {"four_split": {}, "three_split": {}, "external": {}}
    for method, method_runs in runs.items():
        splits = (
            FOUR_SPLITS
            if method in FOUR_SPLIT_METHODS
            else SHARED_SPLITS
        )
        output["four_split" if splits == FOUR_SPLITS else "three_split"][
            method
        ] = summarize_metrics(
            method_runs,
            splits,
            fixed_threshold=fixed_threshold,
        )
    return output


def write_markdown(report: dict[str, Any]) -> Path:
    lines = [
        "# All-metric significance analysis",
        "",
        "Each paired contrast uses ten paired training seeds. The independent "
        "resampling unit is the benchmark chain. Raw and Holm-adjusted p values "
        "are reported for both paired tests and chain bootstrap.",
        "",
        "## Method summaries",
        "",
        "| Context | Method | Metric | Mean | SD |",
        "|:--|:--|:--|--:|--:|",
    ]
    for context, methods in report["summaries"]["internal"].items():
        for method, summary in methods.items():
            for metric in METRICS:
                lines.append(
                    f"| {context} | {method} | {metric} | "
                    f"{summary[metric]['mean']:.4f} | "
                    f"{summary[metric]['sd']:.4f} |"
                )
    for method, summary in report["summaries"]["external"].items():
        for metric in METRICS:
            lines.append(
                f"| external_independent | {method} | {metric} | "
                f"{summary[metric]['mean']:.4f} | "
                f"{summary[metric]['sd']:.4f} |"
            )
    for method, summary in report["summaries"]["source_matched_control"].items():
        for metric in METRICS:
            lines.append(
                f"| source_matched_control | {method} | {metric} | "
                f"{summary[metric]['mean']:.4f} | "
                f"{summary[metric]['sd']:.4f} |"
            )

    lines += [
        "",
        "## Paired comparisons",
        "",
        "| Family | Metric | Comparison | Delta | 95% CI | raw p | Holm p |",
        "|:--|:--|:--|--:|:--|--:|--:|",
    ]
    for family, metric_rows in report["families"].items():
        for metric, rows in metric_rows.items():
            for name, value in rows.items():
                lines.append(
                    f"| {family} | {metric} | {name} | "
                    f"{value['mean_difference']:+.4f} | "
                    f"{value['ci95'][0]:+.4f} to {value['ci95'][1]:+.4f} | "
                    f"{value['p']:.4g} | {value['p_holm']:.4g} |"
                )

    lines += [
        "",
        "## Chain bootstrap",
        "",
        "| Family | Metric | Comparison | Delta | 95% CI | p | Holm p | valid B |",
        "|:--|:--|:--|--:|:--|--:|--:|--:|",
    ]
    for key, result in report["chain_bootstrap"].items():
        for metric, value in result["metrics"].items():
            lines.append(
                f"| {value['family']} | {metric} | {value['comparison']} | "
                f"{value['observed_delta']:+.4f} | "
                f"{value['ci95'][0]:+.4f} to {value['ci95'][1]:+.4f} | "
                f"{value['p_two_sided']:.4g} | "
                f"{value['p_bootstrap_holm']:.4g} | "
                f"{value['valid_resamples']} |"
            )
    lines += [""]
    path = ROOT / "results" / "followup" / "all_metric_stats" / "ALL_METRICS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-resamples", type=int, default=2000)
    args = parser.parse_args()
    internal_runs = load_runs(METHODS)
    external_runs = load_runs(
        [method for method in METHODS if method not in {
            "graph_seq3d_ensemble",
            "graph_seqss3d_ensemble",
        }],
        external=True,
    )
    for ensemble, members in {
        "graph_seq3d_ensemble": ["graph_seq3d", "duet_tcn", "duet_dense"],
        "graph_seqss3d_ensemble": ["graph_seqss3d", "duet_tcn", "duet_dense"],
    }.items():
        external_runs[ensemble] = [
            load_external(ensemble, seed) for seed in SEEDS
        ]

    family_names = {
        "primary_4split": [
            "duet_ensemble_vs_cobra",
            "duet_graph_vs_cobra",
            "duet_tcn_vs_cobra",
            "duet_dense_vs_cobra",
            "duet_ensemble_vs_duet_graph",
        ],
        "primary_4split_fixed_0.5": [
            "duet_ensemble_vs_cobra",
            "duet_graph_vs_cobra",
            "duet_tcn_vs_cobra",
            "duet_dense_vs_cobra",
            "duet_ensemble_vs_duet_graph",
        ],
        "structure_3split": [
            "grassp_ernie_vs_duet_ensemble",
            "grassp_rnafm_vs_duet_ensemble",
            "grassp_ernie_vs_graph_seqss3d",
            "graph_seq3d_vs_duet_graph",
            "graph_seqss3d_vs_graph_seq3d",
            "grassp_ernie_vs_cobra",
        ],
        "structure_3split_fixed_0.5": [
            "grassp_ernie_vs_duet_ensemble",
            "grassp_rnafm_vs_duet_ensemble",
            "grassp_ernie_vs_graph_seqss3d",
            "graph_seq3d_vs_duet_graph",
            "graph_seqss3d_vs_graph_seq3d",
            "grassp_ernie_vs_cobra",
        ],
        "crossed_controls_3split": [
            "grassp_ernie_vs_grassp_rnafm",
            "rgcn_ernie_vs_graph_seq3d",
            "grassp_rnafm_vs_graph_seq3d",
            "grassp_ernie_vs_rgcn_ernie",
        ],
        "external_independent_2split": [
            "duet_ensemble_vs_cobra",
            "grassp_ernie_vs_duet_ensemble",
            "grassp_rnafm_vs_duet_ensemble",
            "grassp_rnafm_vs_cobra",
            "graph_seqss3d_vs_graph_seq3d_external",
        ],
    }
    internal_families = {
        family: comparison_family(
            internal_runs,
            names,
            FOUR_SPLITS
            if family.startswith("primary")
            else SHARED_SPLITS,
            fixed_threshold=family.endswith("fixed_0.5"),
        )
        for family, names in family_names.items()
        if not family.startswith("external")
    }
    external_families = {
        "external_independent_2split": comparison_family(
            external_runs,
            family_names["external_independent_2split"],
            INDEPENDENT_EXTERNAL,
        )
    }
    families = {**internal_families, **external_families}
    bootstrap = {
        **bootstrap_family(
            internal_runs,
            internal_families,
            args.n_resamples,
        ),
        **bootstrap_family(
            external_runs,
            external_families,
            args.n_resamples,
        ),
    }
    report = {
        "protocol": {
            "seeds": SEEDS,
            "metrics": METRICS,
            "paired_unit": "training seed",
            "resampling_unit": "benchmark chain",
            "bootstrap_resamples": args.n_resamples,
            "holm_scope": "within comparison family and metric",
            "fixed_threshold_family": 0.5,
        },
        "summaries": {
            "internal": context_summaries(internal_runs),
            "external": {
                method: summarize_metrics(
                    runs,
                    INDEPENDENT_EXTERNAL,
                )
                for method, runs in external_runs.items()
            },
            "source_matched_control": {
                method: summarize_metrics(runs, ["hariboss_test"])
                for method, runs in external_runs.items()
            },
        },
        "families": families,
        "chain_bootstrap": bootstrap,
    }
    output = ROOT / "results" / "followup" / "all_metric_stats" / "all_metrics.json"
    write_json(output, report)
    markdown = write_markdown(report)
    print(output)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
