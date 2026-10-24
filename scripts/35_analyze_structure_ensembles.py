from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.large_stats import (
    INDEPENDENT_EXTERNAL,
    compare_runs,
    load_external,
    load_final,
    summarize,
)
from duet_clean.followup_stats import METRICS, summarize_metrics


SEEDS = [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]
SPLITS = ["TE18", "TL12", "JL10"]
METHODS = [
    "cobra",
    "duet_graph",
    "duet_ensemble",
    "graph_seq3d",
    "graph_seqss3d",
    "graph_seq3d_ensemble",
    "graph_seqss3d_ensemble",
    "grassp_rnafm",
]
PAIRS = [
    ("graph_seq3d_ensemble_vs_cobra", "graph_seq3d_ensemble", "cobra"),
    ("graph_seq3d_ensemble_vs_duet_ensemble", "graph_seq3d_ensemble", "duet_ensemble"),
    ("graph_seq3d_ensemble_vs_graph_seq3d", "graph_seq3d_ensemble", "graph_seq3d"),
    ("graph_seq3d_ensemble_vs_grassp_rnafm", "graph_seq3d_ensemble", "grassp_rnafm"),
    ("graph_seqss3d_ensemble_vs_cobra", "graph_seqss3d_ensemble", "cobra"),
    ("graph_seqss3d_ensemble_vs_duet_ensemble", "graph_seqss3d_ensemble", "duet_ensemble"),
    ("graph_seq3d_vs_cobra", "graph_seq3d", "cobra"),
    ("graph_seqss3d_vs_cobra", "graph_seqss3d", "cobra"),
]


def main() -> int:
    runs = {
        method: [load_final(method, seed) for seed in SEEDS]
        for method in METHODS
    }
    report = {
        "splits": SPLITS,
        "summaries": {
            method: summarize(runs[method], SPLITS) for method in METHODS
        },
        "metric_summaries": {
            method: summarize_metrics(runs[method], SPLITS)
            for method in METHODS
        },
        "comparisons": {
            name: compare_runs(runs[first], runs[second], SPLITS)
            for name, first, second in PAIRS
        },
    }
    external_runs = {
        method: [load_external(method, seed) for seed in SEEDS]
        for method in METHODS
    }
    report["external_summaries"] = {
        method: summarize(external_runs[method], INDEPENDENT_EXTERNAL)
        for method in METHODS
    }
    report["external_comparisons"] = {
        "graph_seq3d_ensemble_vs_cobra": compare_runs(
            external_runs["graph_seq3d_ensemble"],
            external_runs["cobra"],
            INDEPENDENT_EXTERNAL,
        ),
        "graph_seq3d_ensemble_vs_duet_ensemble": compare_runs(
            external_runs["graph_seq3d_ensemble"],
            external_runs["duet_ensemble"],
            INDEPENDENT_EXTERNAL,
        ),
        "graph_seq3d_ensemble_vs_grassp_rnafm": compare_runs(
            external_runs["graph_seq3d_ensemble"],
            external_runs["grassp_rnafm"],
            INDEPENDENT_EXTERNAL,
        ),
    }
    output = ROOT / "results" / "large_protocol" / "structure_ensembles.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Large-data structure ensembles",
        "",
        "| Method | Mean MCC | SD |",
        "|:--|--:|--:|",
    ]
    for method, item in report["summaries"].items():
        lines.append(
            f"| {method} | {item['average']['mean']:.3f} | "
            f"{item['average']['sd']:.3f} |"
        )
    lines += [
        "",
        "| Comparison | Delta | 95% CI | p | Seeds better |",
        "|:--|--:|:--|--:|--:|",
    ]
    for name, item in report["comparisons"].items():
        lines.append(
            f"| {name} | {item['mean_difference']:+.3f} | "
            f"{item['ci95'][0]:+.3f} to {item['ci95'][1]:+.3f} | "
            f"{item['p']:.4g} | {item['seeds_first_better']}/{item['n']} |"
        )
    lines += [
        "",
        "## External validation",
        "",
        "| Method | Jiang | Time-dependent | Independent mean |",
        "|:--|--:|--:|--:|",
    ]
    for method, item in report["external_summaries"].items():
        lines.append(
            f"| {method} | {item['per_split']['external_jiang']['mean']:.3f} | "
            f"{item['per_split']['time_dependent']['mean']:.3f} | "
            f"{item['average']['mean']:.3f} |"
        )
    lines += [
        "",
        "| External comparison | Delta | 95% CI | p | Seeds better |",
        "|:--|--:|:--|--:|--:|",
    ]
    for name, item in report["external_comparisons"].items():
        lines.append(
            f"| {name} | {item['mean_difference']:+.3f} | "
            f"{item['ci95'][0]:+.3f} to {item['ci95'][1]:+.3f} | "
            f"{item['p']:.4g} | {item['seeds_first_better']}/{item['n']} |"
        )
    lines += [
        "",
        "## All-metric internal summaries",
        "",
        "| Method | Metric | Mean | SD |",
        "|:--|:--|--:|--:|",
    ]
    for method in METHODS:
        summary = report["metric_summaries"][method]
        for metric in METRICS:
            lines.append(
                f"| {method} | {metric} | {summary[metric]['mean']:.4f} | "
                f"{summary[metric]['sd']:.4f} |"
            )
    (ROOT / "results" / "large_protocol" / "STRUCTURE_ENSEMBLES.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
