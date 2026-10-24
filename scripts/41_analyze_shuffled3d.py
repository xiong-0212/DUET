from __future__ import annotations

import json
import shutil
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
METHODS = [
    "cobra",
    "duet_graph",
    "graph_seq3d",
    "graph_seq3d_shuffle",
    "graph_seqss3d",
]
PAIRS = {
    "shuffled_vs_real3d": ("graph_seq3d_shuffle", "graph_seq3d"),
    "shuffled_vs_seqss3d": ("graph_seq3d_shuffle", "graph_seqss3d"),
    "shuffled_vs_seqss": ("graph_seq3d_shuffle", "duet_graph"),
    "shuffled_vs_cobra": ("graph_seq3d_shuffle", "cobra"),
    "real3d_vs_seqss": ("graph_seq3d", "duet_graph"),
    "real3d_vs_seqss3d": ("graph_seq3d", "graph_seqss3d"),
}


def write_markdown(report: dict) -> Path:
    lines = [
        "# Large-data shuffled-3D control",
        "",
        "The shuffle preserves every node's structural out-degree, sequence "
        "edges, node features, labels and folds. Three small graphs required "
        "duplicate targets because their degree exceeded the number of "
        "non-self candidates; this is recorded in `graph_audit.json`.",
        "",
        "## Method summaries",
        "",
        "| Method | Metric | Mean | SD |",
        "|:--|:--|--:|--:|",
    ]
    for method, summary in report["summaries"].items():
        for metric in METRICS:
            lines.append(
                f"| {method} | {metric} | "
                f"{summary[metric]['mean']:.4f} | "
                f"{summary[metric]['sd']:.4f} |"
            )
    lines += [
        "",
        "## Paired comparisons",
        "",
        "| Metric | Comparison | Delta | 95% CI | raw p | Holm p |",
        "|:--|:--|--:|:--|--:|--:|",
    ]
    for metric, rows in report["comparisons"].items():
        for name, value in rows.items():
            lines.append(
                f"| {metric} | {name} | {value['mean_difference']:+.4f} | "
                f"{value['ci95'][0]:+.4f} to {value['ci95'][1]:+.4f} | "
                f"{value['p']:.4g} | {value['p_holm']:.4g} |"
            )
    lines += [
        "",
        "## Chain bootstrap",
        "",
        "| Metric | Comparison | Delta | 95% CI | raw p | Holm p | valid B |",
        "|:--|:--|--:|:--|--:|--:|--:|",
    ]
    for name, result in report["chain_bootstrap"].items():
        for metric, value in result["metrics"].items():
            lines.append(
                f"| {metric} | {name} | {value['observed_delta']:+.4f} | "
                f"{value['ci95'][0]:+.4f} to {value['ci95'][1]:+.4f} | "
                f"{value['p_two_sided']:.4g} | "
                f"{value['p_bootstrap_holm']:.4g} | "
                f"{value['valid_resamples']} |"
            )
    path = ROOT / "results" / "followup" / "shuffled3d" / "SHUFFLED3D.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    runs = {
        method: [load_final(method, seed) for seed in SEEDS]
        for method in METHODS
    }
    output_root = ROOT / "results" / "followup" / "shuffled3d"
    output_root.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        method_dir = output_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            source = (
                ROOT
                / "results"
                / "large_protocol"
                / "final"
                / method
                / f"seed{seed}.json"
            )
            shutil.copy2(source, method_dir / f"seed{seed}.json")

    comparisons = {}
    for metric in METRICS:
        rows = {
            name: compare_metric_runs(
                runs[first],
                runs[second],
                SHARED_SPLITS,
                metric,
            )
            for name, (first, second) in PAIRS.items()
        }
        apply_metric_holm(rows)
        comparisons[metric] = rows

    bootstrap = {}
    for name, (first, second) in PAIRS.items():
        result = bootstrap_metric_compare(
            runs[first],
            runs[second],
            SHARED_SPLITS,
            n_resamples=2000,
            seed=20261024,
        )
        bootstrap[name] = result
    apply_bootstrap_family_holm(bootstrap)
    report = {
        "protocol": {
            "seeds": SEEDS,
            "splits": SHARED_SPLITS,
            "bootstrap_resamples": 2000,
            "bootstrap_seed": 20261024,
            "holm_scope": "within comparison family and metric",
        },
        "summaries": {
            method: summarize_metrics(runs[method], SHARED_SPLITS)
            for method in METHODS
        },
        "comparisons": comparisons,
        "chain_bootstrap": bootstrap,
    }
    path = output_root / "analysis.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = write_markdown(report)
    print(path)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
