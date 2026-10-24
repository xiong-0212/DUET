from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

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
SIZES = [80, 160, 290]
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
COLORS = {
    "cobra": "#7b7b7b",
    "duet_graph": "#4c78a8",
    "duet_tcn": "#f58518",
    "duet_dense": "#54a24b",
    "duet_ensemble": "#b279a2",
    "graph_seq3d": "#e45756",
    "graph_seq3d_ensemble": "#9d755d",
    "grassp_rnafm": "#2f4b7c",
}
DISPLAY = {
    "cobra": "CoBRA-M3",
    "duet_graph": "DUET graph",
    "duet_tcn": "DUET TCN",
    "duet_dense": "DUET DenseNet",
    "duet_ensemble": "DUET ensemble",
    "graph_seq3d": "seq+3D graph",
    "graph_seq3d_ensemble": "seq+3D ensemble",
    "grassp_rnafm": "GRASSP + RNA-FM",
}


def load_result(size: int, method: str, seed: int) -> dict:
    if size == 290:
        path = (
            ROOT
            / "results"
            / "large_protocol"
            / "final"
            / method
            / f"seed{seed}.json"
        )
    else:
        path = (
            ROOT
            / "results"
            / "followup"
            / "size_curve"
            / f"size_{size}"
            / "final"
            / method
            / f"seed{seed}.json"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def plot_curves(report: dict) -> Path:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from audit_figure_collisions import audit_pdf, exit_code
    from audit_panel_alignment import (
        require_matplotlib_panel_alignment,
        write_json_report,
    )

    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(7.2, 4.1),
        sharex=True,
        constrained_layout=False,
    )
    axes = axes.ravel()
    for metric_index, metric in enumerate(METRICS):
        ax = axes[metric_index]
        for method in METHODS:
            means = []
            spreads = []
            for size in SIZES:
                summary = report["summaries"][str(size)][method][metric]
                means.append(summary["mean"])
                spreads.append(summary["sd"])
            means = np.asarray(means)
            spreads = np.asarray(spreads)
            ax.plot(
                SIZES,
                means,
                color=COLORS[method],
                lw=1.3,
                marker="o",
                ms=2.8,
                label=DISPLAY[method],
            )
            ax.fill_between(
                SIZES,
                means - spreads,
                means + spreads,
                color=COLORS[method],
                alpha=0.08,
                linewidth=0,
            )
        ax.set_title(metric, fontsize=7.5, fontweight="bold", pad=3)
        ax.set_xticks(SIZES)
        ax.set_xticklabels([str(size) for size in SIZES])
        ax.set_xlabel("Training chains", fontsize=6.5, labelpad=2)
        ax.tick_params(labelsize=6.5, width=0.7, length=2.5)
        ax.grid(axis="y", color="#d9d9d9", lw=0.45, alpha=0.7)
        ax.text(
            -0.12,
            1.08,
            chr(ord("a") + metric_index),
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=6.4,
        columnspacing=1.0,
        handlelength=1.8,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.95,
        bottom=0.16,
        wspace=0.32,
        hspace=0.32,
    )
    output = (
        ROOT
        / "results"
        / "followup"
        / "size_curve"
        / "SIZE_CURVE"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    alignment_report = require_matplotlib_panel_alignment(
        fig,
        axes=list(axes[:-1]),
        panel_ids=[chr(ord("a") + index) for index in range(7)],
        row_groups=[
            ["a", "b", "c", "d"],
            ["e", "f", "g"],
        ],
        column_groups=[
            ["a", "e"],
            ["b", "f"],
            ["c", "g"],
        ],
        json_out=str(output) + ".alignment-layout.json",
        overlay_svg=str(output) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    write_json_report(
        alignment_report,
        str(output) + ".alignment.json",
    )
    fig.savefig(str(output) + ".svg", bbox_inches="tight")
    fig.savefig(str(output) + ".pdf", bbox_inches="tight")
    fig.savefig(str(output) + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(str(output) + ".tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    collision_report = audit_pdf(Path(str(output) + ".pdf"))
    collision_path = Path(str(output) + ".collision-audit.json")
    collision_path.write_text(
        json.dumps(collision_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    collision_code = exit_code(collision_report, strict=False)
    if collision_code:
        raise RuntimeError(
            f"collision audit failed with exit code {collision_code}"
        )
    return output.with_suffix(".pdf")


def write_markdown(report: dict) -> Path:
    lines = [
        "# Fixed-composition training-size curve",
        "",
        "All rows use the same canonical 44-chain test set. Sizes 80 and 160 "
        "are nested whole-cluster subsets of the 290-chain pool; 290 reuses "
        "the accepted large-data results.",
        "",
        "## Method summaries",
        "",
        "| Size | Method | Metric | Mean | SD |",
        "|--:|:--|:--|--:|--:|",
    ]
    for size in SIZES:
        for method in METHODS:
            for metric in METRICS:
                value = report["summaries"][str(size)][method][metric]
                lines.append(
                    f"| {size} | {method} | {metric} | "
                    f"{value['mean']:.4f} | {value['sd']:.4f} |"
                )
    lines += [
        "",
        "## Comparisons against 80 chains",
        "",
        "| Metric | Size | Method | Delta | 95% CI | raw p | Holm p |",
        "|:--|--:|:--|--:|:--|--:|--:|",
    ]
    for metric in METRICS:
        for key, value in report["comparisons"][metric].items():
            size, method = key.split("/", 1)
            lines.append(
                f"| {metric} | {size} | {method} | "
                f"{value['mean_difference']:+.4f} | "
                f"{value['ci95'][0]:+.4f} to {value['ci95'][1]:+.4f} | "
                f"{value['p']:.4g} | {value['p_holm']:.4g} |"
            )
    lines += [
        "",
        "## Chain bootstrap",
        "",
        "| Metric | Comparison | Delta | 95% CI | raw p | Holm p |",
        "|:--|:--|--:|:--|--:|--:|",
    ]
    for key, result in report["chain_bootstrap"].items():
        for metric, value in result["metrics"].items():
            lines.append(
                f"| {metric} | {key} | {value['observed_delta']:+.4f} | "
                f"{value['ci95'][0]:+.4f} to {value['ci95'][1]:+.4f} | "
                f"{value['p_two_sided']:.4g} | "
                f"{value['p_bootstrap_holm']:.4g} |"
            )
    lines += [
        "",
        "Figure: `SIZE_CURVE.pdf` / `.svg` / `.png` / `.tiff`. "
        "Curves show mean across ten seeds; bands show one seed SD.",
        "",
    ]
    path = ROOT / "results" / "followup" / "size_curve" / "SIZE_CURVE.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        output = ROOT / "results" / "followup" / "size_curve" / "size_curve.json"
        report = json.loads(output.read_text(encoding="utf-8"))
        figure = plot_curves(report)
        print(figure)
        return 0
    runs = {
        size: {
            method: [load_result(size, method, seed) for seed in SEEDS]
            for method in METHODS
        }
        for size in SIZES
    }
    summaries = {
        str(size): {
            method: summarize_metrics(
                runs[size][method],
                FOUR_SPLITS if method in FOUR_SPLIT_METHODS else SHARED_SPLITS,
            )
            for method in METHODS
        }
        for size in SIZES
    }
    comparisons = {}
    comparison_objects = {}
    for metric in METRICS:
        rows = {}
        for size in [160, 290]:
            for method in METHODS:
                splits = (
                    FOUR_SPLITS
                    if method in FOUR_SPLIT_METHODS
                    else SHARED_SPLITS
                )
                rows[f"{size}/{method}"] = compare_metric_runs(
                    runs[size][method],
                    runs[80][method],
                    splits,
                    metric,
                )
        apply_metric_holm(rows)
        comparisons[metric] = rows
        comparison_objects[metric] = rows

    bootstrap = {}
    for size in [160, 290]:
        for method in METHODS:
            key = f"{size}/{method}"
            splits = (
                FOUR_SPLITS
                if method in FOUR_SPLIT_METHODS
                else SHARED_SPLITS
            )
            result = bootstrap_metric_compare(
                runs[size][method],
                runs[80][method],
                splits,
                n_resamples=args.bootstrap_resamples,
                seed=20261024,
            )
            bootstrap[key] = result
    apply_bootstrap_family_holm(bootstrap)

    report = {
        "protocol": {
            "sizes": SIZES,
            "seeds": SEEDS,
            "bootstrap_resamples": args.bootstrap_resamples,
            "paired_unit": "training seed",
            "resampling_unit": "benchmark chain",
        },
        "summaries": summaries,
        "comparisons": comparisons,
        "chain_bootstrap": bootstrap,
    }
    output = ROOT / "results" / "followup" / "size_curve" / "size_curve.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_path = ROOT / "results" / "followup" / "size_curve" / "SIZE_CURVE_source_data.csv"
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("size", "method", "metric", "seed", "value"),
        )
        writer.writeheader()
        for size in SIZES:
            for method in METHODS:
                for metric in METRICS:
                    values = summaries[str(size)][method][metric]["values"]
                    for seed, value in zip(SEEDS, values):
                        writer.writerow(
                            {
                                "size": size,
                                "method": method,
                                "metric": metric,
                                "seed": seed,
                                "value": value,
                            }
                        )
    markdown = write_markdown(report)
    figure = plot_curves(report)
    print(output)
    print(markdown)
    print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
