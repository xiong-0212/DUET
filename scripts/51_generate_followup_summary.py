from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    all_metrics = load(
        ROOT / "results" / "followup" / "all_metric_stats" / "all_metrics.json"
    )
    shuffled = load(
        ROOT / "results" / "followup" / "shuffled3d" / "analysis.json"
    )
    size = load(ROOT / "results" / "followup" / "size_curve" / "size_curve.json")
    repeated = load(
        ROOT / "results" / "followup" / "repeated_cv" / "analysis.json"
    )
    grid = load(ROOT / "results" / "followup" / "grasp_grid" / "analysis.json")
    duplicate = load(
        ROOT / "results" / "followup" / "duplicate_weighting" / "analysis.json"
    )

    lines = [
        "# Follow-up experiment results",
        "",
        "All comparisons use paired training seeds. Chain bootstrap keeps seeds "
        "paired and resamples benchmark chains. Raw and Holm-adjusted p values "
        "are reported for all seven metrics in the machine-readable files.",
        "",
        "## All-metric significance",
        "",
        "The existing comparison families were preserved and expanded from MCC "
        "to MCC, AUC, AUPRC, F1, Precision, Recall and GMean.",
        "",
        "| Family | Metric | Comparison | Delta | Holm p |",
        "|:--|:--|:--|--:|--:|",
    ]
    for family, metric_rows in all_metrics["families"].items():
        for metric in ["MCC", "AUC", "AUPRC", "F1", "Precision", "Recall", "GMean"]:
            for name, value in metric_rows[metric].items():
                lines.append(
                    f"| {family} | {metric} | {name} | "
                    f"{value['mean_difference']:+.4f} | {value['p_holm']:.4g} |"
                )

    lines += [
        "",
        "## Shuffled-3D control",
        "",
        "| Metric | Comparison | Delta | 95% CI | raw p | Holm p |",
        "|:--|:--|--:|:--|--:|--:|",
    ]
    for metric in ["MCC", "AUC", "AUPRC", "F1", "Precision", "Recall", "GMean"]:
        for name, value in shuffled["comparisons"][metric].items():
            lines.append(
                f"| {metric} | {name} | {value['mean_difference']:+.4f} | "
                f"{value['ci95'][0]:+.4f} to {value['ci95'][1]:+.4f} | "
                f"{value['p']:.4g} | {value['p_holm']:.4g} |"
            )

    lines += [
        "",
        "## Fixed-composition training-size curve",
        "",
        "| Method | 80 MCC | 160 MCC | 290 MCC | 160-80 | 290-80 |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for method in [
        "cobra",
        "duet_graph",
        "duet_tcn",
        "duet_dense",
        "duet_ensemble",
        "graph_seq3d",
        "graph_seq3d_ensemble",
        "grassp_rnafm",
    ]:
        values = [
            size["summaries"][str(size_value)][method]["MCC"]["mean"]
            for size_value in [80, 160, 290]
        ]
        delta_160 = size["comparisons"]["MCC"][f"160/{method}"]
        delta_290 = size["comparisons"]["MCC"][f"290/{method}"]
        lines.append(
            f"| {method} | {values[0]:.3f} | {values[1]:.3f} | "
            f"{values[2]:.3f} | {delta_160['mean_difference']:+.3f} "
            f"(Holm {delta_160['p_holm']:.3g}) | "
            f"{delta_290['mean_difference']:+.3f} "
            f"(Holm {delta_290['p_holm']:.3g}) |"
        )
    lines += [
        "",
        "Full seven-metric curves are in `SIZE_CURVE.md`, "
        "`SIZE_CURVE.pdf`, and `SIZE_CURVE_source_data.csv`.",
        "",
        "## Repeated cluster-stratified CV",
        "",
        "| Assignment | Positive-residue range | DUET ensemble - CoBRA (MCC) | Holm p |",
        "|--:|--:|--:|--:|",
    ]
    for repeat in ["101", "102", "103"]:
        positives = repeated["partitions"][repeat]["fold_positive_residues"]
        comparison = repeated["comparisons"][repeat]["MCC"][
            "duet_ensemble_vs_cobra"
        ]
        lines.append(
            f"| {repeat} | {max(positives) - min(positives)} | "
            f"{comparison['mean_difference']:+.4f} | "
            f"{comparison['p_holm']:.4g} |"
        )

    lines += [
        "",
        "## GRASSP hyperparameter sensitivity",
        "",
        "| Encoder | Selected configuration | OOF MCC | Final - DUET (MCC) | Holm p |",
        "|:--|:--|--:|--:|--:|",
    ]
    for feature in ["ernie", "rnafm"]:
        best = grid["selections"][feature]["best"]
        comparison = grid["comparisons"]["MCC"][f"grassp_{feature}_vs_duet"]
        lines.append(
            f"| {feature} | {best['name']} | "
            f"{best['oof_metrics_at_selected_threshold']['MCC']:.4f} | "
            f"{comparison['mean_difference']:+.4f} | "
            f"{comparison['p_holm']:.4g} |"
        )

    lines += [
        "",
        "## Duplicate-weighting ablation",
        "",
        "| Model | Comparison | MCC delta | Holm p |",
        "|:--|:--|--:|--:|",
    ]
    for method in ["duet_graph", "grassp_rnafm"]:
        for name, value in duplicate["comparisons"][method]["MCC"].items():
            lines.append(
                f"| {method} | {name} | {value['mean_difference']:+.4f} | "
                f"{value['p_holm']:.4g} |"
            )
    lines += [""]
    followup_path = ROOT / "results" / "followup" / "FOLLOWUP_RESULTS.md"
    followup_path.write_text("\n".join(lines), encoding="utf-8")

    summary_path = ROOT / "results" / "training" / "SUMMARY.md"
    existing = summary_path.read_text(encoding="utf-8")
    marker = "\n## Follow-up experiments\n"
    existing = existing.split(marker, 1)[0].rstrip() + "\n"
    summary_section = [
        "",
        "## Follow-up experiments",
        "",
        "All seven metrics, Holm-adjusted paired tests and chain bootstrap are "
        "reported under `results/followup/`.",
        "",
        "- Large-data shuffled-3D control completed.",
        "- Fixed-composition 80/160/290 training-size curve completed.",
        "- Three additional cluster-stratified five-fold assignments completed "
        "with five final seeds per partition (reduced from ten for compute).",
        "- Full 24-point GRASSP grid completed for ERNIE-RNA and RNA-FM.",
        "- Duplicate-weighting ablation completed for DUET graph and GRASSP RNA-FM.",
        "",
        "See `results/followup/FOLLOWUP_RESULTS.md` for key values and each "
        "experiment directory for full tables.",
        "",
    ]
    summary_path.write_text(
        existing + "\n".join(summary_section),
        encoding="utf-8",
    )
    print(followup_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
