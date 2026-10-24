from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def comparison_table(comparisons: dict) -> list[str]:
    lines = [
        "| Comparison | Delta MCC | 95% CI | raw p | Holm p | Seeds better |",
        "|:--|--:|:--|--:|--:|--:|",
    ]
    for name, value in comparisons.items():
        lines.append(
            f"| {name} | {value['mean_difference']:+.3f} | "
            f"{value['ci95'][0]:+.3f} to {value['ci95'][1]:+.3f} | "
            f"{value['p']:.4g} | {value['p_holm']:.4g} | "
            f"{value['seeds_first_better']}/{value['n']} |"
        )
    return lines


def main() -> int:
    analysis = load(ROOT / "results" / "large_protocol" / "analysis.json")
    manifest = load(
        ROOT / "data" / "clean" / "manifests" / "large_p80_cv5.json"
    )
    lines = [
        "# Training summary",
        "",
        "The current split is the large test-clean pool with 80% identity "
        "clustering and five homology-aware folds. All final comparisons use "
        "the same ten paired seeds.",
        "",
        "## Large-data protocol",
        "",
        f"- Pool: **{manifest['counts']['pool']}** chains.",
        f"- Homology components: **{manifest['audit']['cluster_count']}**.",
        f"- Retained exact duplicates: "
        f"**{manifest['audit']['retained_exact_duplicate_records']}** records.",
        f"- Fold sizes: **{manifest['audit']['fold_sizes']}**.",
        "- Cross-fold homology violations at 80%: **0**.",
        "- Sampling weight per record: **1 / homology-cluster size**.",
        "- CV selected epoch as the median fold-best epoch and threshold "
        "from pooled out-of-fold MCC.",
        "",
        "## Internal test MCC",
        "",
        "| Method | Mean | SD | TE18 | TL12 | JL10 | RB9 |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    splits = ["TE18", "TL12", "JL10", "RB9"]
    for method, summary in analysis["internal"]["summaries"].items():
        per = summary["per_split"]
        lines.append(
            f"| {method} | {summary['average']['mean']:.3f} | "
            f"{summary['average']['sd']:.3f} | "
            + " | ".join(
                f"{per[split]['mean']:.3f}" if split in per else "-"
                for split in splits
            )
            + " |"
        )
    lines += [
        "",
        "## Primary comparisons",
        "",
        *comparison_table(analysis["internal"]["families"]["primary_4split"]),
        "",
        "At fixed threshold 0.5:",
        "",
        *comparison_table(
            analysis["internal"]["families"]["primary_4split_fixed_0.5"]
        ),
        "",
        "## Structure",
        "",
        *comparison_table(analysis["internal"]["families"]["structure_3split"]),
        "",
        "## Crossed controls",
        "",
        *comparison_table(
            analysis["internal"]["families"]["crossed_controls_3split"]
        ),
        "",
        "## Chain bootstrap",
        "",
        "| Comparison | Delta | 95% CI | p |",
        "|:--|--:|:--|--:|",
    ]
    for name, value in analysis["internal"]["chain_bootstrap"].items():
        lines.append(
            f"| {name} | {value['observed_delta']:+.3f} | "
            f"{value['ci95'][0]:+.3f} to {value['ci95'][1]:+.3f} | "
            f"{value['p_two_sided']:.4g} |"
        )
    lines += [
        "",
        "## SMARTPocket external validation",
        "",
        "Independent sets are `external_jiang` and `time_dependent`; "
        "`hariboss_test` is a source-matched control and is excluded from the "
        "independent average.",
        "",
        "| Method | Jiang MCC | Time-dependent MCC | Independent mean |",
        "|:--|--:|--:|--:|",
    ]
    for method, summary in analysis["external"]["summaries"].items():
        per = summary["per_split"]
        lines.append(
            f"| {method} | {per['external_jiang']['mean']:.3f} | "
            f"{per['time_dependent']['mean']:.3f} | "
            f"{summary['average']['mean']:.3f} |"
        )
    lines += [
        "",
        *comparison_table(analysis["external"]["comparisons"]),
        "",
        "## Reproduction",
        "",
        "Run `run_large_protocol.ps1`. Machine-readable outputs are under "
        "`results/large_protocol/`; the cluster and leakage audits are "
        "`large_p80_cv5_clusters.csv` and `large_p80_cv5_leakage.json`.",
        "",
        "The previous strict-split results are preserved in "
        "`SUMMARY_LEGACY.md`.",
        "",
    ]
    summary_path = ROOT / "results" / "training" / "SUMMARY.md"
    legacy_path = ROOT / "results" / "training" / "SUMMARY_LEGACY.md"
    if summary_path.exists() and not legacy_path.exists():
        shutil.copy2(summary_path, legacy_path)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
