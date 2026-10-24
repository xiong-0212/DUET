from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.followup_stats import METRICS


EXPECTED_SIZE_METHODS = [
    "cobra",
    "duet_graph",
    "duet_tcn",
    "duet_dense",
    "duet_ensemble",
    "graph_seq3d",
    "graph_seq3d_ensemble",
    "grassp_rnafm",
]
EXPECTED_REPEAT_METHODS = [
    "cobra",
    "duet_graph",
    "duet_tcn",
    "duet_dense",
    "duet_ensemble",
    "graph_seq3d",
    "graph_seq3d_ensemble",
    "grassp_rnafm",
]
SEEDS_10 = [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]
SEEDS_5 = [42, 0, 1, 2, 3]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_seed_files(
    errors: list[str],
    base: Path,
    methods: list[str],
    seeds: list[int],
) -> None:
    for method in methods:
        for seed in seeds:
            path = base / method / f"seed{seed}.json"
            if not path.exists():
                errors.append(f"missing {path.relative_to(ROOT)}")
                continue
            result = load(path)
            for split, metrics in result.get("test", {}).items():
                if not math.isfinite(float(metrics["MCC"])):
                    errors.append(
                        f"{path.relative_to(ROOT)}/{split}: non-finite MCC"
                    )


def main() -> int:
    errors = []
    checks = {}

    all_metrics_path = (
        ROOT / "results" / "followup" / "all_metric_stats" / "all_metrics.json"
    )
    if not all_metrics_path.exists():
        errors.append("missing all_metrics.json")
    else:
        all_metrics = load(all_metrics_path)
        if set(all_metrics.get("protocol", {}).get("metrics", [])) != set(METRICS):
            errors.append("all_metrics: incomplete metric set")
        if int(all_metrics["protocol"]["bootstrap_resamples"]) < 1000:
            errors.append("all_metrics: fewer than 1000 bootstrap resamples")
        for key, result in all_metrics.get("chain_bootstrap", {}).items():
            if set(result.get("metrics", {})) != set(METRICS):
                errors.append(f"all_metrics: incomplete bootstrap {key}")
        if "source_matched_control" not in all_metrics.get("summaries", {}):
            errors.append("all_metrics: missing source-matched control summaries")
        if "external_independent_2split" not in all_metrics.get("families", {}):
            errors.append("all_metrics: missing SMARTPocket external family")
        checks["all_metrics"] = {
            "families": len(all_metrics.get("families", {})),
            "comparisons": sum(
                len(rows)
                for metric in all_metrics.get("families", {}).values()
                for rows in metric.values()
            ),
        }

    shuffled_root = ROOT / "results" / "followup" / "shuffled3d"
    shuffled_methods = [
        "cobra",
        "duet_graph",
        "graph_seq3d",
        "graph_seq3d_shuffle",
        "graph_seqss3d",
    ]
    check_seed_files(
        errors,
        shuffled_root,
        shuffled_methods,
        SEEDS_10,
    )
    shuffled_graph_audit = load(shuffled_root / "graph_audit.json")
    if shuffled_graph_audit.get("errors"):
        errors.append("shuffled3d graph audit contains errors")
    shuffled_analysis = load(shuffled_root / "analysis.json")
    if set(shuffled_analysis["comparisons"]) != set(METRICS):
        errors.append("shuffled3d analysis missing metrics")
    for key, result in shuffled_analysis.get("chain_bootstrap", {}).items():
        if set(result.get("metrics", {})) != set(METRICS):
            errors.append(f"shuffled3d bootstrap incomplete: {key}")
    checks["shuffled3d"] = {
        "duplicate_targets_required": shuffled_graph_audit.get(
            "duplicate_targets_required"
        ),
        "comparisons": len(shuffled_analysis["comparisons"]["MCC"]),
    }

    size_root = ROOT / "results" / "followup" / "size_curve"
    size_analysis_path = size_root / "size_curve.json"
    if not size_analysis_path.exists():
        errors.append("missing size_curve.json")
    else:
        size_analysis = load(size_analysis_path)
        if set(size_analysis["summaries"]) != {"80", "160", "290"}:
            errors.append("size curve does not contain sizes 80/160/290")
        for method in EXPECTED_SIZE_METHODS:
            for size in ["80", "160"]:
                if method not in size_analysis["summaries"][size]:
                    errors.append(f"size curve missing {size}/{method}")
        for metric in METRICS:
            if len(size_analysis["comparisons"].get(metric, {})) != 16:
                errors.append(f"size curve comparison count wrong for {metric}")
        for key, result in size_analysis.get("chain_bootstrap", {}).items():
            if set(result.get("metrics", {})) != set(METRICS):
                errors.append(f"size curve bootstrap incomplete: {key}")
        checks["size_curve"] = {
            "sizes": sorted(int(key) for key in size_analysis["summaries"]),
            "methods": len(size_analysis["summaries"]["290"]),
        }
    for size in [80, 160]:
        check_seed_files(
            errors,
            size_root / f"size_{size}" / "final",
            EXPECTED_SIZE_METHODS,
            SEEDS_10,
        )
    for suffix in [
        ".pdf",
        ".svg",
        ".png",
        ".tiff",
        "_source_data.csv",
        ".alignment-layout.json",
        ".alignment.json",
        ".collision-audit.json",
    ]:
        if not (size_root / f"SIZE_CURVE{suffix}").exists():
            errors.append(f"missing SIZE_CURVE{suffix}")
    if (size_root / "SIZE_CURVE.alignment.json").exists():
        if not load(size_root / "SIZE_CURVE.alignment.json").get("verdict") == "PASS":
            errors.append("SIZE_CURVE panel alignment did not pass")
    if (size_root / "SIZE_CURVE.collision-audit.json").exists():
        if not load(size_root / "SIZE_CURVE.collision-audit.json").get("verdict") == "PASS":
            errors.append("SIZE_CURVE collision audit did not pass")

    repeated_root = ROOT / "results" / "followup" / "repeated_cv"
    repeated_path = repeated_root / "analysis.json"
    if not repeated_path.exists():
        errors.append("missing repeated_cv/analysis.json")
    else:
        repeated = load(repeated_path)
        if set(repeated["partitions"]) != {"101", "102", "103"}:
            errors.append("repeated_cv does not contain three partitions")
        for repeat in ["101", "102", "103"]:
            positives = repeated["partitions"][repeat]["fold_positive_residues"]
            if max(positives) - min(positives) >= 695:
                errors.append(f"repeated_cv/{repeat}: positive range not improved")
            check_seed_files(
                errors,
                repeated_root / f"cvrep_{repeat}" / "final",
                EXPECTED_REPEAT_METHODS,
                SEEDS_5,
            )
            for metric in METRICS:
                if metric not in repeated.get("comparisons", {}).get(repeat, {}):
                    errors.append(f"repeated_cv/{repeat}: missing {metric}")
        checks["repeated_cv"] = {
            "partitions": len(repeated["partitions"]),
            "positive_ranges": {
                repeat: max(repeated["partitions"][repeat]["fold_positive_residues"])
                - min(repeated["partitions"][repeat]["fold_positive_residues"])
                for repeat in ["101", "102", "103"]
            },
        }

    grid_root = ROOT / "results" / "followup" / "grasp_grid"
    for feature in ["ernie", "rnafm"]:
        selection_path = grid_root / feature / "grid_selection.json"
        if not selection_path.exists():
            errors.append(f"missing grid selection {feature}")
            continue
        selection = load(selection_path)
        if len(selection.get("fold0_screen", [])) != 24:
            errors.append(f"grasp_grid/{feature}: expected 24 screened configurations")
        if len(selection.get("candidates", [])) < 3:
            errors.append(f"grasp_grid/{feature}: expected top-3 confirmation")
        check_seed_files(
            errors,
            grid_root / "best" / feature / "final",
            [f"grassp_{feature}"],
            SEEDS_10,
        )
    if not (grid_root / "analysis.json").exists():
        errors.append("missing grasp_grid/analysis.json")
    else:
        grid_analysis = load(grid_root / "analysis.json")
        if set(grid_analysis.get("comparisons", {})) != set(METRICS):
            errors.append("grasp_grid analysis missing metrics")
    checks["grasp_grid"] = {
        feature: {
            "screened": len(
                load(grid_root / feature / "grid_selection.json")["fold0_screen"]
            ),
            "confirmed": len(
                load(grid_root / feature / "grid_selection.json")["candidates"]
            ),
        }
        for feature in ["ernie", "rnafm"]
        if (grid_root / feature / "grid_selection.json").exists()
    }

    duplicate_root = ROOT / "results" / "followup" / "duplicate_weighting"
    if not (duplicate_root / "analysis.json").exists():
        errors.append("missing duplicate_weighting/analysis.json")
    else:
        duplicate = load(duplicate_root / "analysis.json")
        if set(duplicate["summaries"]) != {"duet_graph", "grassp_rnafm"}:
            errors.append("duplicate_weighting: missing model")
        for method in ["duet_graph", "grassp_rnafm"]:
            for variant in ["group", "uniform", "dedup"]:
                check_seed_files(
                    errors,
                    duplicate_root / variant / "final",
                    [method],
                    SEEDS_10,
                )
            for metric in METRICS:
                if metric not in duplicate.get("comparisons", {}).get(method, {}):
                    errors.append(
                        f"duplicate_weighting/{method}: missing {metric}"
                    )
        checks["duplicate_weighting"] = {
            "models": list(duplicate["summaries"]),
            "variants": ["group", "uniform", "dedup"],
        }

    protocol_audit = load(
        ROOT / "results" / "followup" / "protocol" / "manifest_audit.json"
    )
    if not all(
        report["ok"]
        for group in protocol_audit.values()
        for report in group.values()
    ):
        errors.append("followup manifest audit failed")

    if not (
        ROOT / "results" / "training" / "SUMMARY.md"
    ).read_text(encoding="utf-8").startswith("# Training summary"):
        errors.append("SUMMARY.md missing or malformed")

    external_audit_path = (
        ROOT / "results" / "large_protocol" / "smartpocket_audit.json"
    )
    external_graphs_path = (
        ROOT / "results" / "large_protocol" / "smartpocket_graphs_audit.json"
    )
    if not external_audit_path.exists():
        errors.append("missing SMARTPocket external audit")
    else:
        external_audit = load(external_audit_path)
        expected_counts = {
            "external_jiang": (2, 97),
            "time_dependent": (11, 885),
            "hariboss_test": (6, 1010),
        }
        for split, (chains, residues) in expected_counts.items():
            value = external_audit.get("splits", {}).get(split, {})
            if (
                int(value.get("retained", -1)) != chains
                or int(value.get("residues", -1)) != residues
            ):
                errors.append(
                    f"SMARTPocket audit mismatch for {split}"
                )
        independent = [
            external_audit["splits"]["external_jiang"],
            external_audit["splits"]["time_dependent"],
        ]
        if any(
            max(
                (
                    record["max_identity_to_large_pool"]["identity"]
                    for record in split["records"].values()
                    if record["status"] == "retained"
                ),
                default=0.0,
            )
            >= 0.8
            for split in independent
        ):
            errors.append("SMARTPocket external set has an 80% pool overlap")
        if any(
            max(
                (
                    record["max_identity_to_canonical_tests"]["identity"]
                    for record in split["records"].values()
                    if record["status"] == "retained"
                ),
                default=0.0,
            )
            >= 0.8
            for split in independent
        ):
            errors.append(
                "SMARTPocket external set has an 80% canonical-test overlap"
            )
    if not external_graphs_path.exists():
        errors.append("missing SMARTPocket graph audit")
    else:
        external_graphs = load(external_graphs_path)
        if int(external_graphs.get("n_chains", -1)) != 19:
            errors.append("SMARTPocket graph audit chain count is not 19")
        imputed = sum(
            int(record.get("imputed_structure_coordinates", 0))
            for split in external_graphs.get("sets", {}).values()
            for record in split.values()
        )
        if imputed != 4:
            errors.append("SMARTPocket graph audit imputation count is not 4")
        checks["smartpocket_external"] = {
            "independent_chains": 13,
            "source_matched_chains": 6,
            "imputed_coordinates": imputed,
        }

    handoff = (ROOT / "HANDOFF.md").read_text(encoding="utf-8")
    if "followup" not in handoff.lower():
        errors.append("HANDOFF.md does not describe follow-up experiments")

    git_status = subprocess.run(
        ["git", "-C", str(ROOT.parent / "duet"), "status", "--porcelain"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if git_status.returncode or git_status.stdout.strip():
        errors.append("original duet Git worktree is not clean")

    report = {
        "ok": not errors,
        "errors": errors,
        "checks": checks,
        "metrics": METRICS,
        "original_duet_git_clean": not git_status.returncode
        and not git_status.stdout.strip(),
    }
    output = ROOT / "results" / "followup" / "acceptance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if errors:
        for error in errors:
            print(f"[error] {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
