from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.large_protocol import load_large_manifest
from duet_clean.large_stats import (
    EXTERNAL_SPLITS,
    FOUR_SPLITS,
    SHARED_SPLITS,
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
THREE_SPLIT_METHODS = {
    "graph_seq3d",
    "graph_seqss3d",
    "grassp_ernie",
    "grassp_rnafm",
    "rgcn_ernie",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    errors = []
    checks = []
    splits, manifest = load_large_manifest()
    external = load(ROOT / "data" / "external" / "external_sets.json")["splits"]
    leakage_path = (
        ROOT
        / "results"
        / "large_protocol"
        / "large_p80_cv5_leakage.json"
    )
    leakage = load(leakage_path)
    if int(leakage["n_violations"]) != 0:
        errors.append(
            f"cross-fold homology violations: {leakage['n_violations']}"
        )

    for method in METHODS:
        selected_path = (
            ROOT / "results" / "large_protocol" / "cv" / method / "selected.json"
        )
        if not selected_path.exists():
            errors.append(f"missing {selected_path.relative_to(ROOT)}")
            continue
        selected = load(selected_path)
        expected_splits = (
            SHARED_SPLITS if method in THREE_SPLIT_METHODS else FOUR_SPLITS
        )
        final_seeds = set()
        for seed in SEEDS:
            path = (
                ROOT
                / "results"
                / "large_protocol"
                / "final"
                / method
                / f"seed{seed}.json"
            )
            if not path.exists():
                errors.append(f"missing {path.relative_to(ROOT)}")
                continue
            final_seeds.add(seed)
            result = load(path)
            if abs(float(result["threshold"]) - float(selected["threshold"])) > 1e-12:
                errors.append(f"{method}/seed{seed}: threshold mismatch")
            if method != "duet_ensemble":
                if int(result["fixed_epoch"]) != int(selected["selected_epoch"]):
                    errors.append(f"{method}/seed{seed}: epoch mismatch")
            if set(result["test"]) != set(expected_splits):
                errors.append(f"{method}/seed{seed}: test split mismatch")
            for split in expected_splits:
                expected_ids = {record.rna_id for record in splits[split]}
                actual_ids = set(result["per_chain"][split])
                if actual_ids != expected_ids:
                    errors.append(
                        f"{method}/seed{seed}/{split}: chain set mismatch"
                    )
                if not math.isfinite(float(result["test"][split]["MCC"])):
                    errors.append(
                        f"{method}/seed{seed}/{split}: non-finite MCC"
                    )
            for split in EXTERNAL_SPLITS:
                external_path = (
                    ROOT
                    / "results"
                    / "large_protocol"
                    / "external"
                    / f"{method}_seed{seed}.json"
                )
                if not external_path.exists():
                    errors.append(f"missing {external_path.relative_to(ROOT)}")
                    continue
                external_result = load(external_path)
                if set(external_result["per_chain"][split]) != set(external[split]):
                    errors.append(
                        f"{method}/seed{seed}/{split}: external chain set mismatch"
                    )
        checks.append(
            {
                "method": method,
                "selected_epoch": int(selected["selected_epoch"]),
                "selected_threshold": float(selected["threshold"]),
                "final_seed_count": len(final_seeds),
            }
        )

    for split in EXTERNAL_SPLITS:
        if not external[split]:
            errors.append(f"external split {split} is empty")

    analysis_path = ROOT / "results" / "large_protocol" / "analysis.json"
    summary_path = ROOT / "results" / "training" / "SUMMARY.md"
    if not analysis_path.exists():
        errors.append("missing results/large_protocol/analysis.json")
    if not summary_path.exists():
        errors.append("missing results/training/SUMMARY.md")

    report = {
        "ok": not errors,
        "errors": errors,
        "expected_seeds": SEEDS,
        "manifest_counts": manifest["counts"],
        "cluster_count": manifest["audit"]["cluster_count"],
        "cross_fold_homology_violations": int(leakage["n_violations"]),
        "external_counts": {
            split: len(records) for split, records in external.items()
        },
        "methods": checks,
    }
    output = ROOT / "results" / "large_protocol" / "acceptance.json"
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
