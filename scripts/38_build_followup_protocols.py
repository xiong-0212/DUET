from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.followup_data import (
    build_repeated_cv_manifests,
    build_exact_deduplicated_manifest,
    build_size_curve_manifests,
    validate_manifest,
    write_manifest,
)
from duet_clean.identity import IdentityEngine
from duet_clean.io import load_config, write_json


def load(name: str) -> dict:
    return json.loads(
        (ROOT / "data" / "clean" / "manifests" / f"{name}.json").read_text(
            encoding="utf-8"
        )
    )


def main() -> int:
    config = load_config("configs/protocol.json")
    engine = IdentityEngine.from_config(config)
    base = load("large_p80_cv5")
    size_manifests = build_size_curve_manifests(base, (80, 160))
    repeated = build_repeated_cv_manifests(base, (101, 102, 103))
    deduplicated = build_exact_deduplicated_manifest(base)
    reports = {}
    for group in ["size", "repeated_cv"]:
        reports[group] = {}
    for size, manifest in size_manifests.items():
        path = write_manifest(manifest)
        report = validate_manifest(base, manifest, engine)
        reports["size"][str(size)] = report
        print(
            f"[size {size}] {path.relative_to(ROOT)} "
            f"pool={manifest['counts']['pool']} "
            f"folds={report['fold_sizes']} positives={report['fold_positive_residues']}"
        )
    for seed, manifest in repeated.items():
        path = write_manifest(manifest)
        report = validate_manifest(base, manifest, engine)
        reports["repeated_cv"][str(seed)] = report
        print(
            f"[cv {seed}] {path.relative_to(ROOT)} "
            f"folds={report['fold_sizes']} positives={report['fold_positive_residues']}"
        )
    dedup_path = write_manifest(deduplicated)
    dedup_report = validate_manifest(base, deduplicated, engine)
    reports["deduplicated"] = {"290": dedup_report}
    print(
        f"[dedup] {dedup_path.relative_to(ROOT)} "
        f"pool={deduplicated['counts']['pool']} "
        f"folds={dedup_report['fold_sizes']} "
        f"positives={dedup_report['fold_positive_residues']}"
    )
    if not all(item["ok"] for group in reports.values() for item in group.values()):
        raise ValueError(json.dumps(reports, indent=2))
    output = ROOT / "results" / "followup" / "protocol" / "manifest_audit.json"
    write_json(output, reports)
    print(output.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
