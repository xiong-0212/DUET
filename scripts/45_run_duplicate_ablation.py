from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEEDS = [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]
METHODS = ["duet_graph", "grassp_rnafm"]
VARIANTS = {
    "uniform": {
        "manifest": "large_p80_cv5",
        "sample_weight_mode": "uniform",
    },
    "dedup": {
        "manifest": "followup_dedup_290_cv5",
        "sample_weight_mode": "uniform",
    },
}


def write_selected(method: str, results_root: Path) -> None:
    source = (
        ROOT
        / "results"
        / "large_protocol"
        / "cv"
        / method
        / "selected.json"
    )
    target = results_root / "cv" / method / "selected.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def command_for(
    method: str,
    variant: str,
    results_root: Path,
    checkpoint_root: Path,
) -> list[str]:
    spec = VARIANTS[variant]
    common = [
        "--manifest",
        spec["manifest"],
        "--results-root",
        str(results_root.relative_to(ROOT)),
        "--checkpoint-root",
        str(checkpoint_root.relative_to(ROOT)),
        "--sample-weight-mode",
        spec["sample_weight_mode"],
        "--stage",
        "final",
        "--seeds",
        *[str(seed) for seed in SEEDS],
    ]
    if method == "duet_graph":
        return [
            sys.executable,
            str(HERE / "24_train_large_duet.py"),
            "--method",
            method,
            "--run-prefix",
            f"duplicate_{variant}",
            *common,
        ]
    return [
        sys.executable,
        str(HERE / "23_train_large_grassp.py"),
        "--method",
        method,
        *common,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    failures = []
    copy_root = (
        ROOT
        / "results"
        / "followup"
        / "duplicate_weighting"
        / "group"
    )
    for method in METHODS:
        target = copy_root / "final" / method
        target.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            source = (
                ROOT
                / "results"
                / "large_protocol"
                / "final"
                / method
                / f"seed{seed}.json"
            )
            shutil.copy2(source, target / f"seed{seed}.json")
    for variant in args.variants:
        active = []
        for method in METHODS:
            results_root = (
                ROOT
                / "results"
                / "followup"
                / "duplicate_weighting"
                / variant
            )
            checkpoint_root = (
                ROOT
                / "checkpoints"
                / "followup"
                / "duplicate_weighting"
                / variant
            )
            write_selected(method, results_root)
            log_dir = ROOT / "logs" / "followup" / "duplicate_weighting"
            log_dir.mkdir(parents=True, exist_ok=True)
            log = (log_dir / f"{variant}_{method}.log").open(
                "w",
                encoding="utf-8",
            )
            process = subprocess.Popen(
                command_for(method, variant, results_root, checkpoint_root),
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active.append((process, method, log))
            print(f"[run] {variant} {method}", flush=True)
        for process, method, log in active:
            code = process.wait()
            log.close()
            if code:
                failures.append(f"{variant}:{method}")
                print(f"[fail] {variant} {method}", flush=True)
            else:
                print(f"[done] {variant} {method}", flush=True)
    if failures:
        print(f"failures: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
