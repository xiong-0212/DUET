from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE_METHODS = [
    "cobra",
    "duet_graph",
    "duet_tcn",
    "duet_dense",
    "graph_seq3d",
    "grassp_rnafm",
]
SEEDS = ["42", "0", "1", "2", "3"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    failures = []
    for repeat in args.repeats:
        results_root = (
            ROOT
            / "results"
            / "followup"
            / "repeated_cv"
            / f"cvrep_{repeat}"
        )
        checkpoint_root = (
            ROOT
            / "checkpoints"
            / "followup"
            / "repeated_cv"
            / f"cvrep_{repeat}"
        )
        command = [
            sys.executable,
            str(HERE / "40_run_protocol_matrix.py"),
            "--manifest",
            f"followup_cvrep_{repeat}_cv5",
            "--results-root",
            str(results_root.relative_to(ROOT)),
            "--checkpoint-root",
            str(checkpoint_root.relative_to(ROOT)),
            "--run-prefix",
            f"cvrep{repeat}",
            "--methods",
            *BASE_METHODS,
            "--stage",
            "all",
            "--jobs",
            str(args.jobs),
            "--final-seeds",
            *SEEDS,
        ]
        print(f"[repeat] {repeat}", flush=True)
        code = subprocess.run(command, cwd=ROOT).returncode
        if code:
            failures.append(f"training:{repeat}")
            continue
        ensemble_code = subprocess.run(
            [
                sys.executable,
                str(HERE / "39_build_followup_ensembles.py"),
                "--results-root",
                str(results_root.relative_to(ROOT)),
                "--stage",
                "all",
                "--seeds",
                *SEEDS,
            ],
            cwd=ROOT,
        ).returncode
        if ensemble_code:
            failures.append(f"ensemble:{repeat}")
    if failures:
        print(f"failures: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
