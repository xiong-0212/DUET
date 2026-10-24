from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEEDS = ["42", "0", "1", "2", "3", "7", "11", "13", "17", "23"]
METHODS = [
    "cobra",
    "duet_graph",
    "duet_tcn",
    "duet_dense",
    "graph_seq3d",
    "graph_seq3d_shuffle",
    "grassp_rnafm",
]
GRASSP_METHODS = {"grassp_rnafm"}


def command_for(args, method: str, stage: str) -> list[str]:
    seeds = ["42"] if stage == "cv" else [str(seed) for seed in args.final_seeds]
    common = [
        "--manifest",
        args.manifest,
        "--results-root",
        args.results_root,
        "--checkpoint-root",
        args.checkpoint_root,
        "--sample-weight-mode",
        args.sample_weight_mode,
    ]
    if method == "cobra":
        return [
            sys.executable,
            str(HERE / "22_train_large_cobra.py"),
            "--stage",
            stage,
            "--seeds",
            *seeds,
            "--epochs",
            str(args.epochs),
            *common,
        ]
    if method in GRASSP_METHODS:
        return [
            sys.executable,
            str(HERE / "23_train_large_grassp.py"),
            "--method",
            method,
            "--stage",
            stage,
            "--seeds",
            *seeds,
            "--epochs",
            str(100 if stage == "cv" else args.epochs),
            *common,
        ]
    return [
        sys.executable,
        str(HERE / "24_train_large_duet.py"),
        "--method",
        method,
        "--stage",
        stage,
        "--seeds",
        *seeds,
        "--epochs",
        str(args.epochs),
        "--run-prefix",
        args.run_prefix,
        *common,
    ]


def select_cv(args, method: str) -> int:
    return subprocess.run(
        [
            sys.executable,
            str(HERE / "25_select_large_cv.py"),
            "--methods",
            method,
            "--seed",
            "42",
            "--results-root",
            args.results_root,
        ],
        cwd=ROOT,
    ).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    parser.add_argument("--stage", choices=["cv", "final", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--final-seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3, 7, 11, 13, 17, 23])
    parser.add_argument(
        "--sample-weight-mode",
        choices=["group", "uniform", "none"],
        default="group",
    )
    args = parser.parse_args()
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    log_dir = ROOT / "logs" / "followup" / args.run_prefix / "queue"
    log_dir.mkdir(parents=True, exist_ok=True)
    failures = []

    def run_stage(stage: str) -> None:
        queue = list(args.methods)
        running = []
        while queue or running:
            while queue and len(running) < args.jobs:
                method = queue.pop(0)
                log_path = log_dir / f"{stage}_{method}.log"
                log = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    command_for(args, method, stage),
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                running.append((process, method, time.time(), log))
                print(f"[start] {stage} {method}", flush=True)
            for index, (process, method, started, log) in enumerate(list(running)):
                code = process.poll()
                if code is None:
                    continue
                running.pop(index)
                log.close()
                elapsed = (time.time() - started) / 60
                if code:
                    failures.append(f"{stage}:{method}")
                    print(f"[fail ] {stage} {method} ({elapsed:.1f} min)", flush=True)
                else:
                    print(f"[done ] {stage} {method} ({elapsed:.1f} min)", flush=True)
                    if stage == "cv":
                        selection = select_cv(args, method)
                        if selection:
                            failures.append(f"cv-select:{method}")
                break
            time.sleep(2)

    if args.stage in {"cv", "all"}:
        run_stage("cv")
    if failures:
        print(f"CV failures: {failures}", file=sys.stderr)
        return 1
    if args.stage in {"final", "all"}:
        run_stage("final")
    if failures:
        print(f"Failures: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
