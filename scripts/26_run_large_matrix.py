from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEEDS = [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]
METHODS = [
    "cobra",
    "duet_graph",
    "duet_tcn",
    "duet_dense",
    "graph_seq3d",
    "graph_seqss3d",
    "grassp_ernie",
    "grassp_rnafm",
    "rgcn_ernie",
]
GRASSP_METHODS = {"grassp_ernie", "grassp_rnafm"}


def command_for(method: str, stage: str, epochs: int) -> list[str]:
    seeds = ["42"] if stage == "cv" else [str(seed) for seed in SEEDS]
    if method == "cobra":
        return [
            sys.executable,
            str(HERE / "22_train_large_cobra.py"),
            "--stage",
            stage,
            "--seeds",
            *seeds,
            "--epochs",
            str(epochs),
        ]
    if method in GRASSP_METHODS:
        method_epochs = 100 if stage == "cv" else epochs
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
            str(method_epochs),
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
        str(epochs),
    ]


def select_cv(method: str) -> int:
    command = [
        sys.executable,
        str(HERE / "25_select_large_cv.py"),
        "--methods",
        method,
        "--seed",
        "42",
    ]
    return subprocess.run(command, cwd=ROOT).returncode


def run_parallel(
    methods: list[str],
    stage: str,
    epochs: int,
    jobs: int,
) -> list[str]:
    failures = []
    running = []
    log_dir = ROOT / "logs" / "large" / "queue"
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = list(methods)
    while queue or running:
        while queue and len(running) < jobs:
            method = queue.pop(0)
            log_path = log_dir / f"{stage}_{method}.log"
            log = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command_for(method, stage, epochs),
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running.append((process, method, time.time(), log))
            print(f"[start] {stage} {method} -> {log_path.relative_to(ROOT)}", flush=True)
        for index, (process, method, started, log) in enumerate(list(running)):
            code = process.poll()
            if code is None:
                continue
            running.pop(index)
            log.close()
            elapsed = (time.time() - started) / 60
            if code:
                failures.append(method)
                print(f"[fail ] {stage} {method} ({elapsed:.1f} min)", flush=True)
            else:
                print(f"[done ] {stage} {method} ({elapsed:.1f} min)", flush=True)
                if stage == "cv":
                    code = select_cv(method)
                    if code:
                        failures.append(f"{method}:selection")
                        print(f"[fail ] select {method}", flush=True)
            break
        time.sleep(2)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["cv", "final", "all"], default="all")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    failures = []
    if args.stage in {"cv", "all"}:
        failures.extend(
            run_parallel(args.methods, "cv", args.epochs, args.jobs)
        )
    if failures:
        print(f"stopping after CV failures: {failures}", file=sys.stderr)
        return 1
    if args.stage in {"final", "all"}:
        failures.extend(
            run_parallel(args.methods, "final", args.epochs, args.jobs)
        )
    if failures:
        print(f"failures: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
