from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SEEDS = [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]
VARIANTS = {
    "seq3d": {
        "tag": "p80_seq3d_graph",
        "graphs": "data/processed/graphs_graph_seq3d_homology_p80.pt",
    },
    "seqss3d": {
        "tag": "p80_seqss3d_graph",
        "graphs": "data/processed/graphs_graph_seqss3d_homology_p80.pt",
    },
    "seq3d_shuffle": {
        "tag": "p80_seq3d_shuffle_graph",
        "graphs": "data/processed/graphs_graph_seq3d_shuffle_homology_p80.pt",
    },
    "seq3d_ernie": {
        "tag": "p80_seq3d_ernie_graph",
        "graphs": "data/processed/graphs_graph_seq3d_ernie_homology_p80.pt",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    log_dir = ROOT / "logs" / "structure3d"
    log_dir.mkdir(parents=True, exist_ok=True)
    running = []
    failures = []

    def finish_one() -> None:
        while True:
            for index, (process, tag, seed, started, log) in enumerate(running):
                if process.poll() is not None:
                    running.pop(index)
                    log.close()
                    if process.returncode:
                        failures.append((tag, seed, process.returncode))
                        print(f"[fail] {tag} seed {seed}", flush=True)
                    else:
                        print(
                            f"[done] {tag} seed {seed} ({time.time() - started:.1f}s)",
                            flush=True,
                        )
                    return
            time.sleep(2)

    for variant in args.variants:
        spec = VARIANTS[variant]
        for seed in args.seeds:
            checkpoint = ROOT / "checkpoints" / "duet" / f"{spec['tag']}_seed{seed}.pt"
            if checkpoint.exists():
                print(f"[skip] {spec['tag']} seed {seed}")
                continue
            command = [
                sys.executable,
                str(ROOT / "src" / "duet_model" / "train.py"),
                "--config",
                str(ROOT / "configs" / "duet_clean.yaml"),
                "--graphs",
                str(ROOT / spec["graphs"]),
                "--tag",
                spec["tag"],
                "--seed",
                str(seed),
                "--loss",
                "tcl_focal",
                "--epochs",
                str(args.epochs),
                "--arch",
                "rgcn",
            ]
            log = (log_dir / f"{spec['tag']}_seed{seed}.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            print(f"[run ] {spec['tag']} seed {seed}", flush=True)
            running.append((process, spec["tag"], seed, time.time(), log))
            while len(running) >= args.jobs:
                finish_one()
    while running:
        finish_one()
    if failures:
        print(failures, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
