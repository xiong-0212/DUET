from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SEEDS = [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]

MANIFEST_PREFIX = {
    "exact_sequence_unique": "exact",
    "homology_p80": "p80",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--towers",
        nargs="+",
        choices=["graph", "tcn", "dense"],
        default=["graph", "tcn", "dense"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--manifest",
        choices=sorted(MANIFEST_PREFIX),
        default="exact_sequence_unique",
    )
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    log_dir = ROOT / "logs" / "duet"
    log_dir.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, int, int]] = []
    running: list[tuple[subprocess.Popen, str, int, float, object]] = []

    def finish_one() -> None:
        while True:
            for index, (process, tag, seed, started, log) in enumerate(running):
                if process.poll() is not None:
                    running.pop(index)
                    log.close()
                    if process.returncode:
                        failures.append((tag, seed, int(process.returncode)))
                        print(f"[fail] {tag} seed {seed} ({process.returncode})", flush=True)
                    else:
                        print(
                            f"[done] {tag} seed {seed} ({time.time() - started:.1f}s)",
                            flush=True,
                        )
                    return
            time.sleep(2)

    prefix = MANIFEST_PREFIX[args.manifest]
    towers = {
        "graph": {
            "tag": f"{prefix}_graph",
            "graphs": f"data/processed/graphs_graph_{args.manifest}.pt",
            "args": ["--arch", "rgcn"],
        },
        "tcn": {
            "tag": f"{prefix}_tcn",
            "graphs": f"data/processed/graphs_dual_{args.manifest}.pt",
            "args": ["--arch", "tcn", "--model_kv", "ch=160"],
        },
        "dense": {
            "tag": f"{prefix}_dense",
            "graphs": f"data/processed/graphs_dual_{args.manifest}.pt",
            "args": ["--arch", "dense", "--model_kv", "stem=160,growth=40"],
        },
    }
    for tower in args.towers:
        spec = towers[tower]
        for seed in args.seeds:
            checkpoint = ROOT / "checkpoints" / "duet" / f"{spec['tag']}_seed{seed}.pt"
            if checkpoint.exists():
                print(f"[skip] {spec['tag']} seed {seed}", flush=True)
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
                *spec["args"],
            ]
            log_path = log_dir / f"{spec['tag']}_seed{seed}.log"
            print(f"[run ] {spec['tag']} seed {seed}", flush=True)
            t0 = time.time()
            log = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running.append((process, spec["tag"], seed, t0, log))
            while len(running) >= args.jobs:
                finish_one()
    while running:
        finish_one()
    if failures:
        print(f"failures: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
