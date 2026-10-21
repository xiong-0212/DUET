from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)

DEV_SEEDS = [42, 0, 1, 2, 3]
CONF_SEEDS = [7, 11, 13, 17, 23]

COMMON = [
    "--config", os.path.join("configs", "duet.yaml"),
    "--loss", "tcl_focal",
    "--drop_val_leaked",
    "--epochs", "120",
]

TOWER_SPECS = {
    "graph": {
        "dev_tag": "clean_seq_ss",
        "conf_tag": "ns_graph",
        "args": [
            "--arch", "rgcn",
            "--graphs", os.path.join("data", "processed", "graphs_seq_ss_clean.pt"),
        ],
    },
    "tcn": {
        "dev_tag": "slim_tcn",
        "conf_tag": "ns_tcn",
        "args": [
            "--arch", "tcn",
            "--graphs", os.path.join("data", "processed", "graphs_dual_clean.pt"),
            "--model_kv", "ch=160",
        ],
    },
    "dense": {
        "dev_tag": "slim_dense",
        "conf_tag": "ns_dense",
        "args": [
            "--arch", "dense",
            "--graphs", os.path.join("data", "processed", "graphs_dual_clean.pt"),
            "--model_kv", "stem=160,growth=40",
        ],
    },
}


def seeds_for_batch(batch: str):
    if batch == "development":
        return DEV_SEEDS
    if batch == "confirmation":
        return CONF_SEEDS
    return DEV_SEEDS + CONF_SEEDS


def tag_for_seed(tower: str, seed: int):
    if seed in DEV_SEEDS:
        return TOWER_SPECS[tower]["dev_tag"]
    if seed in CONF_SEEDS:
        return TOWER_SPECS[tower]["conf_tag"]
    raise ValueError(f"Unregistered seed {seed}; add it to DEV_SEEDS or CONF_SEEDS")


def wait_one(jobs):
    while True:
        for process, log in jobs:
            if process.poll() is not None:
                if not log.closed:
                    log.close()
                return
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tower",
        choices=["graph", "tcn", "dense", "all"],
        default="all",
    )
    parser.add_argument(
        "--batch",
        choices=["development", "confirmation", "all"],
        default="all",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()

    towers = list(TOWER_SPECS) if args.tower == "all" else [args.tower]
    seeds = args.seeds if args.seeds is not None else seeds_for_batch(args.batch)
    log_dir = os.path.join(ROOT, "logs", "controlled_b")
    os.makedirs(log_dir, exist_ok=True)

    jobs = []
    launched = 0
    for seed in seeds:
        for tower in towers:
            tag = tag_for_seed(tower, seed)
            checkpoint = os.path.join(
                ROOT, "checkpoints", "controlled", f"{tag}_seed{seed}.pt"
            )
            if os.path.exists(checkpoint):
                print(f"[skip] {tag} seed {seed} (checkpoint exists)")
                continue

            command = [
                sys.executable,
                os.path.join("src", "train.py"),
                *COMMON,
                *TOWER_SPECS[tower]["args"],
                "--tag", tag,
                "--seed", str(seed),
            ]
            log = open(
                os.path.join(log_dir, f"{tag}_seed{seed}.log"),
                "w",
                encoding="utf-8",
            )
            print(f"[run ] {tag} seed {seed}")
            jobs.append(
                (subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log)
            )
            launched += 1
            while len([p for p, _ in jobs if p.poll() is None]) >= args.jobs:
                wait_one(jobs)

    for process, log in jobs:
        process.wait()
        if not log.closed:
            log.close()

    failures = [process.returncode for process, _ in jobs if process.returncode != 0]
    print(f"[done] launched {launched}; failures {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
