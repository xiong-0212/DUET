from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
PY = sys.executable

SEEDS = [42, 0, 1, 2, 3]


DATASETS = {
    'clean': {
        'mlp':              'data/processed/graphs_seq_ss_clean.pt',
        'seq_ss':           'data/processed/graphs_seq_ss_clean.pt',
        'seq_ss_alt':       'data/processed/graphs_seq_ss_bp_clean.pt',
        'seq_ss_sim':       'data/processed/graphs_seq_ss_sim_clean.pt',
        'seq_ss_alt_sim':   'data/processed/graphs_seq_ss_alt_sim_clean.pt',
    },
    'orig': {
        'mlp':              'data/processed/graphs_seq_ss.pt',
        'seq_ss':           'data/processed/graphs_seq_ss.pt',
        'seq_ss_alt':       'data/processed/graphs_seq_ss_bp.pt',
        'seq_ss_sim':       'data/processed/graphs_seq_ss_sim.pt',
        'seq_ss_alt_sim':   'data/processed/graphs_seq_ss_alt_sim.pt',
    },
}

MAX_WORKERS = 3


def run_one(tag: str, seed: int, graphs: str, arch: str | None):
    res_json = os.path.join(ROOT, 'results', 'controlled', f'train_{tag}_seed{seed}.json')
    if os.path.exists(res_json):
        return f'[skip] {tag}_seed{seed} (already exists)'
    log_path = os.path.join(ROOT, 'logs', 'controlled', f'{tag}_seed{seed}.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    cmd = [PY, os.path.join(ROOT, 'src', 'train.py'),
           '--config', os.path.join(ROOT, 'configs', 'controlled.yaml'),
           '--graphs', os.path.join(ROOT, graphs),
           '--tag', tag, '--seed', str(seed),
           '--loss', 'tcl_focal', '--drop_val_leaked', '--epochs', '120']
    if arch:
        cmd += ['--arch', arch]

    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8') as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    status = 'ok' if p.returncode == 0 else f'FAILED(rc={p.returncode})'
    return f'[{status}] {tag}_seed{seed}  {dt:.0f}s'


def main():
    only = sys.argv[1:] or None
    jobs = []
    for ds, models in DATASETS.items():
        if only and ds not in only:
            continue
        for model, graphs in models.items():
            for seed in SEEDS:
                arch = 'mlp' if model == 'mlp' else None
                jobs.append((f'{ds}_{model}', seed, graphs, arch))

    print(f'{len(jobs)} training jobs, {MAX_WORKERS} parallel workers', flush=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for msg in ex.map(lambda a: run_one(*a), jobs):
            done += 1
            print(f'  ({done}/{len(jobs)}) {msg}   elapsed {time.time()-t0:.0f}s', flush=True)
    print(f'All done in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
