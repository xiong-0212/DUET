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


ARMS = {
    'ssalt_p10':     'graphs_ssalt_p10_clean.pt',
    'ssalt_p20':     'graphs_ssalt_p20_clean.pt',
    'ssalt_p50':     'graphs_ssalt_p50_clean.pt',
    'ssalt_p10w':    'graphs_ssalt_p10w_clean.pt',
    'ssalt_p10shuf': 'graphs_ssalt_p10shuf_clean.pt',
    'ssalt_p10rand': 'graphs_ssalt_p10rand_clean.pt',
    'ssalt_p30w':    'graphs_ssalt_p30w_clean.pt',
    'ss_pw':         'graphs_ss_pw_clean.pt',
    'ss_pwshuf':     'graphs_ss_pwshuf_clean.pt',
    'ss_w1':         'graphs_ss_w1_clean.pt',
    'bp10':          'graphs_bp10_clean.pt',
    'bp10w':         'graphs_bp10w_clean.pt',

    'bplow_bin':     'graphs_bplow_bin_clean.pt',
    'bplow_w':       'graphs_bplow_w_clean.pt',
    'bplow_shuf':    'graphs_bplow_shuf_clean.pt',
    'ssalt_p01w':    'graphs_ssalt_p01w_clean.pt',

    'pf_ss':         'graphs_pf_ss_clean.pt',
    'pf_ss_shuf':    'graphs_pf_ss_shuf_clean.pt',

    'ss_prune90':    'graphs_ss_prune90_clean.pt',
    'ss_prune70':    'graphs_ss_prune70_clean.pt',

    'rinalmo_ss':    'graphs_seq_ss_rinalmo_clean.pt',
}

MAX_WORKERS = 3


def run_one(tag, seed, graphs, wnorm=None):
    res_tag = f'clean_{tag}'
    res_json = os.path.join(ROOT, 'results', 'controlled', f'train_{res_tag}_seed{seed}.json')
    if os.path.exists(res_json):
        return f'[skip] {res_tag}_seed{seed} (already exists)'
    log_path = os.path.join(ROOT, 'logs', 'controlled_b', f'{res_tag}_seed{seed}.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    cmd = [PY, os.path.join(ROOT, 'src', 'train.py'),
           '--config', os.path.join(ROOT, 'configs', 'controlled.yaml'),
           '--graphs', os.path.join(ROOT, 'data', 'processed', graphs),
           '--tag', res_tag, '--seed', str(seed),
           '--loss', 'tcl_focal', '--drop_val_leaked', '--epochs', '120']
    if wnorm:
        cmd += ['--wnorm', wnorm]

    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8') as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    status = 'ok' if p.returncode == 0 else f'FAILED(rc={p.returncode})'
    return f'[{status}] {res_tag}_seed{seed}  {dt:.0f}s'


def main():
    only = sys.argv[1:] or None
    if only:
        unknown = [t for t in only if t not in ARMS]
        assert not unknown, f'unknown tag: {unknown}'
    jobs = []
    for tag, gf in ARMS.items():
        if only and tag not in only:
            continue
        for seed in SEEDS:
            jobs.append((tag, seed, gf))
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
