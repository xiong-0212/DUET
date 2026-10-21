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


CONFIGS = {

    'ssalt_p10':     dict(preset='seq_ss_alt', thr=0.10, wm='none'),
    'ssalt_p20':     dict(preset='seq_ss_alt', thr=0.20, wm='none'),
    'ssalt_p50':     dict(preset='seq_ss_alt', thr=0.50, wm='none'),

    'ssalt_p10w':    dict(preset='seq_ss_alt', thr=0.10, wm='prob'),
    'ssalt_p10shuf': dict(preset='seq_ss_alt', thr=0.10, wm='shuffle'),
    'ssalt_p10rand': dict(preset='seq_ss_alt', thr=0.10, wm='ones', rand_sub=True),
    'ssalt_p30w':    dict(preset='seq_ss_alt', thr=0.30, wm='prob'),

    'ss_pw':         dict(preset='seq_ss', wm='prob'),
    'ss_pwshuf':     dict(preset='seq_ss', wm='shuffle'),
    'ss_w1':         dict(preset='seq_ss', wm='ones'),

    'bp10':          dict(preset='seq_bp', thr=0.10, wm='none'),
    'bp10w':         dict(preset='seq_bp', thr=0.10, wm='prob'),

    'bplow_bin':     dict(preset='seq_bp', thr=0.01, wm='none', low=True),
    'bplow_w':       dict(preset='seq_bp', thr=0.01, wm='prob', low=True),
    'bplow_shuf':    dict(preset='seq_bp', thr=0.01, wm='shuffle', low=True),
    'ssalt_p01w':    dict(preset='seq_ss_alt', thr=0.01, wm='prob', low=True),

    'pf_ss':         dict(preset='seq_ss', wm='none', node_feats=True, low=True),
    'pf_ss_shuf':    dict(preset='seq_ss', wm='none', node_feats=True,
                          feat_shuffle=True, low=True),

    'ss_prune90':    dict(preset='seq_ss', wm='none', mfe_min=0.90),
    'ss_prune70':    dict(preset='seq_ss', wm='none', mfe_min=0.70),
}

MAX_WORKERS = 3


def out_path(tag):
    return os.path.join(ROOT, 'data', 'processed', f'graphs_{tag}_clean.pt')


def run_one(tag, cfg):
    out = out_path(tag)
    if os.path.exists(out):
        return f'[skip] {tag} (already exists)'
    log_path = os.path.join(ROOT, 'logs', 'build_graphs_b.log')
    cmd = [PY, os.path.join(ROOT, 'src', 'graph_builder.py'),
           '--preset', cfg['preset'], '--out', out, '--remove_leaked',
           '--weight_mode', cfg['wm']]
    if cfg.get('thr') is not None:
        cmd += ['--bp_threshold', str(cfg['thr'])]
    if cfg.get('low'):
        cmd += ['--bp_prob_file', os.path.join(ROOT, 'data', 'raw', 'ss_bp_prob_low.tsv'),
                '--prob_min', '0.01']
    if cfg.get('rand_sub'):
        cmd += ['--rand_sub']
    if cfg.get('node_feats'):
        cmd += ['--node_feats']
    if cfg.get('feat_shuffle'):
        cmd += ['--feat_shuffle']
    if cfg.get('mfe_min') is not None:
        cmd += ['--mfe_prob_min', str(cfg['mfe_min'])]
    t0 = time.time()
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f'\n===== {tag} @ {time.strftime("%Y-%m-%d %H:%M:%S")} =====\n')
        f.write(' '.join(cmd) + '\n')
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    status = 'ok' if p.returncode == 0 else f'FAILED(rc={p.returncode})'
    return f'[{status}] {tag}  {dt:.0f}s'


def main():
    only = sys.argv[1:] or None
    jobs = [(t, c) for t, c in CONFIGS.items() if not only or t in only]
    if only:
        unknown = [t for t in only if t not in CONFIGS]
        assert not unknown, f'unknown tag: {unknown}'
    print(f'{len(jobs)} graph-building jobs, {MAX_WORKERS} parallel workers', flush=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for msg in ex.map(lambda a: run_one(*a), jobs):
            done += 1
            print(f'  ({done}/{len(jobs)}) {msg}   elapsed {time.time()-t0:.0f}s', flush=True)
    print(f'All done in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
