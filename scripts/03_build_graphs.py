from __future__ import annotations

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
PROC = os.path.join(ROOT, 'data', 'processed')
BACKUP = os.path.join(ROOT, 'archive', 'graphs_pre_overlap_fix')
PY = sys.executable
sys.path.insert(0, os.path.join(ROOT, 'src'))


sys.path.insert(0, HERE)
from build_graphs_b import CONFIGS


BASE = {
    'graphs_seq_ss.pt':        dict(preset='seq_ss', remove_leaked=False),
    'graphs_seq_ss_clean.pt':  dict(preset='seq_ss', remove_leaked=True),
    'graphs_seq_ss_bp.pt':     dict(preset='seq_ss_alt', remove_leaked=False, thr=0.3),
    'graphs_seq_ss_bp_clean.pt': dict(preset='seq_ss_alt', remove_leaked=True, thr=0.3),
    'graphs_seq_ss_sim.pt':    dict(preset='seq_ss_sim', remove_leaked=False),
    'graphs_seq_ss_sim_clean.pt': dict(preset='seq_ss_sim', remove_leaked=True),
    'graphs_seq_ss_alt_sim.pt': dict(preset='seq_ss_alt_sim', remove_leaked=False, thr=0.3),
    'graphs_seq_ss_alt_sim_clean.pt': dict(preset='seq_ss_alt_sim', remove_leaked=True, thr=0.3),
}


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                       errors='replace')
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-2000:])
        raise RuntimeError(f'failed: {" ".join(cmd)}')


def build(name, preset, remove_leaked, thr=None, wm='none', rand_sub=False,
          node_feats=False, feat_shuffle=False, low=False, mfe_min=None):
    out = os.path.join(PROC, name)
    cmd = [PY, os.path.join(ROOT, 'src', 'graph_builder.py'),
           '--preset', preset, '--out', out, '--weight_mode', wm]
    if remove_leaked:
        cmd += ['--remove_leaked']
    if thr is not None:
        cmd += ['--bp_threshold', str(thr)]
    if low:
        cmd += ['--bp_prob_file', os.path.join(ROOT, 'data', 'raw', 'ss_bp_prob_low.tsv'),
                '--prob_min', '0.01']
    if rand_sub:
        cmd += ['--rand_sub']
    if node_feats:
        cmd += ['--node_feats']
    if feat_shuffle:
        cmd += ['--feat_shuffle']
    if mfe_min is not None:
        cmd += ['--mfe_prob_min', str(mfe_min)]
    run(cmd)


def main():
    os.makedirs(BACKUP, exist_ok=True)
    todo = []
    for name, cfg in BASE.items():
        todo.append((name, dict(preset=cfg['preset'],
                                remove_leaked=cfg.get('remove_leaked', False),
                                thr=cfg.get('thr'), wm='none')))
    for tag, cfg in CONFIGS.items():
        todo.append((f'graphs_{tag}_clean.pt',
                     dict(preset=cfg['preset'], remove_leaked=True,
                          thr=cfg.get('thr'), wm=cfg.get('wm', 'none'),
                          rand_sub=cfg.get('rand_sub', False),
                          node_feats=cfg.get('node_feats', False),
                          feat_shuffle=cfg.get('feat_shuffle', False),
                          low=cfg.get('low', False),
                          mfe_min=cfg.get('mfe_min'))))

    print(f'{len(todo)} graph files to rebuild', flush=True)
    for name, kw in todo:
        src = os.path.join(PROC, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(BACKUP, name))
        build(name, **kw)
        print(f'  [ok] {name}', flush=True)


    run([PY, os.path.join(HERE, 'make_rinalmo_graphs.py')])
    print('  [ok] graphs_seq_ss_rinalmo_clean.pt', flush=True)


    import torch
    print('\nVerify:', flush=True)
    for name, _ in todo:
        obj = torch.load(os.path.join(PROC, name), map_location='cpu', weights_only=False)
        n_rb9 = len(obj['graphs']['RB9'])
        n_tr = len(obj['graphs']['train'])
        old = torch.load(os.path.join(BACKUP, name), map_location='cpu', weights_only=False)
        assert n_rb9 == 6, f'{name}: RB9={n_rb9}≠6'
        assert n_tr == len(old['graphs']['train']), f'{name}: train chain count changed'
        assert len(obj['graphs']['val']) == len(old['graphs']['val']), f'{name}: val chain count changed'
        assert len(obj['graphs']['TE18']) == len(old['graphs']['TE18']), 'TE18 modified'
    print(f'  all {len(todo) + 1} files: RB9=6, train/val/TE18 unchanged ✓')
    print(f'  old graphs backed up to {BACKUP}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
