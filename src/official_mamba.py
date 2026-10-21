from __future__ import annotations

import importlib.util
import os
import sys
import types

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MAMBA_ROOT = os.environ.get('MAMBA_ROOT', os.path.join(ROOT, 'third_party', 'mamba-main'))
for _p in (MAMBA_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _stub_pkg(name):
    m = types.ModuleType(name)
    m.__path__ = []
    sys.modules[name] = m
    return m


def _load_file(module_name, rel_path):
    path = os.path.join(MAMBA_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_available():
    
    if 'mamba_ssm.ops.selective_scan_interface' in sys.modules and \
            'mamba_ssm.modules.mamba_simple' in sys.modules:
        return

    for name in ('mamba_ssm', 'mamba_ssm.utils', 'mamba_ssm.ops', 'mamba_ssm.ops.triton',
                 'mamba_ssm.modules'):
        if name not in sys.modules:
            _stub_pkg(name)

    sys.modules.setdefault('causal_conv1d', types.ModuleType('causal_conv1d'))
    ln = types.ModuleType('mamba_ssm.ops.triton.layer_norm')
    ln._layer_norm_fwd = lambda *a, **k: None
    sys.modules.setdefault('mamba_ssm.ops.triton.layer_norm', ln)
    ssu = types.ModuleType('mamba_ssm.ops.triton.selective_state_update')
    ssu.selective_state_update = None
    sys.modules.setdefault('mamba_ssm.ops.triton.selective_state_update', ssu)
    if not os.path.isdir(MAMBA_ROOT):
        raise FileNotFoundError(
            f'official Mamba repository not found: {MAMBA_ROOT}\n'
            f'git clone https://github.com/state-spaces/mamba third_party/mamba-main '
            f'(or set MAMBA_ROOT).')

    if 'mamba_ssm.utils.torch' not in sys.modules:
        _load_file('mamba_ssm.utils.torch', os.path.join('mamba_ssm', 'utils', 'torch.py'))
    if 'mamba_ssm.ops.selective_scan_interface' not in sys.modules:
        _load_file('mamba_ssm.ops.selective_scan_interface',
                   os.path.join('mamba_ssm', 'ops', 'selective_scan_interface.py'))
    if 'mamba_ssm.modules.mamba_simple' not in sys.modules:
        mod = _load_file('mamba_ssm.modules.mamba_simple',
                         os.path.join('mamba_ssm', 'modules', 'mamba_simple.py'))


        from mamba_ssm.ops import selective_scan_interface as ssi
        mod.selective_scan_fn = ssi.selective_scan_ref
        mod.causal_conv1d_fn = None


def official_selective_scan_ref():
    _ensure_available()
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
    return selective_scan_ref


class OfficialMamba(nn.Module):
    

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=None,
                 **kwargs):
        super().__init__()
        _ensure_available()
        from mamba_ssm.modules.mamba_simple import Mamba
        self.block = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv,
                           expand=expand, dt_rank=dt_rank or 'auto', **kwargs)
        self.block.use_fast_path = False

    def forward(self, x, mask=None):
        return self.block(x)
