import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from official_mamba import official_selective_scan_ref

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def main():
    ref = official_selective_scan_ref()
    torch.manual_seed(0)
    B, D, L, N = 2, 8, 24, 6
    dt = torch.rand(B, D, L, dtype=torch.float32) * 0.5
    u = torch.randn(B, D, L, dtype=torch.float32)
    A = -torch.exp(torch.randn(D, N, dtype=torch.float32))
    Bm = torch.randn(B, N, L, dtype=torch.float32) * 0.5
    C = torch.randn(B, N, L, dtype=torch.float32) * 0.5
    Dp = torch.rand(D, dtype=torch.float32)
    z = torch.randn(B, D, L, dtype=torch.float32)

    y_ref = ref(u, dt, A, Bm, C, Dp, z, None, False)


    h = torch.zeros(B, D, N, dtype=torch.float32)
    ys = []
    for t in range(L):
        d_t = dt[:, :, t].unsqueeze(-1)
        a_t = torch.exp(d_t * A)
        b_t = d_t * Bm[:, :, t].unsqueeze(1)
        h = a_t * h + b_t * u[:, :, t].unsqueeze(-1)
        ys.append((h * C[:, :, t].unsqueeze(1)).sum(-1))
    y_mine = torch.stack(ys, dim=2) + Dp[None, :, None] * u
    y_mine = y_mine * F.silu(z)

    dev = float((y_ref - y_mine).abs().max())
    print(f'shapes: official {tuple(y_ref.shape)}  hand-written {tuple(y_mine.shape)}')
    print(f'max |official - hand-written| = {dev:.3e}')
    assert y_ref.shape == y_mine.shape, 'shape mismatch'
    assert dev < 1e-5, f'scan implementations disagree ({dev:.3e})'
    print('OK — the two selective-scan implementations are numerically identical')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
