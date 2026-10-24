from __future__ import annotations

import torch
import torch.nn as nn


def pad_sequences(x, batch):

    n = x.size(0)
    c = x.size(1)
    counts = torch.bincount(batch, minlength=int(batch.max()) + 1)
    max_len = int(counts.max())
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(n, device=x.device) - start[batch]
    rows = x.new_zeros(counts.numel(), max_len, c)
    rows[batch, pos] = x
    mask = (torch.arange(max_len, device=x.device)[None, :] < counts[:, None]).to(x.dtype)
    return rows.transpose(1, 2).contiguous(), mask, batch, pos


class _Stem(nn.Module):


    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        h = self.conv(x)
        return torch.relu(self.norm(h.transpose(1, 2)).transpose(1, 2))


class _DenseLayer(nn.Module):


    def __init__(self, in_ch, growth, k=3, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_ch)
        self.conv = nn.Conv1d(in_ch, growth, k, padding=k // 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(self.conv(torch.relu(self.norm(x.transpose(1, 2)).transpose(1, 2))))
        return torch.cat([x, h], dim=1)


class _DenseBlock(nn.Module):


    def __init__(self, in_ch, growth, layers, k=3, dropout=0.1):
        super().__init__()
        blocks = []
        for i in range(layers):
            blocks.append(_DenseLayer(in_ch + i * growth, growth, k, dropout))
        self.layers = nn.ModuleList(blocks)
        self.out_ch = in_ch + layers * growth

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x) * mask[:, None, :]
        return x


class _Transition(nn.Module):


    def __init__(self, in_ch, dropout=0.1):
        super().__init__()
        out = in_ch // 2
        self.norm = nn.LayerNorm(in_ch)
        self.conv = nn.Conv1d(in_ch, out, 1)
        self.drop = nn.Dropout(dropout)
        self.out_ch = out

    def forward(self, x, mask=None):
        h = torch.relu(self.norm(x.transpose(1, 2)).transpose(1, 2))
        h = self.drop(self.conv(h))


        return h if mask is None else h * mask[:, None, :]


class DenseSeqTower(nn.Module):


    def __init__(self, in_dim=1280, stem=256, growth=64, layers=(4, 4, 4), k=3,
                 dropout=0.1, head_hidden=64, head_dropout=0.4, split_dim=0, **kwargs):
        super().__init__()
        self.split_dim = split_dim
        self.stem = _Stem(in_dim - split_dim, stem)
        blocks = []
        ch = stem
        for i, n in enumerate(layers):
            blocks.append(_DenseBlock(ch, growth, n, k, dropout))
            ch = blocks[-1].out_ch
            if i < len(layers) - 1:
                blocks.append(_Transition(ch, dropout))
                ch = blocks[-1].out_ch
        self.blocks = nn.ModuleList(blocks)


        self.feat_dim = head_hidden
        self.head = nn.Sequential(
            nn.Linear(ch, head_hidden), nn.LayerNorm(head_hidden), nn.ReLU(),
            nn.Dropout(head_dropout))
        self.fc = nn.Linear(head_hidden, 1)


    _pad = staticmethod(pad_sequences)

    def forward(self, x, edge_index=None, edge_type=None, batch=None, return_feat=False,
                edge_weight=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h, mask, rows, pos = self._pad(x[:, self.split_dim:], batch)
        h = self.stem(h) * mask[:, None, :]
        for block in self.blocks:
            h = block(h, mask) * mask[:, None, :]
        feat = self.head(h.transpose(1, 2)[rows, pos])
        logit = self.fc(feat).squeeze(-1)
        if return_feat:
            return logit, feat
        return logit
