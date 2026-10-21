from __future__ import annotations

import torch
import torch.nn as nn

from densenet_model import _Stem, pad_sequences


class _TCNBlock(nn.Module):
    

    def __init__(self, ch, k=3, dilation=1, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.conv = nn.Conv1d(ch, ch, k, padding=dilation * (k - 1) // 2, dilation=dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = torch.relu(self.norm(x.transpose(1, 2)).transpose(1, 2))
        h = self.drop(self.conv(h))
        return (x + h) * mask[:, None, :]


class TCNSeqTower(nn.Module):
    

    def __init__(self, in_dim=1280, split_dim=0, ch=320, dilations=(1, 2, 4, 8), k=3,
                 dropout=0.1, head_hidden=64, head_dropout=0.4, **kwargs):
        super().__init__()
        self.split_dim = split_dim
        self.stem = _Stem(in_dim - split_dim, ch)
        self.blocks = nn.ModuleList([_TCNBlock(ch, k, d, dropout) for d in dilations])
        self.feat_dim = head_hidden
        self.head = nn.Sequential(
            nn.Linear(ch, head_hidden), nn.LayerNorm(head_hidden), nn.ReLU(),
            nn.Dropout(head_dropout))
        self.fc = nn.Linear(head_hidden, 1)

    def forward(self, x, edge_index=None, edge_type=None, batch=None, return_feat=False,
                edge_weight=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h, mask, rows, pos = pad_sequences(x[:, self.split_dim:], batch)
        h = self.stem(h) * mask[:, None, :]
        for blk in self.blocks:
            h = blk(h, mask)
        feat = self.head(h.transpose(1, 2)[rows, pos])
        logit = self.fc(feat).squeeze(-1)
        return (logit, feat) if return_feat else logit


class _TransformerBlock(nn.Module):
    

    def __init__(self, d, heads=4, ffn=512, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask, mask):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x * mask[:, :, None]


class TransformerSeqTower(nn.Module):
    

    def __init__(self, in_dim=1280, split_dim=0, d=256, heads=4, ffn=640, layers=2,
                 max_len=161, dropout=0.1, head_hidden=64, head_dropout=0.4, **kwargs):
        super().__init__()
        self.split_dim = split_dim
        self.proj = nn.Linear(in_dim - split_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([_TransformerBlock(d, heads, ffn, dropout)
                                     for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.feat_dim = head_hidden
        self.head = nn.Sequential(
            nn.Linear(d, head_hidden), nn.LayerNorm(head_hidden), nn.ReLU(),
            nn.Dropout(head_dropout))
        self.fc = nn.Linear(head_hidden, 1)

    def forward(self, x, edge_index=None, edge_type=None, batch=None, return_feat=False,
                edge_weight=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h, mask, rows, pos = pad_sequences(x[:, self.split_dim:], batch)
        h = self.proj(h.transpose(1, 2)) + self.pos[:, :h.size(2)]
        h = h * mask[:, :, None]
        key_padding_mask = mask == 0
        for blk in self.blocks:
            h = blk(h, key_padding_mask, mask)
        feat = self.head(self.norm(h))[rows, pos]
        logit = self.fc(feat).squeeze(-1)
        return (logit, feat) if return_feat else logit
