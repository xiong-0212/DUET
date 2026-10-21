from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import to_dense_batch

from rgcn_model import RGCNBlock, make_norm


class MambaBlock(nn.Module):
    

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=None):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_model // 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, u, mask=None):
        B, L, _ = u.shape
        xz = self.in_proj(u)
        x, z = xz.chunk(2, dim=-1)
        x = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x = F.silu(x)
        dbc = self.x_proj(x)
        dt, Bm, Cm = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        h = x.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(L):
            d_t = dt[:, t].unsqueeze(-1)
            a_t = torch.exp(d_t * A)
            b_t = d_t * Bm[:, t].unsqueeze(1)
            h = a_t * h + b_t * x[:, t].unsqueeze(-1)
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + self.D * x
        return self.out_proj(y * F.silu(z))


class DualViewBranch(nn.Module):
    

    def __init__(self, in_dim=640, proj=128, hidden=128, layers=2, dropout=0.4,
                 nhead=4, ffn=512, use_gate=True, seq_block='transformer'):
        super().__init__()
        self.use_gate = use_gate
        self.seq_block = seq_block
        self.hidden = hidden
        self.proj = nn.Sequential(nn.Linear(in_dim, proj), nn.LayerNorm(proj),
                                  nn.ReLU())
        self.lstm = nn.LSTM(proj, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0, bidirectional=True)
        d = hidden * 2
        self.tr_in = nn.Linear(proj, d)
        if seq_block == 'mamba_official':
            from official_mamba import OfficialMamba
            self.trans = OfficialMamba(d, d_state=16, d_conv=4, expand=2)
        elif seq_block == 'mamba':
            self.trans = MambaBlock(d, d_state=16, d_conv=4, expand=2)
        else:
            self.trans = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=ffn,
                                                    dropout=dropout, batch_first=True)
        self.gate = nn.Linear(d * 2, d)
        self.d = d

    def forward(self, x_pad, mask):

        p = self.proj(x_pad)
        lengths = mask.sum(-1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(p, lengths, batch_first=True,
                                                   enforce_sorted=False)
        h_bi, _ = self.lstm(packed)
        h_bi, _ = nn.utils.rnn.pad_packed_sequence(h_bi, batch_first=True)
        tr_in = self.tr_in(p)
        h_tr = (self.trans(tr_in) if self.seq_block in ('mamba', 'mamba_official')
                else self.trans(tr_in, src_key_padding_mask=~mask))
        if self.use_gate:
            g = torch.sigmoid(self.gate(torch.cat([h_bi, h_tr], dim=-1)))
            h = g * h_bi + (1.0 - g) * h_tr
        else:
            h = 0.5 * h_bi + 0.5 * h_tr
        return h[mask], h_bi[mask]


class _Head(nn.Module):
    

    def __init__(self, d, hh, dropout):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(d, hh), nn.LayerNorm(hh), nn.ReLU(),
                                 nn.Dropout(dropout))
        self.fc = nn.Linear(hh, 1)

    def forward(self, z):
        f = self.seq(z)
        return self.fc(f).squeeze(-1), f


class MLPBranch(nn.Module):
    

    def __init__(self, in_dim=1280, d=128, dropout=0.1, widths=(1024, 256, 128)):
        super().__init__()
        layers, prev = [], in_dim
        for w in widths:
            layers += [nn.Linear(prev, w), nn.LayerNorm(w), nn.ReLU(),
                       nn.Dropout(dropout)]
            prev = w
        layers += [nn.Linear(prev, d), nn.LayerNorm(d), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.d = d

    def forward(self, x):
        return self.net(x)


class DualBackboneModel(nn.Module):
    

    def __init__(self, in_dim=1920, split_dim=640, hidden=128, num_relations=2,
                 num_layers=2, dropout=0.4, norm='graph', residual=True,
                 num_bases=None, head_hidden=None, fusion='gate',
                 mlp_d=128, mlp_dropout=0.1, mlp_widths=None,
                 use_graph=True, use_mlp=True):
        super().__init__()
        self.fusion = fusion
        self.split_dim = split_dim
        self.use_graph, self.use_mlp = use_graph, use_mlp
        d = hidden
        self.graph = None
        if use_graph:
            blocks = []
            for i in range(num_layers):
                fin = split_dim if i == 0 else hidden
                blocks.append(RGCNBlock(fin, hidden, num_relations, norm=norm,
                                        dropout=dropout, residual=residual,
                                        num_bases=num_bases))
            self.graph = nn.ModuleList(blocks)
        self.mlp = nn.ModuleList()
        if use_mlp:
            self.mlp.append(MLPBranch(in_dim - split_dim, mlp_d, mlp_dropout,
                                      widths=tuple(mlp_widths) if mlp_widths
                                      else (1024, 256, 128)))
            self.proj_m = nn.Linear(mlp_d, d)
        self.proj_g = nn.Linear(hidden, d) if use_graph else None
        if fusion == 'gate' and use_graph and use_mlp:
            self.fuse_gate = nn.Linear(2 * d, d)

        hh = head_hidden or hidden // 2
        self.feat_dim = hh
        if fusion == 'logit_avg':
            self.head_g = _Head(d, hh, dropout)
            self.head_m = _Head(d, hh, dropout)
            self.head_shared = None
        else:
            self.head_shared = _Head(d, hh, dropout)

    def _encode(self, x, edge_index, edge_type, batch, edge_weight):
        h_g = h_m = None
        if self.graph is not None:
            h = x[:, :self.split_dim]
            for blk in self.graph:
                h = blk(h, edge_index, edge_type, batch, edge_weight)
            h_g = self.proj_g(h)
        if len(self.mlp):
            h_m = self.proj_m(self.mlp[0](x[:, self.split_dim:]))
        return h_g, h_m

    def forward(self, x, edge_index, edge_type, batch=None, return_feat=False,
                edge_weight=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h_g, h_m = self._encode(x, edge_index, edge_type, batch, edge_weight)
        if self.fusion == 'logit_avg' and h_g is not None and h_m is not None:
            lg, fg = self.head_g(h_g)
            lm, fm = self.head_m(h_m)
            logit = 0.5 * (lg + lm)
            feat = 0.5 * (fg + fm)
        else:
            if h_g is not None and h_m is not None:
                if self.fusion == 'gate':
                    g = torch.sigmoid(self.fuse_gate(torch.cat([h_g, h_m], dim=-1)))
                    z = g * h_g + (1.0 - g) * h_m
                else:
                    z = 0.5 * (h_g + h_m)
            else:
                z = h_g if h_g is not None else h_m
            logit, feat = self.head_shared(z)
        if return_feat:
            return logit, feat
        return logit


class FusionModel(nn.Module):
    

    def __init__(self, in_dim=640, hidden=128, num_relations=2, num_layers=2,
                 dropout=0.4, norm='graph', residual=True, num_bases=None,
                 head_hidden=None, fusion='gate', dv_hidden=128, dv_layers=2,
                 dv_proj=128, dv_gate=True, use_graph=True, use_seq=True,
                 use_aux=False, aux_hidden=64, dv_seq_block='transformer'):
        super().__init__()
        self.fusion = fusion
        self.use_graph, self.use_seq = use_graph, use_seq
        self.aux = None
        d = hidden
        self.graph = None
        if use_graph:
            blocks = []
            for i in range(num_layers):
                fin = in_dim if i == 0 else hidden
                blocks.append(RGCNBlock(fin, hidden, num_relations, norm=norm,
                                        dropout=dropout, residual=residual,
                                        num_bases=num_bases))
            self.graph = nn.ModuleList(blocks)
        self.dv = None
        if use_seq:
            self.dv = DualViewBranch(in_dim, dv_proj, dv_hidden, dv_layers, dropout,
                                     use_gate=dv_gate, seq_block=dv_seq_block)
            self.proj_dv = nn.Linear(self.dv.d, d)
        self.proj_g = nn.Linear(hidden, d) if use_graph else None
        n_in = (d if use_graph else 0) + (d if use_seq else 0)
        if fusion == 'gate' and use_graph and use_seq:
            self.fuse_gate = nn.Linear(2 * d, d)
        elif fusion == 'concat' and n_in != d:
            self.fuse_proj = nn.Linear(n_in, d)

        hh = head_hidden or hidden // 2
        self.feat_dim = hh
        self.head = nn.Sequential(nn.Linear(d, hh), nn.LayerNorm(hh), nn.ReLU(),
                                  nn.Dropout(dropout))
        self.fc = nn.Linear(hh, 1)
        if use_aux and self.dv is not None:

            self.aux = nn.Sequential(nn.Linear(self.dv.hidden * 2, aux_hidden),
                                     nn.ReLU(), nn.Linear(aux_hidden, 4))

    @staticmethod
    def _shifted_aux_input(h_bi, batch):
        
        idx = torch.arange(h_bi.size(0), device=h_bi.device)
        first = torch.ones_like(batch, dtype=torch.bool)
        first[1:] = batch[1:] != batch[:-1]
        starts = torch.zeros(int(batch.max()) + 1, dtype=torch.long, device=batch.device)
        starts[batch[first]] = idx[first]
        pos = idx - starts[batch]
        lens = torch.zeros_like(starts).scatter_add_(0, batch, torch.ones_like(batch))
        has_left = pos > 0
        has_right = pos < (lens[batch] - 1)
        prev_i = torch.where(has_left, idx - 1, idx)
        next_i = torch.where(has_right, idx + 1, idx)
        h = h_bi.size(-1) // 2
        fwd, bwd = h_bi[:, :h], h_bi[:, h:]
        return torch.cat([fwd[prev_i], bwd[next_i]], dim=-1)

    def forward(self, x, edge_index, edge_type, batch=None, return_feat=False,
                edge_weight=None, return_aux=False):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        parts = []
        h_bi = None
        if self.graph is not None:
            h = x
            for blk in self.graph:
                h = blk(h, edge_index, edge_type, batch, edge_weight)
            parts.append(self.proj_g(h))
        if self.dv is not None:
            x_pad, mask = to_dense_batch(x, batch)
            h_dv, h_bi = self.dv(x_pad, mask)
            parts.append(self.proj_dv(h_dv))
        if len(parts) == 2:
            a, b = parts
            if self.fusion == 'gate':
                g = torch.sigmoid(self.fuse_gate(torch.cat([a, b], dim=-1)))
                z = g * a + (1.0 - g) * b
            elif self.fusion == 'avg':
                z = 0.5 * (a + b)
            else:
                z = self.fuse_proj(torch.cat([a, b], dim=-1))
        else:
            z = parts[0]
        feat = self.head(z)
        logit = self.fc(feat).squeeze(-1)
        if return_aux:
            aux_logits = (self.aux(self._shifted_aux_input(h_bi, batch))
                          if self.aux is not None else None)
            return logit, feat, aux_logits, h_bi
        if return_feat:
            return logit, feat
        return logit
