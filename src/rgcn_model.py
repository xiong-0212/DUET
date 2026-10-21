from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GraphNorm


def make_norm(kind: str, dim: int):
    if kind == 'graph':
        return GraphNorm(dim)
    if kind == 'layer':
        return nn.LayerNorm(dim)
    if kind == 'none':
        return nn.Identity()
    raise ValueError(kind)


class RGCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, norm='graph',
                 dropout=0.4, residual=True, num_bases=None, wnorm='wsum'):
        super().__init__()
        self.conv = RGCNConv(in_dim, out_dim, num_relations, num_bases=num_bases)
        self.norm = make_norm(norm, out_dim)
        self.use_batch = (norm == 'graph')
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.wnorm = wnorm
        self.proj = (nn.Linear(in_dim, out_dim) if (residual and in_dim != out_dim)
                     else None)

    def _weighted_conv(self, x, edge_index, edge_type, edge_weight):
        
        conv = self.conv
        weight = conv.weight
        out = x.new_zeros(x.size(0), conv.out_channels)
        for r in range(conv.num_relations):
            mask = edge_type == r
            if not bool(mask.any()):
                continue
            src, dst = edge_index[0, mask], edge_index[1, mask]
            w = edge_weight[mask]
            msg = x[src] @ weight[r]
            if self.wnorm == 'wsum':
                den = x.new_zeros(x.size(0)).index_add_(0, dst, w).clamp_(min=1e-8)
                msg = msg / den[dst].unsqueeze(-1)
            else:
                deg = x.new_zeros(x.size(0)).index_add_(
                    0, dst, torch.ones_like(w))
                msg = msg * (w / deg[dst].clamp_(min=1.0)).unsqueeze(-1)
            out.index_add_(0, dst, msg)
        if conv.root is not None:
            out = out + x @ conv.root
        if conv.bias is not None:
            out = out + conv.bias
        return out

    def forward(self, x, edge_index, edge_type, batch=None, edge_weight=None):
        if edge_weight is None:
            h = self.conv(x, edge_index, edge_type)
        else:
            h = self._weighted_conv(x, edge_index, edge_type, edge_weight)
        h = self.norm(h, batch) if (self.use_batch and batch is not None) else self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        if self.residual:
            r = x if self.proj is None else self.proj(x)
            h = h + r
        return h


class RGCNBindingSiteModel(nn.Module):
    

    def __init__(self, in_dim=640, hidden=128, num_relations=3, num_layers=2,
                 dropout=0.4, norm='graph', residual=True, num_bases=None,
                 head_hidden=None, input_skip=False, wnorm='wsum'):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.num_layers = num_layers
        self.input_skip = input_skip

        blocks = []
        for i in range(num_layers):
            fin = in_dim if i == 0 else hidden
            blocks.append(RGCNBlock(fin, hidden, num_relations, norm=norm,
                                    dropout=dropout, residual=residual,
                                    num_bases=num_bases, wnorm=wnorm))
        self.blocks = nn.ModuleList(blocks)

        hh = head_hidden or hidden // 2
        self.feat_dim = hh
        head_in = hidden + (in_dim if input_skip else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, hh),
            nn.LayerNorm(hh),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(hh, 1)

    def encode(self, x, edge_index, edge_type, batch=None, edge_weight=None):
        for blk in self.blocks:
            x = blk(x, edge_index, edge_type, batch, edge_weight)
        return x

    def forward(self, x, edge_index, edge_type, batch=None, return_feat=False,
                edge_weight=None):
        raw = x
        h = self.encode(x, edge_index, edge_type, batch, edge_weight)
        if self.input_skip:
            h = torch.cat([h, raw], dim=-1)
        feat = self.head(h)
        logit = self.fc(feat).squeeze(-1)
        if return_feat:
            return logit, feat
        return logit


class MLPOnly(nn.Module):
    

    def __init__(self, in_dim=640, hidden=128, dropout=0.4, head_hidden=None):
        super().__init__()
        hh = head_hidden or hidden // 2
        self.feat_dim = hh
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hh),
            nn.LayerNorm(hh),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(hh, 1)

    def forward(self, x, edge_index=None, edge_type=None, batch=None, return_feat=False,
                edge_weight=None):
        feat = self.net(x)
        logit = self.fc(feat).squeeze(-1)
        if return_feat:
            return logit, feat
        return logit


def build_model(cfg, num_relations):
    arch = cfg.get('arch', 'rgcn')
    if arch == 'dense':

        from densenet_model import DenseSeqTower
        return DenseSeqTower(
            in_dim=cfg['in_dim'], split_dim=cfg.get('split_dim', 0),
            stem=cfg.get('stem', 256), growth=cfg.get('growth', 72),
            layers=tuple(cfg.get('block_layers', (4, 4, 4))), k=cfg.get('kernel_size', 3),
            dropout=cfg.get('mlp_dropout', 0.1), head_hidden=cfg.get('head_hidden') or 64,
            head_dropout=cfg.get('dropout', 0.4))
    if arch in ('tcn', 'transformer'):

        from seq_towers import TCNSeqTower, TransformerSeqTower
        cls = TCNSeqTower if arch == 'tcn' else TransformerSeqTower
        kw = {k: v for k, v in (('ch', cfg.get('ch')), ('d', cfg.get('d')),
                                ('ffn', cfg.get('ffn')), ('heads', cfg.get('heads')),
                                ('layers', cfg.get('seq_layers'))) if v}
        return cls(in_dim=cfg['in_dim'], split_dim=cfg.get('split_dim', 0),
                   dropout=cfg.get('mlp_dropout', 0.1), head_hidden=cfg.get('head_hidden') or 64,
                   head_dropout=cfg.get('dropout', 0.4), **kw)
    if arch == 'mlp':
        return MLPOnly(in_dim=cfg['in_dim'], hidden=cfg['hidden'],
                       dropout=cfg['dropout'], head_hidden=cfg.get('head_hidden'))
    if arch in ('dual2', 'dual2_avg', 'dual2_logit', 'dual2_graph', 'dual2_mlp'):

        from fusion_model import DualBackboneModel
        fusion = {'dual2': 'gate', 'dual2_avg': 'avg',
                  'dual2_logit': 'logit_avg'}.get(arch, 'gate')
        return DualBackboneModel(
            in_dim=cfg['in_dim'], split_dim=cfg.get('split_dim', 640),
            hidden=cfg['hidden'], num_relations=num_relations,
            num_layers=cfg['num_layers'], dropout=cfg['dropout'],
            norm=cfg.get('norm', 'graph'), residual=cfg.get('residual', True),
            num_bases=cfg.get('num_bases'), head_hidden=cfg.get('head_hidden'),
            fusion=fusion, mlp_d=cfg.get('mlp_d', 128),
            mlp_dropout=cfg.get('mlp_dropout', 0.1),
            mlp_widths=cfg.get('mlp_widths'),
            use_graph=arch != 'dual2_mlp', use_mlp=arch != 'dual2_graph',
        )
    if arch in ('fusion', 'fusion_avg', 'fusion_concat', 'dualview', 'dualview_nogate',
                'fusion_mamba', 'dualview_mamba',
                'fusion_mamba_official', 'dualview_mamba_official'):

        from fusion_model import FusionModel
        fusion = {'fusion_avg': 'avg', 'fusion_concat': 'concat'}.get(arch, 'gate')
        seq_only = arch in ('dualview', 'dualview_nogate', 'dualview_mamba',
                            'dualview_mamba_official')
        return FusionModel(
            in_dim=cfg['in_dim'], hidden=cfg['hidden'], num_relations=num_relations,
            num_layers=cfg['num_layers'], dropout=cfg['dropout'],
            norm=cfg.get('norm', 'graph'), residual=cfg.get('residual', True),
            num_bases=cfg.get('num_bases'), head_hidden=cfg.get('head_hidden'),
            fusion=fusion, dv_hidden=cfg.get('dv_hidden', cfg['hidden']),
            dv_layers=cfg.get('dv_layers', 2), dv_proj=cfg.get('dv_proj', 128),
            dv_gate=arch != 'dualview_nogate',
            dv_seq_block=('mamba_official' if arch.endswith('mamba_official') else
                          ('mamba' if arch.endswith('mamba') else 'transformer')),
            use_graph=not seq_only, use_seq=True,
        )
    return RGCNBindingSiteModel(
        in_dim=cfg['in_dim'], hidden=cfg['hidden'], num_relations=num_relations,
        num_layers=cfg['num_layers'], dropout=cfg['dropout'],
        norm=cfg.get('norm', 'graph'), residual=cfg.get('residual', True),
        num_bases=cfg.get('num_bases'), head_hidden=cfg.get('head_hidden'),
        input_skip=cfg.get('input_skip', False), wnorm=cfg.get('wnorm', 'wsum'),
    )
