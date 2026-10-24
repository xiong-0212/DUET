from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pos_weight(labels, floor: float = 3.0):

    y = torch.as_tensor(labels).long().view(-1)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    w = max(neg / max(pos, 1), floor)
    return torch.tensor([w], dtype=torch.float32), pos, neg


class WeightedBCE(nn.Module):
    def __init__(self, pos_weight=None, floor: float = 3.0):
        super().__init__()
        self.registered = pos_weight is not None
        if self.registered:
            self.register_buffer('pos_weight', pos_weight.view(1))
        self.floor = floor

    def set_pos_weight(self, pw):
        if self.registered:
            self.pos_weight = pw.view(1)
        else:
            self.register_buffer('pos_weight', pw.view(1))
            self.registered = True

    def forward(self, logits, target):
        return F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight.to(logits.device))


class FocalLoss(nn.Module):


    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = float(alpha), gamma

    @staticmethod
    def alpha_from_counts(counts):
        n_neg, n_pos = counts
        return 1.0 - n_pos / max(n_neg + n_pos, 1)

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = p * target + (1 - p) * (1 - target)

        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (at * (1 - pt).pow(self.gamma) * ce).mean()


class ClassBalancedFocal(nn.Module):


    def __init__(self, counts=(10000, 4000), gamma=3.0, beta=0.999):
        super().__init__()
        self.gamma, self.beta = gamma, beta
        n = torch.tensor([float(c) for c in counts], dtype=torch.float32)
        en = (1.0 - beta) / (1.0 - torch.pow(beta, n))
        en = en / en.mean()
        self.register_buffer('en', en.view(1, 2))

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = p * target + (1 - p) * (1 - target)

        w = self.en[0, 1] * target + self.en[0, 0] * (1 - target)
        return (w * (1 - pt).pow(self.gamma) * ce).mean()


class TCLFocalLoss(nn.Module):


    def __init__(self, feat_dim=128, num_classes=2, lam=0.2, margin=4.0,
                 gamma=3.0, beta=0.999, counts=(10000, 4000)):
        super().__init__()
        self.lam, self.margin = lam, margin
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.1)
        self.focal = ClassBalancedFocal(counts=counts, gamma=gamma, beta=beta)

    def forward(self, logits, target, feat=None):
        l_focal = self.focal(logits, target)
        if feat is None:
            return l_focal, torch.tensor(0.0, device=logits.device)


        assert self.centers.device == feat.device, (
            f'TCL centers are on {self.centers.device} but feat is on {feat.device}; '
            f'call .to(device) on the whole loss module first')
        c = self.centers
        d = torch.cdist(feat, c)
        y = target.long()
        d_y = d.gather(1, y.view(-1, 1)).squeeze(1)
        d_o = d.gather(1, (1 - y).view(-1, 1)).squeeze(1)
        l_tcl = torch.clamp(d_y + self.margin - d_o, min=0).mean()
        return l_focal + self.lam * l_tcl, l_tcl


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, eps=1e-6):
        super().__init__()
        self.alpha, self.beta, self.eps = alpha, beta, eps

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        tp = (p * target).sum()
        fp = (p * (1 - target)).sum()
        fn = ((1 - p) * target).sum()
        return 1 - (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)


class LDAMLoss(nn.Module):


    def __init__(self, counts=(10000, 4000), max_m=0.5, s=1.0):
        super().__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(counts, dtype=torch.float32)))
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer('m_list', m_list)
        self.s = s

    def forward(self, logits, target):
        y = target.long().view(-1)
        m_neg, m_pos = self.m_list[0], self.m_list[1]
        logit_adj = torch.where(y == 1, logits - m_pos, logits + m_neg)
        return F.binary_cross_entropy_with_logits(logit_adj * self.s, target)


BUILDERS = {
    'bce': lambda counts=None, **kw: WeightedBCE(),

    'focal': lambda counts=None, **kw: FocalLoss(
        alpha=(kw['focal_alpha'] if kw.get('focal_alpha') is not None
               else (FocalLoss.alpha_from_counts(counts) if counts else 0.75)),
        gamma=2.0),

    'tcl_focal': lambda counts=None, **kw: TCLFocalLoss(
        counts=counts or (10000, 4000),
        feat_dim=kw.get('feat_dim', 128), lam=0.2, margin=4.0, gamma=3.0),

    'tversky': lambda counts=None, **kw: TverskyLoss(),

    'ldam': lambda counts=None, **kw: LDAMLoss(counts=counts or (10000, 4000)),
}


def build_loss(name, counts=None, feat_dim=128, focal_alpha=None):
    if name not in BUILDERS:
        raise ValueError(f'unknown loss: {name}, expected one of {sorted(BUILDERS)}')
    return BUILDERS[name](counts=counts, feat_dim=feat_dim, focal_alpha=focal_alpha)
