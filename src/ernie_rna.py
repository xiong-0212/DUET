from __future__ import annotations

import math

import torch
import torch.nn as nn

BASES = {'A': 6, 'C': 7, 'G': 8, 'U': 9, 'T': 9}

PAIR_WEIGHTS = {('A', 'U'): 2.0, ('U', 'A'): 2.0,
                ('C', 'G'): 3.0, ('G', 'C'): 3.0,
                ('G', 'U'): 0.8, ('U', 'G'): 0.8}
VOCAB_INV = {v: k for k, v in BASES.items()}


def pairing_matrix(input_ids: torch.Tensor) -> torch.Tensor:
    
    B, T = input_ids.shape
    ids = input_ids.cpu().numpy()
    m = [[[0.0] * T for _ in range(T)] for _ in range(B)]
    for b in range(B):
        for i in range(T):
            ci = VOCAB_INV.get(int(ids[b][i]))
            if ci is None:
                continue
            for j in range(T):
                if i == j:
                    continue
                cj = VOCAB_INV.get(int(ids[b][j]))
                if cj is None:
                    continue
                m[b][i][j] = PAIR_WEIGHTS.get((ci, cj), 0.0)
    return torch.tensor(m, dtype=torch.float32,
                        device=input_ids.device).unsqueeze(-1)


class ErnieEmbeddings(nn.Module):
    def __init__(self, vocab_size=26, hidden=768, max_pos=1026, type_vocab=2, eps=1e-12):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(type_vocab, hidden)
        self.layer_norm = nn.LayerNorm(hidden, eps=eps)
        self.register_buffer('position_embeddings',
                             self._sinusoidal(max_pos, hidden), persistent=False)
        self.hidden = hidden

    @staticmethod
    def _sinusoidal(max_len, d):
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, input_ids, token_type_ids=None):
        T = input_ids.size(1)
        x = self.word_embeddings(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        x = x + self.token_type_embeddings(token_type_ids)
        x = x + self.position_embeddings[:T].unsqueeze(0)
        return self.layer_norm(x)


class ErnieSelfAttention(nn.Module):
    def __init__(self, hidden=768, heads=12):
        super().__init__()
        self.num_attention_heads = heads
        self.attention_head_size = hidden // heads
        self.all_head_size = hidden
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)

    def _split(self, x, B, T):
        return x.view(B, T, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

    def forward(self, hidden, attention_mask=None, bias=None):
        B, T, _ = hidden.shape
        q = self._split(self.query(hidden), B, T)
        k = self._split(self.key(hidden), B, T)
        v = self._split(self.value(hidden), B, T)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        if bias is not None:
            scores = scores + bias
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = nn.functional.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v).transpose(1, 2).reshape(B, T, self.all_head_size)
        return ctx, scores, probs


class ErnieSelfOutput(nn.Module):
    def __init__(self, hidden=768, eps=1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.layer_norm = nn.LayerNorm(hidden, eps=eps)

    def forward(self, hidden, input_tensor):
        return self.layer_norm(self.dense(hidden) + input_tensor)


class ErnieAttention(nn.Module):
    def __init__(self, hidden=768, heads=12, eps=1e-12):
        super().__init__()
        self.self = ErnieSelfAttention(hidden, heads)
        self.output = ErnieSelfOutput(hidden, eps)


class ErnieIntermediate(nn.Module):
    def __init__(self, hidden=768, inter=3072):
        super().__init__()
        self.dense = nn.Linear(hidden, inter)

    def forward(self, x):
        return nn.functional.gelu(self.dense(x))


class ErnieOutput(nn.Module):
    def __init__(self, hidden=768, inter=3072, eps=1e-12):
        super().__init__()
        self.dense = nn.Linear(inter, hidden)
        self.layer_norm = nn.LayerNorm(hidden, eps=eps)

    def forward(self, x, input_tensor):
        return self.layer_norm(self.dense(x) + input_tensor)


class ErnieLayer(nn.Module):
    def __init__(self, hidden=768, heads=12, inter=3072, eps=1e-12):
        super().__init__()
        self.attention = ErnieAttention(hidden, heads, eps)
        self.intermediate = ErnieIntermediate(hidden, inter)
        self.output = ErnieOutput(hidden, inter, eps)

    def forward(self, x, attention_mask=None, bias=None):
        ctx, scores, probs = self.attention.self(x, attention_mask, bias)
        h = self.attention.output(ctx, x)
        return self.output(self.intermediate(h), h), scores, probs


class ErnieEncoder(nn.Module):
    def __init__(self, layers=12, hidden=768, heads=12, inter=3072, eps=1e-12):
        super().__init__()
        self.layer = nn.ModuleList([ErnieLayer(hidden, heads, inter, eps)
                                    for _ in range(layers)])


class ErnieModel(nn.Module):
    

    def __init__(self, vocab_size=26, hidden=768, layers=12, heads=12, inter=3072,
                 max_pos=1026, type_vocab=2, eps=1e-12, bias_mode='recurrent'):
        super().__init__()
        self.bias_mode = bias_mode
        self.embeddings = ErnieEmbeddings(vocab_size, hidden, max_pos, type_vocab, eps)
        self.encoder = ErnieEncoder(layers, hidden, heads, inter, eps)
        self.pairwise_bias_proj = nn.Sequential(
            nn.Linear(1, 6), nn.GELU(), nn.Linear(6, heads))
        self.pooler = nn.Linear(hidden, hidden)

    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings(input_ids)
        B, T, _ = x.shape
        am = None
        if attention_mask is not None:
            am = (1.0 - attention_mask[:, None, None, :]) * -1e9
        pair = pairing_matrix(input_ids)
        cur_bias_in = pair
        prev_scores = None
        for li, layer in enumerate(self.encoder.layer):
            if li == 0:
                bias_in = pair
            elif self.bias_mode == 'pair_only':
                bias_in = None
            elif self.bias_mode == 'fixed':
                bias_in = pair
            else:
                bias_in = prev_scores
            if bias_in is not None:
                bias = self.pairwise_bias_proj(bias_in).permute(0, 3, 1, 2)
            else:
                bias = None
            x, scores, probs = layer(x, am, bias)

            if self.bias_mode == 'recurrent':
                prev_scores = probs.mean(dim=1).unsqueeze(-1)
            else:
                prev_scores = scores.mean(dim=1).unsqueeze(-1)
        return x
