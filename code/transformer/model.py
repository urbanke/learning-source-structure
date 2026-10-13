"""A small decoder-only transformer (GPT-2 style: pre-layer-norm blocks, causal multi-head attention, MLP with GELU,
learned positional embeddings). Options: number of layers and heads, width, attention-only blocks (no MLP)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head; self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.manual = False

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if not self.manual:
            try:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
            except (RuntimeError, NotImplementedError):
                self.manual = True          # fall back to explicit attention (some backends lack the fused kernel)
        if self.manual:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(k.shape[-1])
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model, n_head, mlp=True, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_head, dropout)
        self.mlp = None
        if mlp:
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, n_ctx, n_layer=2, n_head=4, d_model=128, mlp=True, dropout=0.0):
        super().__init__()
        self.cfg = dict(vocab=vocab, n_ctx=n_ctx, n_layer=n_layer, n_head=n_head, d_model=d_model, mlp=mlp, dropout=dropout)
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(n_ctx, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, mlp, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.apply(self._init)
        for name, p in self.named_parameters():          # GPT-2 scaling of the residual projections
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        """idx: (B, T) tokens. Returns logits (B, T, vocab) for the next token at every position."""
        B, T = idx.shape
        assert T <= self.cfg["n_ctx"], "sequence longer than the context"
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
