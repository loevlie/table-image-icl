"""Attention blocks. Every tensor is batch-first: (batch, tokens, width)."""
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Attention(nn.Module):
    """Multi-head attention with separate inputs for queries, keys and values."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.o = nn.Linear(width, width)

    def split(self, t: Tensor) -> Tensor:
        """(batch, tokens, width) -> (batch, heads, tokens, width / heads)"""
        return t.unflatten(-1, (self.heads, -1)).transpose(1, 2)

    def forward(self, x: Tensor, keys: Tensor, values: Tensor, key_mask: Tensor | None = None) -> Tensor:
        """key_mask: (batch, keys) with False on keys nobody may attend to, or (batch, queries, keys) with the keys
        each query may attend to."""
        mask = None
        if key_mask is not None:
            mask = key_mask[:, None, None, :] if key_mask.dim() == 2 else key_mask[:, None]
        q, k, v = self.split(self.q(x)), self.split(self.k(keys)), self.split(self.v(values))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).flatten(-2))


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer: x attends to `source` (itself by default), then an MLP."""

    def __init__(self, width: int, heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, mlp_ratio * width), nn.GELU(), nn.Linear(mlp_ratio * width, width))

    def forward(self, x: Tensor, source: Tensor | None = None, source_mask: Tensor | None = None) -> Tensor:
        source = self.norm_attention(x if source is None else source)
        x = x + self.attention(self.norm_attention(x), source, source, key_mask=source_mask)
        return x + self.mlp(self.norm_mlp(x))
