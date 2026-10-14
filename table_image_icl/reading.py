"""The labelled read: every patch of every photo attends to the patches of the labelled context photos. Content
decides where to look (queries and keys are patch tokens); what comes back is the label of the photo a key patch
belongs to (values are label embeddings). A test photo that looks like context photos labelled 5 reads back "5".

With read="top_k", a photo first retrieves the k context photos whose mean token is most similar to its own, and
reads from those only. The attention a photo faces is then the same size whatever the context, which is what lets a
model trained on 256-row tables be scored on 10,000-row ones."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from table_image_icl.attention import Attention


def nearest_photos(tokens: Tensor, sources: Tensor, k: int) -> Tensor:
    """For every photo, the k source photos whose mean patch token is most similar (cosine) to its own.
    tokens (batch, photos, patches, width), sources (batch, S, patches, width) -> (batch, photos, k) indices into S."""
    similarity = F.normalize(tokens.mean(2), dim=-1) @ F.normalize(sources.mean(2), dim=-1).transpose(1, 2)
    return similarity.topk(k, dim=-1).indices


def neighbour_mask(nearest: Tensor, n_sources: int) -> Tensor:
    """nearest (batch, photos, k) -> (batch, photos, S), True where source j is among photo i's neighbours."""
    allowed = torch.zeros(*nearest.shape[:2], n_sources, dtype=torch.bool, device=nearest.device)
    return allowed.scatter(2, nearest, True)


class LabelRead(nn.Module):
    """One cross-attention from reading patches to source patches, with the sources' labels as the values."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm_readers = nn.LayerNorm(width)
        self.norm_sources = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.chunk = 32                      # reading photos per attention call in the top-k branch

    def forward(self, readers: Tensor, sources: Tensor, source_labels: Tensor, nearest: Tensor | None = None) -> Tensor:
        """readers (batch, photos, patches, width); sources (batch, S, patches, width); source_labels (batch, S, width).
        Without `nearest`, every patch attends to every source patch. With `nearest` (batch, photos, k), each photo
        attends only to the patches of its k neighbours. Returns what each reading patch read: the shape of readers."""
        batch, photos, patches, width = readers.shape
        queries, keys = self.norm_readers(readers), self.norm_sources(sources)
        if nearest is None:
            values = source_labels[:, :, None].expand(-1, -1, patches, -1).flatten(1, 2)
            read = self.attention(queries.flatten(1, 2), keys.flatten(1, 2), values)
            return read.reshape(batch, photos, patches, width)

        rows = torch.arange(batch, device=readers.device)[:, None, None]
        out = []
        for start in range(0, photos, self.chunk):
            idx = nearest[:, start:start + self.chunk]                                       # (batch, c, k)
            chunk_keys = keys[rows, idx].flatten(0, 1).flatten(1, 2)                         # (batch * c, k * patches, width)
            chunk_values = source_labels[rows, idx].flatten(0, 1)[:, :, None].expand(-1, -1, patches, -1).flatten(1, 2)
            chunk_queries = queries[:, start:start + self.chunk].flatten(0, 1)               # (batch * c, patches, width)
            read = self.attention(chunk_queries, chunk_keys, chunk_values)
            out.append(read.reshape(batch, -1, patches, width))
        return torch.cat(out, 1)

    def forward_masked(self, readers: Tensor, sources: Tensor, source_labels: Tensor, nearest: Tensor) -> Tensor:
        """The top-k read written as one attention over every source patch, with a key allowed only when its photo is
        among the reader's neighbours. The same function as forward with `nearest`: a softmax with -inf outside the
        allowed keys is the softmax over the allowed keys. The mask is dense here, so this is a reference, not the
        way to train; a block-sparse kernel over neighbour_mask would make it the memory-cheap way."""
        batch, photos, patches, width = readers.shape
        n_sources = sources.shape[1]
        queries, keys = self.norm_readers(readers), self.norm_sources(sources)
        values = source_labels[:, :, None].expand(-1, -1, patches, -1).flatten(1, 2)
        allowed = neighbour_mask(nearest, n_sources)
        mask = allowed[:, :, None, :, None].expand(-1, -1, patches, -1, patches).reshape(batch, photos * patches, n_sources * patches)
        read = self.attention(queries.flatten(1, 2), keys.flatten(1, 2), values, key_mask=mask)
        return read.reshape(batch, photos, patches, width)


class LabelledRead(nn.Module):
    """A stack of label reads, each followed by an MLP, over the photos of one table. The context photos are the
    sources; a context photo is among its own sources, since the context is given, labels included. Test photos
    only read."""

    def __init__(self, width: int, heads: int, layers: int, n_slots: int, max_sources: int, read: str = "top_k", top_k: int = 32):
        super().__init__()
        assert read in ("all", "top_k"), read
        self.max_sources, self.read, self.top_k = max_sources, read, top_k
        self.label = nn.Embedding(n_slots, width)
        nn.init.orthogonal_(self.label.weight)
        self.reads = nn.ModuleList([LabelRead(width, heads) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)) for _ in range(layers)])

    def pick_sources(self, n_context: int, device) -> Tensor:
        """Which context photos are sources: all of them up to max_sources; beyond that, a fresh random subset at
        every training step and the first max_sources otherwise."""
        if n_context > self.max_sources and self.training:
            return torch.randperm(n_context, device=device)[: self.max_sources]
        return torch.arange(min(n_context, self.max_sources), device=device)

    def forward(self, tokens: Tensor, y_context: Tensor) -> Tensor:
        """tokens (batch, photos, patches, width), context photos first; y_context (batch, n_context) label slots.
        Returns the tokens with what they read added: the same shape."""
        picked = self.pick_sources(y_context.shape[1], tokens.device)
        sources = tokens[:, picked]
        source_labels = self.label(y_context[:, picked])
        nearest = None
        if self.read == "top_k" and self.top_k < len(picked):
            nearest = nearest_photos(tokens, sources, self.top_k)
        h = tokens
        for read, norm, mlp in zip(self.reads, self.norms, self.mlps):
            h = h + read(h, sources, source_labels, nearest)
            h = h + mlp(norm(h))
        return h
