"""The model. TabICL v2's three table stages, with each row's photo joining the row stage as extra tokens.

    column stage      every column's cells attend to each other (TabICL, pretrained)
    image side        the photo's patches go through a small encoder, then read labels from the context photos (ours)
    row stage         a row's cells and its patches attend to each other and are pooled into one token (TabICL)
    in-context stage  rows attend to rows; test rows read the context labels and score the classes (TabICL)

The image side starts from patch tokens, and the encoder setting says which: "block4" (the chosen one), "block8" or
"block12" are the patch tokens of that block of a frozen DINOv2 ViT-B/14 through one trained linear layer; "cls" is
DINOv2's final CLS and register tokens the same way; "pixels" is ViT's linear patch embedding trained from scratch.

Labels are slots 0..9, and a task's classes may sit on any slots; the output is log-probabilities over the slots
present in the context."""
import json
import os
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

# ---------- TabICL v2's table stages (Qu et al., 2025) and what they expect of a table ----------
# The weights are the released checkpoint from Hugging Face (jingang/TabICL, BSD-3), pretrained on about 82M
# synthetic tables.

REPO = "jingang/TabICL"
CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
WIDTH = 128                      # TabICL's token width; the image side uses the same
N_SLOTS = 10                     # TabICL's class limit: the label slots of a task
TABICL_V2 = {"max_classes": N_SLOTS, "num_quantiles": 999, "embed_dim": WIDTH, "col_num_blocks": 3, "col_nhead": 8, "col_num_inds": 128,
             "col_affine": False, "col_feature_group": "same", "col_feature_group_size": 3, "col_target_aware": True,
             "col_ssmax": "qassmax-mlp-elementwise", "row_num_blocks": 3, "row_nhead": 8, "row_num_cls": 4, "row_rope_base": 100000,
             "row_rope_interleaved": False, "icl_num_blocks": 12, "icl_nhead": 8, "icl_ssmax": "qassmax-mlp-elementwise",
             "ff_factor": 2, "dropout": 0.0, "activation": "gelu", "norm_first": True, "bias_free_ln": False, "recompute": False}
OUTLIER_SIGMAS = 4.0


def load_tabicl(pretrained: bool = True) -> nn.Module:
    """TabICL v2 with its released weights, or the same architecture freshly initialised."""
    from tabicl._model.tabicl import TabICL
    model = TabICL(**TABICL_V2)
    if pretrained:
        from huggingface_hub import hf_hub_download
        state = torch.load(hf_hub_download(REPO, CHECKPOINT), map_location="cpu", weights_only=False)
        assert state["config"] == TABICL_V2, "the checkpoint is not the architecture this module expects"
        model.load_state_dict(state["state_dict"])
    return model


@jaxtyped(typechecker=beartype)
def prepare_table(x: Float[Tensor, "batch rows columns"], n_context: int) -> Float[Tensor, "batch rows columns"]:
    """TabICL's preprocessing, fitted on the context rows of each table (batch, rows, columns): standardise, fill
    missing cells with the mean, clip, then soften values beyond OUTLIER_SIGMAS of the cleaned context with a log."""
    context = x[:, :n_context]
    valid = ~torch.isnan(context)
    count = valid.sum(1, keepdim=True).clamp_min(1)
    mean = torch.where(valid, context, 0.0).sum(1, keepdim=True) / count
    std = (torch.where(valid, (context - mean) ** 2, 0.0).sum(1, keepdim=True) / count).sqrt() + 1e-6
    z = ((x - mean) / std).nan_to_num(0.0).clamp(-100, 100)
    lower, upper = outlier_bounds(z[:, :n_context])
    z = torch.maximum(-torch.log1p(z.abs()) + lower, z)
    return torch.minimum(torch.log1p(z.abs()) + upper, z)


def outlier_bounds(context: Tensor) -> tuple[Tensor, Tensor]:
    """Mean and spread of the context without the values beyond OUTLIER_SIGMAS of a first-pass mean and spread,
    as (batch, 1, columns) bounds."""
    def stats(values: Tensor, keep: Tensor) -> tuple[Tensor, Tensor]:
        n = keep.sum(1, keepdim=True)
        m = torch.where(keep, values, 0.0).sum(1, keepdim=True) / n.clamp_min(1)
        var = torch.where(keep, (values - m) ** 2, 0.0).sum(1, keepdim=True) / (n - 1).clamp_min(1)
        return m, var.sqrt().clamp_min(1e-6)

    mean, std = stats(context, torch.ones_like(context, dtype=torch.bool))
    inlier = (context - mean).abs() <= OUTLIER_SIGMAS * std
    mean, std = stats(context, inlier)
    return mean - OUTLIER_SIGMAS * std, mean + OUTLIER_SIGMAS * std


@jaxtyped(typechecker=beartype)
def contiguous_codes(y_context: Int[Tensor, "batch context"], n_slots: int = N_SLOTS) -> tuple[Int[Tensor, "batch context"], Int[Tensor, "batch slots"]]:
    """TabICL wants classes numbered 0..k-1. Per table, the context labels renumbered in slot order, (batch, n_context),
    and which slot each code stands for, (batch, n_slots) with -1 past the k classes."""
    codes, slots = [], torch.full((len(y_context), n_slots), -1, dtype=torch.long, device=y_context.device)
    for b, y in enumerate(y_context):
        present, inverse = torch.unique(y, return_inverse=True)
        codes.append(inverse)
        slots[b, : len(present)] = present
    return torch.stack(codes), slots


# ---------- attention blocks; every tensor is batch-first: (batch, tokens, width) ----------

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

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "batch queries width"], keys: Float[Tensor, "batch keys width"], values: Float[Tensor, "batch keys width"]) -> Float[Tensor, "batch queries width"]:
        q, k, v = self.split(self.q(x)), self.split(self.k(keys)), self.split(self.v(values))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.o(out.transpose(1, 2).flatten(-2))


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer: self-attention, then an MLP."""

    def __init__(self, width: int, heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, mlp_ratio * width), nn.GELU(), nn.Linear(mlp_ratio * width, width))

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "batch tokens width"]) -> Float[Tensor, "batch tokens width"]:
        h = self.norm_attention(x)
        x = x + self.attention(h, h, h)
        return x + self.mlp(self.norm_mlp(x))


# ---------- the image side before any label is involved: patch tokens, and an encoder over one photo's patches ----------

PATCH = 14                       # pixels per patch side: DINOv2's patch size, and the from-scratch embedding's too
DINO_WIDTH = 768                 # token width of DINOv2 ViT-B/14
DINO_SUMMARY_TOKENS = 5          # DINOv2 ViT-B/14 with registers: CLS + 4 register tokens
ENCODERS = ("block4", "block8", "block12", "cls", "pixels")


class PixelPatches(nn.Module):
    """One token per PATCH x PATCH square of pixels, from a linear layer trained from scratch."""

    def __init__(self, width: int):
        super().__init__()
        self.n_tokens = None                 # one token per patch: the count follows the image size
        self.project = nn.Conv2d(3, width, kernel_size=PATCH, stride=PATCH)

    @jaxtyped(typechecker=beartype)
    def forward(self, images: Float[Tensor, "n 3 h w"]) -> Float[Tensor, "n patches width"]:
        """images (n, 3, H, W), normalised -> (n, patches, width), patches in row-major order."""
        return self.project(images).flatten(2).transpose(1, 2)


class DinoPatches(nn.Module):
    """Tokens of a frozen DINOv2 ViT-B/14 (computed outside the model, see inputs.py) through a layer norm and one
    trained linear layer. By default one token per 14 x 14 patch; with n_tokens, a fixed number of summary tokens
    instead (the final CLS and register tokens)."""

    def __init__(self, width: int, n_tokens: int | None = None):
        super().__init__()
        self.n_tokens = n_tokens
        self.project = nn.Sequential(nn.LayerNorm(DINO_WIDTH), nn.Linear(DINO_WIDTH, width))

    @jaxtyped(typechecker=beartype)
    def forward(self, tokens: Float[Tensor, "n patches dino"]) -> Float[Tensor, "n patches width"]:
        """tokens (n, patches, DINO_WIDTH) -> (n, patches, width)"""
        return self.project(tokens)


class ImageEncoder(nn.Module):
    """A few transformer layers over the patches of one photo. No labels, no other photos."""

    def __init__(self, width: int, heads: int, layers: int, n_patches: int):
        super().__init__()
        self.position = nn.Parameter(torch.randn(n_patches, width) * 0.02)
        self.layers = nn.ModuleList([TransformerLayer(width, heads) for _ in range(layers)])

    @jaxtyped(typechecker=beartype)
    def forward(self, tokens: Float[Tensor, "n patches width"]) -> Float[Tensor, "n patches width"]:
        """tokens (n, patches, width) -> (n, patches, width)"""
        tokens = tokens + self.position
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens


# ---------- the labelled read ----------
# Every patch of every photo attends to the patches of the labelled context photos. Content decides where to look
# (queries and keys are patch tokens); what comes back is the label of the photo a key patch belongs to (values are
# label embeddings). A test photo that looks like context photos labelled 5 reads back "5".
#
# With read="top_k", a photo first retrieves the k context photos whose mean token is most similar to its own, and
# reads from those only. The attention a photo faces is then the same size whatever the context, which is what lets
# a model trained on 256-row tables be scored on 10,000-row ones.

@jaxtyped(typechecker=beartype)
def nearest_photos(tokens: Float[Tensor, "batch photos patches width"], sources: Float[Tensor, "batch sources patches width"], k: int) -> Int[Tensor, "batch photos k"]:
    """For every photo, the k source photos whose mean patch token is most similar (cosine) to its own.
    tokens (batch, photos, patches, width), sources (batch, S, patches, width) -> (batch, photos, k) indices into S."""
    similarity = F.normalize(tokens.mean(2), dim=-1) @ F.normalize(sources.mean(2), dim=-1).transpose(1, 2)
    return similarity.topk(k, dim=-1).indices


class LabelRead(nn.Module):
    """One cross-attention from reading patches to source patches, with the sources' labels as the values."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm_readers = nn.LayerNorm(width)
        self.norm_sources = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.chunk = 32                      # reading photos per attention call in the top-k branch

    @jaxtyped(typechecker=beartype)
    def forward(self, readers: Float[Tensor, "batch photos patches width"], sources: Float[Tensor, "batch sources patches width"],
                source_labels: Float[Tensor, "batch sources width"], nearest: Int[Tensor, "batch photos k"] | None = None) -> Float[Tensor, "batch photos patches width"]:
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

    @jaxtyped(typechecker=beartype)
    def forward(self, tokens: Float[Tensor, "batch photos patches width"], y_context: Int[Tensor, "batch context"]) -> Float[Tensor, "batch photos patches width"]:
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


# ---------- the model ----------

@dataclass(frozen=True)
class ModelConfig:
    encoder: str = "block4"       # one of ENCODERS: where the patch tokens come from
    image_size: int = 224         # 224 / PATCH = 16, so 256 patches per photo
    encoder_layers: int = 4       # transformer layers over the patches of one photo
    patch_layers: int = 2         # label reads from the context photos
    heads: int = 4
    max_sources: int = 256        # context photos read from; raise it at inference for longer contexts
    patch_read: str = "top_k"     # "top_k": read from the nearest context photos; "all": from every one
    patch_top_k: int = 32


def dino_block(encoder: str) -> int | None:
    """The frozen DINOv2 block an encoder takes its patch tokens from: 4, 8 or 12 for blockN, None otherwise."""
    return int(encoder[len("block"):]) if encoder.startswith("block") else None


def patch_embedder(encoder: str, width: int) -> nn.Module:
    assert encoder in ENCODERS, f"unknown encoder {encoder!r}: one of {ENCODERS}"
    if encoder == "pixels":
        return PixelPatches(width)
    if encoder == "cls":
        return DinoPatches(width, n_tokens=DINO_SUMMARY_TOKENS)
    return DinoPatches(width)


class TableImageICL(nn.Module):
    """TabICL v2's table stages with the image side of cfg.encoder in the rows. Without photos it is TabICL v2 on the
    table."""

    def __init__(self, cfg: ModelConfig = ModelConfig(), pretrained: bool = True):
        super().__init__()
        self.cfg = cfg
        self.table = load_tabicl(pretrained)
        self.embed = patch_embedder(cfg.encoder, WIDTH)
        n_patches = self.embed.n_tokens or (cfg.image_size // PATCH) ** 2
        self.encoder = ImageEncoder(WIDTH, cfg.heads, cfg.encoder_layers, n_patches)
        self.read = LabelledRead(WIDTH, cfg.heads, cfg.patch_layers, N_SLOTS, cfg.max_sources, cfg.patch_read, cfg.patch_top_k)
        self.patch_marker = nn.Parameter(torch.randn(WIDTH) * 0.02)       # tells the row stage a token is a patch, not a cell

    def load_weights(self, state: dict) -> None:
        self.load_state_dict(state)

    @jaxtyped(typechecker=beartype)
    def photo_tokens(self, images: Float[Tensor, "batch rows *photo"], y_context: Int[Tensor, "batch context"]) -> Float[Tensor, "batch rows patches width"]:
        """images: (batch, rows, 3, H, W) for encoder pixels, (batch, rows, tokens, DINO_WIDTH) for the DINOv2 encoders.
        Returns each row's patch tokens after the encoder and the labelled read: (batch, rows, patches, width)."""
        batch, rows = images.shape[:2]
        tokens = self.encoder(self.embed(images.flatten(0, 1))).unflatten(0, (batch, rows))
        return self.read(tokens, y_context) + self.patch_marker

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "batch rows columns"], y_context: Int[Tensor, "batch context"],
                images: Float[Tensor, "batch rows *photo"] | None = None) -> Float[Tensor, "batch test slots"]:
        """x (batch, rows, columns), context rows first, NaN = missing; y_context (batch, n_context) label slots.
        Returns log-probabilities over the label slots for the test rows: (batch, rows - n_context, N_SLOTS)."""
        n_context = y_context.shape[1]
        codes, slots = contiguous_codes(y_context)
        tokens = self.table.col_embedder._train_forward(prepare_table(x, n_context), codes, None, False)   # (batch, rows, cls + columns, width)
        if images is not None:
            tokens = torch.cat([tokens, self.photo_tokens(images, y_context)], 2)
        rows = self.table.row_interactor._train_forward(tokens, None)
        logits = self.table.icl_predictor._icl_predictions(rows, codes)[:, n_context:]                    # over the codes
        out = logits.new_full((len(x), logits.shape[1], N_SLOTS), float("-inf"))
        for b in range(len(x)):
            k = int((slots[b] >= 0).sum())
            out[b][:, slots[b, :k]] = logits[b][:, :k]
        return out.log_softmax(-1)


IMAGE_SIDE = ("embed.", "encoder.", "read.", "patch_marker")       # the parameters trained from scratch


def config_from_record(record: dict) -> ModelConfig:
    """The model config of a run record. A record written by scripts/train.py carries "encoder" among its model
    settings; the records under results/ are older and name the encoder next to them as an arm: "pixels", "dino" with
    the block in "dino_block", or "dino_cls"."""
    settings = {f.name: record["model"][f.name] for f in fields(ModelConfig) if f.name in record["model"]}
    if "arm" in record:
        settings["encoder"] = {"pixels": "pixels", "dino_cls": "cls", "dino": f"block{record['dino_block']}"}[record["arm"]]
    return ModelConfig(**settings)


def load_run(folder: str, device: str = "cpu") -> tuple[TableImageICL, dict]:
    """A trained run: its config.json and model.pt. Returns the model in eval mode and the record."""
    with open(os.path.join(folder, "config.json")) as f:
        record = json.load(f)
    model = TableImageICL(config_from_record(record), pretrained=False)
    model.load_weights(torch.load(os.path.join(folder, "model.pt"), map_location="cpu", weights_only=True))
    return model.to(device).eval(), record
