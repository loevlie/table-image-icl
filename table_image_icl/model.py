"""The model. TabICL v2's three table stages, with each row's photo joining the row stage as extra tokens.

    column stage      every column's cells attend to each other (TabICL, pretrained)
    image side        the photo's patches go through a small encoder, then read labels from the context photos (ours)
    row stage         a row's cells and its patches attend to each other and are pooled into one token (TabICL)
    in-context stage  rows attend to rows; test rows read the context labels and score the classes (TabICL)

Labels are slots 0..9, and a task's classes may sit on any slots; the output is log-probabilities over the slots
present in the context."""
import json
import os
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
from torch import Tensor

from table_image_icl.images import ImageEncoder, patch_embedder
from table_image_icl.reading import LabelledRead
from table_image_icl.tables import N_SLOTS, WIDTH, contiguous_codes, load_tabicl, prepare_table


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 224
    patch_size: int = 14          # arm pixels: 16 x 16 = 256 patches per photo (DINOv2's patch size too)
    encoder_layers: int = 4       # transformer layers over the patches of one photo
    patch_layers: int = 2         # label reads from the context photos
    heads: int = 4
    max_sources: int = 256        # context photos read from; raise it at inference for longer contexts
    patch_read: str = "top_k"     # "top_k": read from the nearest context photos; "all": from every one
    patch_top_k: int = 32
    dino_width: int = 768         # token width of DINOv2 ViT-B/14


class TableImageICL(nn.Module):
    """arm: "pixels" (raw pixels, trained from scratch), "dino" (frozen DINOv2 block tokens) or "dino_cls"
    (frozen DINOv2 CLS and register tokens). Without photos it is TabICL v2 on the table."""

    def __init__(self, cfg: ModelConfig = ModelConfig(), arm: str = "dino", pretrained: bool = True):
        super().__init__()
        self.cfg, self.arm = cfg, arm
        self.table = load_tabicl(pretrained)
        self.embed = patch_embedder(arm, WIDTH, cfg.patch_size, cfg.dino_width)
        n_patches = self.embed.n_tokens or (cfg.image_size // self.embed.stride) ** 2
        self.encoder = ImageEncoder(WIDTH, cfg.heads, cfg.encoder_layers, n_patches)
        self.read = LabelledRead(WIDTH, cfg.heads, cfg.patch_layers, N_SLOTS, cfg.max_sources, cfg.patch_read, cfg.patch_top_k)
        self.patch_marker = nn.Parameter(torch.randn(WIDTH) * 0.02)       # tells the row stage a token is a patch, not a cell

    def load_weights(self, state: dict) -> None:
        self.load_state_dict(state)

    def photo_tokens(self, images: Tensor, y_context: Tensor) -> Tensor:
        """images: (batch, rows, 3, H, W) for arm pixels, (batch, rows, tokens, dino_width) for the DINOv2 arms.
        Returns each row's patch tokens after the encoder and the labelled read: (batch, rows, patches, width)."""
        batch, rows = images.shape[:2]
        tokens = self.encoder(self.embed(images.flatten(0, 1))).unflatten(0, (batch, rows))
        return self.read(tokens, y_context) + self.patch_marker

    def forward(self, x: Tensor, y_context: Tensor, images: Tensor | None = None) -> Tensor:
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
    return ModelConfig(**{f.name: record["model"][f.name] for f in fields(ModelConfig) if f.name in record["model"]})


def load_run(folder: str, device: str = "cpu") -> tuple[TableImageICL, dict]:
    """A trained run: its config.json and model.pt. Returns the model in eval mode and the record."""
    with open(os.path.join(folder, "config.json")) as f:
        record = json.load(f)
    model = TableImageICL(config_from_record(record), record["arm"], pretrained=False)
    model.load_weights(torch.load(os.path.join(folder, "model.pt"), map_location="cpu", weights_only=True))
    return model.to(device).eval(), record
