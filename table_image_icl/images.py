"""The image side before any label is involved: a photo becomes patch tokens, and a small encoder mixes the patches
of one photo. Two kinds of patch token: raw pixels through one linear layer (ViT's patch embedding, trained from
scratch), or the tokens of a frozen DINOv2 block through one trained linear layer."""
import torch
import torch.nn as nn
from torch import Tensor

from table_image_icl.attention import TransformerLayer

DINO_SUMMARY_TOKENS = 5          # DINOv2 ViT-B/14 with registers: CLS + 4 register tokens


class PixelPatches(nn.Module):
    """One token per patch_size x patch_size square of pixels, from a linear layer trained from scratch."""

    def __init__(self, width: int, patch_size: int):
        super().__init__()
        self.stride = patch_size
        self.n_tokens = None                 # one token per patch: the count follows the image size
        self.project = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: Tensor) -> Tensor:
        """images (n, 3, H, W), normalised -> (n, patches, width), patches in row-major order."""
        return self.project(images).flatten(2).transpose(1, 2)


class DinoPatches(nn.Module):
    """Tokens of a frozen DINOv2 ViT-B/14 (computed outside the model, see inputs.py) through a layer norm and one
    trained linear layer. By default one token per 14 x 14 patch; with n_tokens, a fixed number of summary tokens
    instead (the final CLS and register tokens)."""

    def __init__(self, width: int, dino_width: int, n_tokens: int | None = None):
        super().__init__()
        self.stride = 14
        self.n_tokens = n_tokens
        self.project = nn.Sequential(nn.LayerNorm(dino_width), nn.Linear(dino_width, width))

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens (n, patches, dino_width) -> (n, patches, width)"""
        return self.project(tokens)


class ImageEncoder(nn.Module):
    """A few transformer layers over the patches of one photo. No labels, no other photos."""

    def __init__(self, width: int, heads: int, layers: int, n_patches: int):
        super().__init__()
        self.position = nn.Parameter(torch.randn(n_patches, width) * 0.02)
        self.layers = nn.ModuleList([TransformerLayer(width, heads) for _ in range(layers)])

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens (n, patches, width) -> (n, patches, width)"""
        tokens = tokens + self.position
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens


def patch_embedder(arm: str, width: int, patch_size: int, dino_width: int) -> nn.Module:
    """The embedder of an arm: "pixels", "dino" (block tokens) or "dino_cls" (summary tokens)."""
    if arm == "pixels":
        return PixelPatches(width, patch_size)
    if arm == "dino":
        return DinoPatches(width, dino_width)
    if arm == "dino_cls":
        return DinoPatches(width, dino_width, n_tokens=DINO_SUMMARY_TOKENS)
    raise ValueError(f"unknown arm {arm!r}")
