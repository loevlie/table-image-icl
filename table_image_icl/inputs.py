"""From stored uint8 photos (n, H, W, 3) to what each arm's patch embedder takes. Nothing here is trained.
The ImageNet mean and spread are torchvision's; DINOv2 comes from its hub repository, pinned to one commit."""
import numpy as np
import torch
from PIL import Image
from torch import Tensor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINOV2 = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_WEIGHTS = "dinov2_vitb14_reg"              # ViT-B/14 with registers
PAD_COLOUR = (124, 116, 104)                      # the ImageNet mean colour: zero after normalisation


def letterbox(image, size: int = 224) -> np.ndarray:
    """A PIL image or a path -> padded to a square (never stretched) and resized to size x size, uint8 (H, W, 3)."""
    if isinstance(image, str):
        image = Image.open(image)
        image.draft("RGB", (2 * size, 2 * size))                  # decode JPEGs at reduced size: much faster
    image = image.convert("RGB")
    scale = size / max(image.size)
    image = image.resize((max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale))), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), PAD_COLOUR)
    canvas.paste(image, ((size - image.size[0]) // 2, (size - image.size[1]) // 2))
    return np.asarray(canvas)


class NormalisedPixels:
    """Arm pixels: float images (n, 3, H, W) with ImageNet's mean and spread removed."""

    def __init__(self, device: str):
        self.device = device
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)

    def __call__(self, images: np.ndarray) -> Tensor:
        x = torch.from_numpy(np.ascontiguousarray(images)).to(self.device).permute(0, 3, 1, 2).float() / 255
        return (x - self.mean) / self.std


class DinoTokens:
    """Frozen DINOv2 ViT-B/14 with registers. Arm dino: the patch tokens after `block` blocks, (n, 256, 768), CLS and
    registers dropped. Arm dino_cls (summary=True): the CLS and register tokens after the last block and the final
    norm, (n, 5, 768), the network's own summary of the photo."""

    def __init__(self, device: str, block: int = 4, summary: bool = False, model: torch.nn.Module | None = None, chunk: int = 256):
        self.device, self.block, self.summary, self.chunk = device, block, summary, chunk
        self.model = model or torch.hub.load(DINOV2, DINOV2_WEIGHTS)
        self.model.to(device).eval().requires_grad_(False)
        self.pixels = NormalisedPixels(device)

    @torch.no_grad()
    def __call__(self, images: np.ndarray) -> Tensor:
        n_summary = 1 + self.model.num_register_tokens
        out = []
        for start in range(0, len(images), self.chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                t = self.model.prepare_tokens_with_masks(self.pixels(images[start:start + self.chunk]))
                for block in self.model.blocks if self.summary else self.model.blocks[: self.block]:
                    t = block(t)
                if self.summary:
                    t = self.model.norm(t)
            out.append((t[:, :n_summary] if self.summary else t[:, n_summary:]).float())
        return torch.cat(out)


def inputs_for(arm: str, device: str, block: int = 4, model: torch.nn.Module | None = None):
    """The photo input of an arm: normalised pixels, or frozen DINOv2 tokens."""
    if arm == "pixels":
        return NormalisedPixels(device)
    return DinoTokens(device, block=block, summary=arm == "dino_cls", model=model)
