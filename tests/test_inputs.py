"""From stored uint8 photos to what each arm's embedder takes."""
import os

import numpy as np
import pytest
import torch

from table_image_icl.inputs import PAD_COLOUR, DinoTokens, NormalisedPixels, inputs_for, letterbox


def photos(n=3, size=28, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (n, size, size, 3), dtype=np.uint8)


def test_pixels_are_channel_first_and_normalised():
    x = NormalisedPixels("cpu")(photos())
    assert x.shape == (3, 3, 28, 28) and x.dtype == torch.float32
    padding = np.broadcast_to(np.array(PAD_COLOUR, dtype=np.uint8), (1, 28, 28, 3))
    assert NormalisedPixels("cpu")(padding).abs().max() < 0.02, "the padding colour is zero after normalisation"


def test_letterbox_pads_to_a_square_without_stretching():
    from PIL import Image
    wide = Image.fromarray(np.full((10, 40, 3), 255, np.uint8))
    out = letterbox(wide, size=40)
    assert out.shape == (40, 40, 3)
    assert (out[15:25] == 255).all() and tuple(out[0, 0]) == PAD_COLOUR


class TinyViT(torch.nn.Module):
    """The parts of DINOv2's interface that DinoTokens uses. Block i adds 10**i, so the output says which blocks ran."""
    num_register_tokens = 2

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([AddConstant(10.0 ** i) for i in range(4)])
        self.norm = AddConstant(0.5)

    def prepare_tokens_with_masks(self, x):
        patches = x.unflatten(2, (2, 14)).unflatten(4, (2, 14)).mean((1, 3, 5)).flatten(1)[..., None]    # 4 patch means
        special = torch.full((len(x), 3, 1), -1000.0)                                                       # CLS + 2 registers
        return torch.cat([special, patches], 1)


class AddConstant(torch.nn.Module):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def forward(self, x):
        return x + self.constant


def test_block_tokens_are_the_patch_tokens_after_the_chosen_block():
    raw = photos()
    tokens = DinoTokens("cpu", block=2, model=TinyViT())(raw)
    assert tokens.shape == (3, 4, 1), "CLS and register tokens are dropped"
    patch_means = NormalisedPixels("cpu")(raw).unflatten(2, (2, 14)).unflatten(4, (2, 14)).mean((1, 3, 5)).flatten(1)
    assert torch.allclose(tokens[..., 0], patch_means + 1 + 10, atol=1e-4), "blocks 1 and 2 ran, 3 and 4 did not"


def test_summary_tokens_are_the_normed_cls_and_registers_after_the_last_block():
    tokens = DinoTokens("cpu", summary=True, model=TinyViT())(photos())
    assert tokens.shape == (3, 3, 1), "CLS + 2 registers, patches dropped"
    assert torch.allclose(tokens, torch.full((3, 3, 1), -1000.0 + 1 + 10 + 100 + 1000 + 0.5))


def test_tokens_do_not_depend_on_the_chunk_size_and_carry_no_gradient():
    raw = photos(n=5)
    whole = DinoTokens("cpu", block=3, model=TinyViT(), chunk=64)
    in_chunks = DinoTokens("cpu", block=3, model=TinyViT(), chunk=2)
    assert torch.equal(whole(raw), in_chunks(raw))
    assert not whole(raw).requires_grad and not any(p.requires_grad for p in whole.model.parameters())


def test_inputs_for_picks_the_input_by_arm():
    assert isinstance(inputs_for("pixels", "cpu"), NormalisedPixels)
    assert inputs_for("dino", "cpu", block=2, model=TinyViT()).summary is False
    assert inputs_for("dino_cls", "cpu", model=TinyViT()).summary is True


@pytest.mark.skipif(not os.path.isdir(os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8")),
                    reason="DINOv2 is not in the torch hub cache")
def test_real_dino_block_4_gives_256_patch_tokens():
    raw = photos(n=2, size=224)
    dino = DinoTokens("cpu", block=4)
    tokens = dino(raw)
    assert tokens.shape == (2, 256, 768)
    with torch.no_grad():
        reference = dino.model.prepare_tokens_with_masks(NormalisedPixels("cpu")(raw))
        for block in dino.model.blocks[:4]:
            reference = block(reference)
    assert torch.allclose(tokens, reference[:, 5:], atol=1e-4)
