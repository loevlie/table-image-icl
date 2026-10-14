"""Patch embedders and the within-photo encoder."""
import inspect

import torch

from table_image_icl.images import DinoPatches, ImageEncoder, PixelPatches, patch_embedder

D, P = 16, 4


def test_pixel_patches_give_one_token_per_patch():
    embed = PixelPatches(D, patch_size=14)
    assert embed(torch.randn(5, 3, 224, 224)).shape == (5, 256, D)
    assert embed(torch.randn(5, 3, 28, 28)).shape == (5, 4, D)


def test_a_pixel_token_sees_only_its_own_patch():
    torch.manual_seed(0)
    embed = PixelPatches(D, patch_size=14)
    images = torch.randn(1, 3, 28, 28)
    changed = images.clone()
    changed[:, :, :14, 14:] += 1.0                               # the top-right patch: index 1 in row-major order
    a, b = embed(images), embed(changed)
    assert torch.allclose(a[:, [0, 2, 3]], b[:, [0, 2, 3]], atol=1e-6) and not torch.allclose(a[:, 1], b[:, 1])


def test_dino_patches_project_each_token():
    embed = DinoPatches(D, dino_width=24)
    assert embed(torch.randn(5, P, 24)).shape == (5, P, D)


def test_the_arms_pick_their_embedder():
    assert isinstance(patch_embedder("pixels", D, 14, 24), PixelPatches)
    assert patch_embedder("dino", D, 14, 24).n_tokens is None
    assert patch_embedder("dino_cls", D, 14, 24).n_tokens == 5


def encoder(seed=0):
    torch.manual_seed(seed)
    return ImageEncoder(D, heads=2, layers=2, n_patches=P).eval()


def test_encoder_keeps_the_shape_and_takes_no_labels():
    assert encoder()(torch.randn(7, P, D)).shape == (7, P, D)
    assert list(inspect.signature(ImageEncoder.forward).parameters) == ["self", "tokens"]


def test_encoder_treats_photos_independently():
    e, x = encoder(), torch.randn(5, P, D)
    x2 = x.clone()
    x2[2] = torch.randn(P, D)
    a, b = e(x), e(x2)
    assert torch.allclose(a[[0, 1, 3, 4]], b[[0, 1, 3, 4]], atol=1e-6) and not torch.allclose(a[2], b[2])


def test_encoder_knows_where_a_patch_is():
    e, x = encoder(), torch.randn(1, P, D)
    swapped = x[:, [1, 0, 2, 3]]
    assert not torch.allclose(e(x)[:, [1, 0, 2, 3]], e(swapped), atol=1e-4)
