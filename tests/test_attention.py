"""The attention block gives the same numbers as a hand-written scaled dot-product attention, with and without masks."""
import math

import torch

from table_image_icl.attention import Attention, TransformerLayer

D, HEADS, B, N_Q, N_K = 16, 2, 2, 5, 7


def attention(seed=0):
    torch.manual_seed(seed)
    return Attention(D, HEADS).eval()


def inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, N_Q, D, generator=g), torch.randn(B, N_K, D, generator=g), torch.randn(B, N_K, D, generator=g)


def reference(att, x, keys, values, key_mask=None):
    """softmax(q k^T / sqrt(d_head)) v per head, written out by hand."""
    q, k, v = att.split(att.q(x)), att.split(att.k(keys)), att.split(att.v(values))
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if key_mask is not None:
        scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
    out = scores.softmax(-1) @ v
    return att.o(out.transpose(1, 2).flatten(-2))


def test_attention_equals_the_hand_written_reference():
    att, (x, k, v) = attention(), inputs()
    assert torch.allclose(att(x, k, v), reference(att, x, k, v), atol=1e-5)


def test_attention_with_a_key_mask_equals_the_reference():
    att, (x, k, v) = attention(), inputs()
    mask = torch.tensor([[True, True, True, False, False, True, False], [True] * 6 + [False]])
    assert torch.allclose(att(x, k, v, key_mask=mask), reference(att, x, k, v, key_mask=mask), atol=1e-5)


def test_masked_keys_get_no_weight():
    att, (x, k, v) = attention(), inputs()
    mask = torch.tensor([[True, True, True, False, False, True, False], [True] * 6 + [False]])
    base = att(x, k, v, key_mask=mask)
    k2, v2 = k.clone(), v.clone()
    k2[~mask] = torch.randn(int((~mask).sum()), D)
    v2[~mask] = torch.randn(int((~mask).sum()), D)
    assert torch.allclose(base, att(x, k2, v2, key_mask=mask), atol=1e-6)
    assert torch.isfinite(base).all()


def test_queries_do_not_interact():
    att, (x, k, v) = attention(), inputs()
    base = att(x, k, v)
    x2 = x.clone()
    x2[:, 2] = torch.randn(D)
    changed = att(x2, k, v)
    assert torch.allclose(base[:, [0, 1, 3, 4]], changed[:, [0, 1, 3, 4]], atol=1e-6)
    assert not torch.allclose(base[:, 2], changed[:, 2])


def test_a_per_query_key_mask_equals_attending_to_each_querys_own_allowed_keys():
    torch.manual_seed(0)
    a = Attention(8, 2).eval()
    x, keys, values = torch.randn(1, 3, 8), torch.randn(1, 6, 8), torch.randn(1, 6, 8)
    allowed = torch.tensor([[[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0], [1, 0, 0, 0, 0, 1]]], dtype=torch.bool)
    out = a(x, keys, values, key_mask=allowed)
    for q in range(3):
        own = a(x[:, q:q + 1], keys[:, allowed[0, q]], values[:, allowed[0, q]])
        assert torch.allclose(out[:, q:q + 1], own, atol=1e-6)


def test_a_layer_with_a_source_reads_the_source_not_the_other_queries():
    torch.manual_seed(0)
    layer = TransformerLayer(D, HEADS).eval()
    x, source, _ = inputs()
    base = layer(x, source=source)
    x2 = x.clone()
    x2[:, 0] = torch.randn(D)
    assert torch.allclose(base[:, 1:], layer(x2, source=source)[:, 1:], atol=1e-6)
    source2 = source.clone()
    source2[:, 3] = torch.randn(D)
    assert not torch.allclose(base, layer(x, source=source2))
