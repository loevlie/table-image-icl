"""The model, stage by stage: what TabICL expects of a table, the attention block, the patch embedders and the
within-photo encoder, the labelled read, and the whole model with the image side's tokens in the rows."""
import inspect
import json
import math
import os

import pytest
import torch
import torch.nn.functional as F

from table_image_icl.model import (CHECKPOINT, DINO_WIDTH, IMAGE_SIDE, REPO, Attention, DinoPatches, ImageEncoder, LabelledRead, LabelRead,
                                   ModelConfig, PixelPatches, TableImageICL, TransformerLayer, config_from_record, contiguous_codes, dino_block,
                                   load_run, nearest_photos, patch_embedder, prepare_table)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")


# ---------- what TabICL expects of a table ----------

def test_contiguous_codes_number_the_classes_of_each_table_from_zero_and_remember_the_slots():
    y = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1], [5, 6, 5, 6, 5, 6, 5, 6]])
    codes, slots = contiguous_codes(y, n_slots=10)
    assert codes.tolist() == [[0, 1, 2, 0, 1, 2, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1]]
    assert slots[0, :4].tolist() == [0, 1, 2, -1] and slots[1, :3].tolist() == [5, 6, -1]


def test_prepare_table_standardises_on_the_context_fills_missing_cells_and_softens_outliers():
    n_ctx = 8
    x = torch.randn(2, n_ctx + 4, 3, generator=torch.Generator().manual_seed(0))
    x[0, n_ctx + 1, 0] = float("nan")                                        # a missing test cell
    x[1, 2, 1] = 1e4                                                         # a wild context value
    z = prepare_table(x, n_ctx)
    assert torch.isfinite(z).all()
    assert z[0, n_ctx + 1, 0] == 0.0, "missing = the context mean, which is 0 after standardising"
    context = z[0, :n_ctx, 2]
    assert abs(context.mean()) < 1e-5 and abs(context.std(correction=0) - 1) < 1e-3
    assert z[1, 2, 1] < 4.0 + math.log1p(1e4), "clipped softly at four standard deviations"
    assert torch.allclose(prepare_table(z, n_ctx)[0, :, 2], z[0, :, 2], atol=1e-4), "a standard column passes through"


def test_test_rows_do_not_change_the_scale():
    n_ctx = 8
    x = torch.randn(1, n_ctx + 4, 2, generator=torch.Generator().manual_seed(0))
    z = prepare_table(x, n_ctx)
    x[0, n_ctx:] *= 100
    assert torch.allclose(prepare_table(x, n_ctx)[0, :n_ctx], z[0, :n_ctx])


# ---------- attention ----------

D, HEADS, B, N_Q, N_K = 16, 2, 2, 5, 7


def attention(seed=0):
    torch.manual_seed(seed)
    return Attention(D, HEADS).eval()


def qkv(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, N_Q, D, generator=g), torch.randn(B, N_K, D, generator=g), torch.randn(B, N_K, D, generator=g)


def reference(att, x, keys, values):
    """softmax(q k^T / sqrt(d_head)) v per head, written out by hand."""
    q, k, v = att.split(att.q(x)), att.split(att.k(keys)), att.split(att.v(values))
    out = (q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).softmax(-1) @ v
    return att.o(out.transpose(1, 2).flatten(-2))


def test_attention_equals_the_hand_written_reference():
    att, (x, k, v) = attention(), qkv()
    assert torch.allclose(att(x, k, v), reference(att, x, k, v), atol=1e-5)


def test_queries_do_not_interact():
    att, (x, k, v) = attention(), qkv()
    base = att(x, k, v)
    x2 = x.clone()
    x2[:, 2] = torch.randn(D)
    changed = att(x2, k, v)
    assert torch.allclose(base[:, [0, 1, 3, 4]], changed[:, [0, 1, 3, 4]], atol=1e-6)
    assert not torch.allclose(base[:, 2], changed[:, 2])


def test_a_layer_mixes_its_tokens_and_keeps_the_shape():
    torch.manual_seed(0)
    layer = TransformerLayer(D, HEADS).eval()
    x = qkv()[0]
    base = layer(x)
    assert base.shape == x.shape
    x2 = x.clone()
    x2[:, 0] = torch.randn(D)
    assert not torch.allclose(base[:, 1:], layer(x2)[:, 1:]), "every token attends to token 0"


# ---------- patch embedders and the within-photo encoder ----------

P = 4


def test_pixel_patches_give_one_token_per_14_px_patch():
    embed = PixelPatches(D)
    assert embed(torch.randn(5, 3, 224, 224)).shape == (5, 256, D)
    assert embed(torch.randn(5, 3, 28, 28)).shape == (5, 4, D)


def test_a_pixel_token_sees_only_its_own_patch():
    torch.manual_seed(0)
    embed = PixelPatches(D)
    images = torch.randn(1, 3, 28, 28)
    changed = images.clone()
    changed[:, :, :14, 14:] += 1.0                               # the top-right patch: index 1 in row-major order
    a, b = embed(images), embed(changed)
    assert torch.allclose(a[:, [0, 2, 3]], b[:, [0, 2, 3]], atol=1e-6) and not torch.allclose(a[:, 1], b[:, 1])


def test_dino_patches_project_each_token():
    embed = DinoPatches(D)
    assert embed(torch.randn(5, P, DINO_WIDTH)).shape == (5, P, D)


def test_the_encoders_pick_their_embedder_and_their_dino_block():
    assert isinstance(patch_embedder("pixels", D), PixelPatches)
    assert patch_embedder("block4", D).n_tokens is None and patch_embedder("block12", D).n_tokens is None
    assert patch_embedder("cls", D).n_tokens == 5
    with pytest.raises(AssertionError):
        patch_embedder("dino", D)
    assert [dino_block(e) for e in ("block4", "block8", "block12", "cls", "pixels")] == [4, 8, 12, None, None]


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


# ---------- the labelled read: what is read back is labels, who reads from whom, and the top-k retrieval ----------

N_CTX, N_TEST, SLOTS = 6, 3, 8
Y_READ = torch.tensor([[0, 1, 2, 0, 1, 2], [3, 4, 3, 4, 3, 4]])


def tokens(seed=0, batch=2, n=N_CTX + N_TEST):
    return torch.randn(batch, n, P, D, generator=torch.Generator().manual_seed(seed))


def stage(seed=0, max_sources=256, read="all", top_k=32):
    torch.manual_seed(seed)
    return LabelledRead(D, heads=2, layers=2, n_slots=SLOTS, max_sources=max_sources, read=read, top_k=top_k).eval()


def test_the_stage_keeps_the_shape_and_takes_only_context_labels():
    assert stage()(tokens(), Y_READ).shape == (2, N_CTX + N_TEST, P, D)
    assert list(inspect.signature(LabelledRead.forward).parameters) == ["self", "tokens", "y_context"]


def test_a_test_photo_influences_only_itself():
    s, t = stage(), tokens()
    t2 = t.clone()
    t2[:, N_CTX + 1] = torch.randn(P, D)
    a, b = s(t, Y_READ), s(t2, Y_READ)
    others = [i for i in range(N_CTX + N_TEST) if i != N_CTX + 1]
    assert torch.allclose(a[:, others], b[:, others], atol=1e-6) and not torch.allclose(a[:, N_CTX + 1], b[:, N_CTX + 1])


def test_every_context_label_reaches_the_test_photos():
    s, t = stage(), tokens()
    base = s(t, Y_READ)
    for j in range(N_CTX):
        y2 = Y_READ.clone()
        y2[:, j] = (Y_READ[:, j] + 1) % SLOTS
        assert not torch.allclose(base[:, N_CTX:], s(t, y2)[:, N_CTX:]), f"context label {j} is ignored"


def test_a_context_photo_reads_its_own_label_too():
    s, t = stage(), tokens()
    y2 = Y_READ.clone()
    y2[:, 0] = (Y_READ[:, 0] + 1) % SLOTS
    assert not torch.allclose(s(t, Y_READ)[:, 0], s(t, y2)[:, 0])


def test_what_is_read_back_is_labels_not_content():
    """If every context photo has the same label, a test photo reads the same thing whatever the context shows."""
    s = stage()
    same = torch.zeros(2, N_CTX, dtype=torch.long)
    t, t2 = tokens(0), tokens(0)
    t2[:, :N_CTX] = tokens(1)[:, :N_CTX]
    assert torch.allclose(s(t, same)[:, N_CTX:], s(t2, same)[:, N_CTX:], atol=1e-5)


def test_at_most_max_sources_context_photos_are_read():
    s, t = stage(max_sources=4), tokens()
    base = s(t, Y_READ)
    y2 = Y_READ.clone()
    y2[:, 4:] = (Y_READ[:, 4:] + 1) % SLOTS                      # context photos 4 and 5 are beyond the cap
    assert torch.allclose(base[:, N_CTX:], s(t, y2)[:, N_CTX:], atol=1e-6)
    s.train()
    picks = {tuple(sorted(s.pick_sources(N_CTX, "cpu").tolist())) for _ in range(30)}
    assert len(picks) > 1 and all(len(p) == 4 for p in picks), "training draws a new subset each time"


def test_the_source_cap_is_a_setting_not_a_weight():
    s = stage(max_sources=4)
    assert s.pick_sources(N_CTX, "cpu").tolist() == [0, 1, 2, 3]
    s.max_sources = 1024
    assert s.pick_sources(N_CTX, "cpu").tolist() == list(range(N_CTX))


def test_gradients_reach_every_parameter():
    s = stage().train()
    s(tokens(), Y_READ).square().sum().backward()
    for name, p in s.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_the_stage_learns_where_to_look_from_the_labelled_context():
    """Each task draws new class markers; a photo of class c hides marker c in one random patch. Which marker means
    which class is known only from the labelled context photos."""
    torch.manual_seed(0)
    d, patches, n_ctx, n_test = 32, 5, 16, 8
    s = LabelledRead(d, heads=4, layers=2, n_slots=2, max_sources=256, read="all")
    head = torch.nn.Linear(d, 1)
    opt = torch.optim.Adam(list(s.parameters()) + list(head.parameters()), lr=1e-3)
    g = torch.Generator().manual_seed(1)

    def task(batch=4):
        markers = F.normalize(torch.randn(batch, 2, d, generator=g), dim=-1) * 6
        y = torch.randint(0, 2, (batch, n_ctx + n_test), generator=g)
        x = torch.randn(batch, n_ctx + n_test, patches, d, generator=g)
        where = torch.randint(0, patches, (batch, n_ctx + n_test), generator=g)
        index = where[..., None, None].expand(-1, -1, 1, d)
        x.scatter_add_(2, index, markers.gather(1, y[..., None].expand(-1, -1, d))[:, :, None])
        return x, y[:, :n_ctx], y[:, n_ctx:]

    def logits(x, y_ctx):
        return head(s(x, y_ctx)[:, n_ctx:].amax(2))[..., 0]

    for _ in range(1500):
        x, y_ctx, y_test = task()
        loss = F.binary_cross_entropy_with_logits(logits(x, y_ctx), y_test.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x, y_ctx, y_test = task(batch=64)
        accuracy = ((logits(x, y_ctx) > 0) == y_test.bool()).float().mean().item()
    assert accuracy > 0.75, accuracy


def test_nearest_photos_are_the_most_similar_by_mean_token():
    t = torch.zeros(1, 5, P, D)
    for i in range(5):
        t[0, i, :, i] = 1.0                                          # photo i points along axis i
    t[0, 4] = t[0, 1] * 0.9 + 0.01                                   # photo 4 looks like photo 1
    nearest = nearest_photos(t, t[:, :4], k=2)
    assert nearest.shape == (1, 5, 2)
    assert nearest[0, 4, 0] == 1, "the test photo's nearest context photo is the one it resembles"
    assert all(nearest[0, i, 0] == i for i in range(4)), "a context photo is its own nearest source"


def test_top_k_reads_only_from_the_nearest_context_photos():
    s, t = stage(read="top_k", top_k=2), tokens()
    nearest = nearest_photos(t, t[:, :N_CTX], k=2)[:, N_CTX:]
    base = s(t, Y_READ)
    for j in range(N_CTX):
        y2 = Y_READ.clone()
        y2[:, j] = (Y_READ[:, j] + 1) % SLOTS
        changed = ~torch.isclose(s(t, y2)[:, N_CTX:], base[:, N_CTX:], atol=1e-6).all(-1).all(-1)
        assert torch.equal(changed, (nearest == j).any(-1)), f"context photo {j}'s label reaches exactly the test photos that retrieved it"


def test_top_k_with_every_context_photo_is_the_plain_read():
    plain, retrieving = stage(read="all"), stage(read="top_k", top_k=N_CTX)
    retrieving.load_state_dict(plain.state_dict())
    assert torch.allclose(plain(tokens(), Y_READ), retrieving(tokens(), Y_READ), atol=1e-5)


def test_top_k_reads_do_not_depend_on_the_chunk_of_photos():
    s, t = stage(read="top_k", top_k=2), tokens()
    base = s(t, Y_READ)
    for read in s.reads:
        read.chunk = 1
    assert torch.allclose(base, s(t, Y_READ), atol=1e-5)


def test_the_top_k_read_is_the_same_whether_photos_come_in_chunks_or_one_by_one():
    torch.manual_seed(0)
    read = LabelRead(D, heads=2).eval()
    photos, n_sources, k = 80, 64, 32
    t = torch.randn(1, photos, P, D)
    sources, labels = t[:, :n_sources], torch.randn(1, n_sources, D)
    nearest = nearest_photos(t, sources, k)
    chunked = read(t, sources, labels, nearest)
    read.chunk = photos
    assert torch.allclose(chunked, read(t, sources, labels, nearest), atol=1e-5)


# ---------- the whole model ----------

CFG = ModelConfig(encoder="pixels", encoder_layers=1, patch_layers=1, image_size=28)
N_CTX_M, N_TEST_M, COLUMNS = 8, 4, 3
Y = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1], [5, 6, 5, 6, 5, 6, 5, 6]])


def table(seed=0):
    return torch.randn(2, N_CTX_M + N_TEST_M, COLUMNS, generator=torch.Generator().manual_seed(seed))


def photos(seed=0):
    return torch.randn(2, N_CTX_M + N_TEST_M, 3, 28, 28, generator=torch.Generator().manual_seed(seed))


def fresh(cfg=CFG):
    """The architecture with no weights loaded. A fresh TabICL starts every block's output at zero, so labels and
    photos cannot change its prediction until it has trained; the tests below that need a reaction use pretrained()."""
    torch.manual_seed(0)
    return TableImageICL(cfg, pretrained=False).eval()


def tabicl_is_cached():
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(REPO, CHECKPOINT, local_files_only=True)
        return True
    except Exception:
        return False


def pretrained(cfg=CFG):
    if not tabicl_is_cached():
        pytest.skip("TabICL v2 checkpoint not in the Hugging Face cache")
    torch.manual_seed(0)
    return TableImageICL(cfg).eval()


def test_log_probabilities_cover_only_the_slots_in_the_context_and_sum_to_one():
    out = fresh()(table(), Y)
    assert out.shape == (2, N_TEST_M, 10)
    assert torch.isinf(out[0, :, 3:]).all() and torch.isfinite(out[0, :, :3]).all()
    assert torch.isinf(out[1, :, :5]).all() and torch.isinf(out[1, :, 7:]).all()
    assert torch.allclose(out.exp().sum(-1), torch.ones(2, N_TEST_M), atol=1e-5)


def test_moving_the_classes_to_other_slots_does_not_change_the_prediction():
    model = fresh()
    out = model(table(), Y)
    moved = torch.tensor([[3, 7, 9, 3, 7, 9, 3, 7], [1, 8, 1, 8, 1, 8, 1, 8]])
    out2 = model(table(), moved)
    assert torch.allclose(out[0][:, [0, 1, 2]], out2[0][:, [3, 7, 9]], atol=1e-5)
    assert torch.allclose(out[1][:, [5, 6]], out2[1][:, [1, 8]], atol=1e-5)


def test_photos_join_the_rows_and_change_the_prediction():
    model = pretrained()
    plain, with_photos = model(table(), Y), model(table(), Y, images=photos())
    assert with_photos.shape == plain.shape and not torch.allclose(plain, with_photos, atol=1e-4)
    assert model.photo_tokens(photos(), Y).shape == (2, N_CTX_M + N_TEST_M, P, 128)


def test_without_photos_the_model_is_tabicl_on_the_standardised_table():
    model = pretrained()
    x = table(3) * 2 + 1
    ours = model(x, Y)
    codes, slots = contiguous_codes(Y)
    with torch.no_grad():
        model.table.train()
        theirs = model.table(prepare_table(x, N_CTX_M), codes, d=None)       # TabICL's own forward: logits over the codes
    for b, k in enumerate([3, 2]):
        assert torch.allclose(ours[b][:, slots[b, :k]], theirs[b, :, :k].log_softmax(-1), atol=1e-4)


def test_the_dino_encoders_take_frozen_tokens():
    block = fresh(ModelConfig(encoder="block4", encoder_layers=1, patch_layers=1, image_size=28))
    assert block(table(), Y, images=torch.randn(2, N_CTX_M + N_TEST_M, P, DINO_WIDTH)).shape == (2, N_TEST_M, 10)
    summary = fresh(ModelConfig(encoder="cls", encoder_layers=1, patch_layers=1, image_size=28))
    assert summary(table(), Y, images=torch.randn(2, N_CTX_M + N_TEST_M, 5, DINO_WIDTH)).shape == (2, N_TEST_M, 10)


def test_the_image_side_is_exactly_what_is_not_tabicl():
    model = fresh()
    names = [n for n, _ in model.named_parameters()]
    ours = [n for n in names if n.startswith(IMAGE_SIDE)]
    assert ours and all(n.startswith("table.") for n in names if n not in ours)


def test_weights_round_trip_strictly():
    model = fresh()
    again = TableImageICL(CFG, pretrained=False).eval()
    again.load_weights(model.state_dict())
    assert torch.allclose(again(table(), Y), model(table(), Y))
    with pytest.raises(RuntimeError):
        again.load_weights({k: v for k, v in model.state_dict().items() if k != "patch_marker"})


# ---------- run records, old and new, and the trained checkpoint ----------

OLD_MODEL = {"image_size": 224, "patch_size": 14, "encoder_layers": 4, "patch_layers": 2, "heads": 4, "max_sources": 256,
             "patch_read": "top_k", "patch_top_k": 32, "dino_width": 768}


def test_an_old_record_names_its_encoder_as_an_arm_and_a_block():
    assert config_from_record({"model": OLD_MODEL, "arm": "dino", "dino_block": 8}) == ModelConfig(encoder="block8")
    assert config_from_record({"model": OLD_MODEL, "arm": "dino", "dino_block": 4}) == ModelConfig()
    assert config_from_record({"model": OLD_MODEL, "arm": "dino_cls", "dino_block": 4}) == ModelConfig(encoder="cls")
    assert config_from_record({"model": OLD_MODEL, "arm": "pixels", "dino_block": 4}) == ModelConfig(encoder="pixels")


def test_a_new_record_round_trips_and_the_shipped_records_are_read():
    from dataclasses import asdict
    cfg = ModelConfig(encoder="block12", patch_read="all", max_sources=512)
    assert config_from_record({"model": asdict(cfg)}) == cfg
    for run, encoder in [("block4", "block4"), ("block8", "block8"), ("block12", "block12"), ("cls", "cls"), ("pixels", "pixels")]:
        with open(os.path.join(ROOT, "results", run, "config.json")) as f:
            assert config_from_record(json.load(f)) == ModelConfig(encoder=encoder), run


def test_the_parameter_names_are_the_ones_the_released_run_was_trained_with():
    """tests/fixtures/block4_keys.json is the state dict of results/block4's run, frozen; a renamed module or
    parameter would leave the released weights unloadable."""
    with open(os.path.join(FIXTURES, "block4_keys.json")) as f:
        frozen = set(json.load(f))
    with open(os.path.join(ROOT, "results", "block4", "config.json")) as f:
        model = TableImageICL(config_from_record(json.load(f)), pretrained=False)
    assert set(model.state_dict()) == frozen
    assert set(TableImageICL(ModelConfig(encoder="block4"), pretrained=False).state_dict()) == frozen


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "checkpoints", "block4", "model.pt")), reason="released weights not downloaded")
def test_the_released_checkpoint_loads_and_gives_the_recorded_probabilities():
    """tests/fixtures/block4_probabilities.json holds what the released weights predicted on a fixed table, with and
    without random DINOv2 tokens, when the record was made."""
    model, record = load_run(os.path.join(ROOT, "checkpoints", "block4"))
    assert model.cfg.encoder == "block4" and not model.training
    with open(os.path.join(FIXTURES, "block4_probabilities.json")) as f:
        expected = json.load(f)
    g = torch.Generator().manual_seed(expected["seed"])
    x = torch.randn(1, 12, 4, generator=g)
    x[0, 9, 1] = float("nan")
    y = torch.tensor([[3, 5, 3, 5, 7, 3, 5, 7]])
    images = torch.randn(1, 12, 256, DINO_WIDTH, generator=g)
    with torch.no_grad():
        plain, with_photo = model(x, y).exp()[0], model(x, y, images=images).exp()[0]
    assert torch.allclose(plain, torch.tensor(expected["plain"]), atol=1e-4)
    assert torch.allclose(with_photo, torch.tensor(expected["with_photo"]), atol=1e-4)
