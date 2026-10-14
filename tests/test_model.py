"""The whole model: TabICL's table stages with the image side's tokens in the rows."""
import pytest
import torch

from table_image_icl.model import IMAGE_SIDE, TableImageICL, ModelConfig, config_from_record
from table_image_icl.tables import contiguous_codes, prepare_table

CFG = ModelConfig(encoder_layers=1, patch_layers=1, image_size=28, patch_size=14, dino_width=24)
N_CTX, N_TEST, COLUMNS, P = 8, 4, 3, 4
Y = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1], [5, 6, 5, 6, 5, 6, 5, 6]])


def table(seed=0):
    return torch.randn(2, N_CTX + N_TEST, COLUMNS, generator=torch.Generator().manual_seed(seed))


def photos(seed=0):
    return torch.randn(2, N_CTX + N_TEST, 3, 28, 28, generator=torch.Generator().manual_seed(seed))


def fresh(arm="pixels"):
    """The architecture with no weights loaded. A fresh TabICL starts every block's output at zero, so labels and
    photos cannot change its prediction until it has trained; the tests below that need a reaction use pretrained()."""
    torch.manual_seed(0)
    return TableImageICL(CFG, arm, pretrained=False).eval()


def pretrained(arm="pixels"):
    try:
        from huggingface_hub import hf_hub_download
        from table_image_icl.tables import CHECKPOINT, REPO
        hf_hub_download(REPO, CHECKPOINT, local_files_only=True)
    except Exception:
        pytest.skip("TabICL v2 checkpoint not in the Hugging Face cache")
    torch.manual_seed(0)
    return TableImageICL(CFG, arm).eval()


def test_log_probabilities_cover_only_the_slots_in_the_context_and_sum_to_one():
    out = fresh()(table(), Y)
    assert out.shape == (2, N_TEST, 10)
    assert torch.isinf(out[0, :, 3:]).all() and torch.isfinite(out[0, :, :3]).all()
    assert torch.isinf(out[1, :, :5]).all() and torch.isinf(out[1, :, 7:]).all()
    assert torch.allclose(out.exp().sum(-1), torch.ones(2, N_TEST), atol=1e-5)


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
    assert model.photo_tokens(photos(), Y).shape == (2, N_CTX + N_TEST, P, 128)


def test_without_photos_the_model_is_tabicl_on_the_standardised_table():
    model = pretrained()
    x = table(3) * 2 + 1
    ours = model(x, Y)
    codes, slots = contiguous_codes(Y)
    with torch.no_grad():
        model.table.train()
        theirs = model.table(prepare_table(x, N_CTX), codes, d=None)         # TabICL's own forward: logits over the codes
    for b, k in enumerate([3, 2]):
        assert torch.allclose(ours[b][:, slots[b, :k]], theirs[b, :, :k].log_softmax(-1), atol=1e-4)


def test_the_dino_arms_take_frozen_tokens():
    assert fresh("dino")(table(), Y, images=torch.randn(2, N_CTX + N_TEST, P, CFG.dino_width)).shape == (2, N_TEST, 10)
    assert fresh("dino_cls")(table(), Y, images=torch.randn(2, N_CTX + N_TEST, 5, CFG.dino_width)).shape == (2, N_TEST, 10)


def test_the_image_side_is_exactly_what_is_not_tabicl():
    model = fresh()
    names = [n for n, _ in model.named_parameters()]
    ours = [n for n in names if n.startswith(IMAGE_SIDE)]
    assert ours and all(n.startswith("table.") for n in names if n not in ours)


def test_weights_round_trip_strictly():
    model = fresh()
    again = TableImageICL(CFG, "pixels", pretrained=False).eval()
    again.load_weights(model.state_dict())
    assert torch.allclose(again(table(), Y), model(table(), Y))
    with pytest.raises(RuntimeError):
        again.load_weights({k: v for k, v in model.state_dict().items() if k != "patch_marker"})


def test_a_config_is_read_from_a_run_record():
    record = {"model": {"image_size": 224, "patch_size": 14, "encoder_layers": 4, "patch_layers": 2, "heads": 4, "max_sources": 256,
                        "patch_read": "top_k", "patch_top_k": 32, "dino_width": 768}, "arm": "dino"}
    assert config_from_record(record) == ModelConfig()
