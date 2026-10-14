"""Using the model on a real table: numpy in, class probabilities or regression values out."""
import numpy as np
import pytest
import torch

from table_image_icl.inputs import NormalisedPixels
from table_image_icl.model import TableImageICL, ModelConfig
from table_image_icl.predict import bin_targets, expected_value, predict_classes, predict_values

TINY = ModelConfig(encoder_layers=1, patch_layers=1, image_size=28, patch_size=14)
N_CTX = 200


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return TableImageICL(TINY, "pixels", pretrained=False).eval()


def data(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(40, 5)), rng.normal(size=(9, 5)), rng


def test_one_probability_row_per_test_row_over_the_context_classes(model):
    x_context, x_test, rng = data()
    probabilities, classes = predict_classes(model, x_context, rng.choice([3, 7, 9], 40), x_test)
    assert classes.tolist() == [3, 7, 9] and probabilities.shape == (9, 3)
    assert np.allclose(probabilities.sum(1), 1, atol=1e-5)


def test_label_names_do_not_matter(model):
    x_context, x_test, rng = data()
    y = rng.choice([0, 1, 2], 40)
    base, _ = predict_classes(model, x_context, y, x_test)
    renamed, classes = predict_classes(model, x_context, y * 10 + 5, x_test)
    assert classes.tolist() == [5, 15, 25] and np.allclose(base, renamed, atol=1e-6)


def test_chunking_the_test_rows_changes_nothing(model):
    x_context, x_test, rng = data()
    y = rng.choice([0, 1], 40)
    whole, _ = predict_classes(model, x_context, y, x_test, chunk=100)
    in_chunks, _ = predict_classes(model, x_context, y, x_test, chunk=2)
    assert np.allclose(whole, in_chunks, atol=1e-5)


def test_missing_values_are_accepted(model):
    x_context, x_test, rng = data()
    x_context[2, 1] = np.nan
    x_test[0, 3] = np.nan
    probabilities, _ = predict_classes(model, x_context, rng.choice([0, 1], 40), x_test)
    assert np.isfinite(probabilities).all()


def test_photos_go_with_their_rows(model):
    x_context, x_test, rng = data()
    y = rng.choice([0, 1, 2], 40)
    photos = (rng.integers(0, 256, (40, 28, 28, 3), dtype=np.uint8), rng.integers(0, 256, (9, 28, 28, 3), dtype=np.uint8))
    whole, _ = predict_classes(model, x_context, y, x_test, photos=photos, to_tokens=NormalisedPixels("cpu"))
    in_chunks, _ = predict_classes(model, x_context, y, x_test, chunk=4, photos=photos, to_tokens=NormalisedPixels("cpu"))
    assert whole.shape == (9, 3) and np.allclose(whole, in_chunks, atol=1e-5)


def test_regression_predictions_stay_within_the_context_targets(model):
    x_context, x_test, rng = data()
    y = rng.normal(size=40) * 50 + 200
    predictions = predict_values(model, x_context, y, x_test, n_bins=8)
    assert predictions.shape == (9,) and (predictions >= y.min()).all() and (predictions <= y.max()).all()
    few = predict_values(model, x_context, rng.choice([1.0, 2.0, 5.0], 40), x_test, n_bins=10)
    assert np.isfinite(few).all() and (few >= 1.0).all() and (few <= 5.0).all(), "fewer distinct targets than bins"


# ---------- regression as classification ----------

def targets(seed=0, n=260):
    return np.random.default_rng(seed).normal(size=n)


@pytest.mark.parametrize("seed", range(3))
def test_every_bin_holds_a_context_row_and_order_is_kept(seed):
    y = targets(seed)
    bins, means = bin_targets(y, N_CTX, n_bins=10)
    assert set(bins[:N_CTX]) == set(range(len(means))) and len(means) <= 10
    assert (np.diff(bins[np.argsort(y)]) >= 0).all() and (np.diff(means) > 0).all()


def test_bins_come_from_the_context_only():
    y = targets()
    y2 = y.copy()
    y2[N_CTX:] = y2[N_CTX:] * 100 + 7
    y2[N_CTX], y2[N_CTX + 1] = 1e6, -1e6
    bins, means = bin_targets(y, N_CTX, 10)
    bins2, means2 = bin_targets(y2, N_CTX, 10)
    assert np.array_equal(bins[:N_CTX], bins2[:N_CTX]) and np.allclose(means, means2)
    assert bins2[N_CTX] == len(means) - 1 and bins2[N_CTX + 1] == 0, "a test target beyond the context range joins the end bin"


def test_repeated_values_leave_no_empty_bin():
    y = np.random.default_rng(0).choice([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 50.0], size=260)
    bins, means = bin_targets(y, N_CTX, 10)
    assert set(bins[:N_CTX]) == set(range(len(means))) and len(means) == 4


def test_bin_means_are_the_mean_context_target_of_each_bin():
    y = targets()
    bins, means = bin_targets(y, N_CTX, 10)
    for b in range(len(means)):
        assert np.isclose(means[b], y[:N_CTX][bins[:N_CTX] == b].mean())


def test_the_expected_value_of_a_certain_bin_is_that_bins_mean():
    y = targets()
    _, means = bin_targets(y, N_CTX, 10)
    certain = torch.full((len(means), len(means)), -1e9)
    certain[torch.arange(len(means)), torch.arange(len(means))] = 0.0
    means = torch.tensor(means, dtype=torch.float32)
    assert torch.allclose(expected_value(certain, means), means, atol=1e-6)
