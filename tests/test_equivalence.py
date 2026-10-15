"""Where our code re-implements or truncates someone else's, the two are run side by side here. Each test names the
upstream file it checks against (the tour at tour/app.py links both; tour/provenance.py lists the ports)."""
import math
import os

import numpy as np
import pytest
import torch
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, train_test_split

from table_image_icl.model import OUTLIER_SIGMAS, prepare_table
from table_image_icl.multabench import split
from table_image_icl.training import TrainConfig, learning_rate

HUB_CACHE = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8")
needs_dinov2 = pytest.mark.skipif(not os.path.isdir(HUB_CACHE), reason="DINOv2 is not in the torch hub cache")


# ---------- model.prepare_table vs TabICL's own preprocessing (tabicl/_sklearn/preprocessing.py) ----------

def test_prepare_table_is_tabicl_s_standard_scaler_then_its_outlier_remover_fitted_on_the_context():
    """TabICL's classifier (norm_methods "none") standardises with CustomStandardScaler (z-score with 1e-6, clipped to
    +-100) and softens outliers with OutlierRemover (two-stage bounds at 4 sigma, log beyond them), both fitted on the
    training rows. prepare_table does the same on a batch of tables in torch, fitted on each table's context rows."""
    from tabicl._sklearn.preprocessing import CustomStandardScaler, OutlierRemover
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 40, 5)) * rng.uniform(0.5, 5, (2, 1, 5)) + rng.uniform(-3, 3, (2, 1, 5))
    x[0, 3, 1] = 60.0                                                         # an outlier in the context
    x[1, 35, 2] = -45.0                                                       # and one among the test rows
    n_context = 30
    ours = prepare_table(torch.tensor(x, dtype=torch.float64), n_context).numpy()
    for t in range(2):
        scaler = CustomStandardScaler().fit(x[t, :n_context])
        remover = OutlierRemover(threshold=OUTLIER_SIGMAS).fit(scaler.transform(x[t, :n_context]))
        theirs = remover.transform(scaler.transform(x[t]))
        assert np.allclose(ours[t], theirs, atol=1e-6), f"table {t}: max difference {np.abs(ours[t] - theirs).max()}"


# ---------- multabench.split vs MulTaBench's get_split (multabench/benchmark/splits.py, commit eebb15e) ----------

def multabench_get_split(y, is_cls, split_k, ten_k=True):
    """get_split of https://github.com/alanarazi7/MulTaBench/blob/eebb15e/multabench/benchmark/splits.py (MIT),
    copied here with its constants inlined: repeated 3-fold, 2 repeats, seed 42, caps 10,000 / 5,000."""
    def smallest(y):
        return np.unique(y, return_counts=True)[1].min()

    def cap(idx, size, seed):
        if len(idx) <= size:
            return idx
        stratify = y[idx] if is_cls and smallest(y[idx]) >= 2 else None
        return train_test_split(idx, train_size=size, random_state=seed, stratify=stratify)[0]

    y = np.asarray(y)
    cv_cls = RepeatedStratifiedKFold if is_cls and smallest(y) >= 3 else RepeatedKFold
    train, test = list(cv_cls(n_splits=3, n_repeats=2, random_state=42).split(np.zeros(len(y)), y))[split_k]
    if ten_k:
        train, test = cap(train, 10_000, 42 + split_k), cap(test, 5_000, 42 + split_k)
    return np.sort(train), np.sort(test)


@pytest.mark.parametrize("is_cls, n", [(True, 900), (True, 20_000), (False, 700), (False, 18_000)])
def test_split_gives_the_benchmark_s_rows_for_every_split_and_both_sizes(is_cls, n):
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, n) if is_cls else rng.standard_normal(n)
    for k in range(6):
        ours = split(y, is_cls, k)
        theirs = multabench_get_split(y, is_cls, k)
        assert all(np.array_equal(a, b) for a, b in zip(ours, theirs, strict=True)), f"split {k}"


def test_split_falls_back_to_plain_folds_when_a_class_has_fewer_than_three_rows():
    y = np.array([0] * 50 + [1] * 50 + [2] * 2)
    assert all(np.array_equal(a, b) for a, b in zip(split(y, True, 4), multabench_get_split(y, True, 4), strict=True))


# ---------- training.learning_rate vs torch's CosineAnnealingLR after the warm-up ----------

def test_learning_rate_is_cosine_annealing_after_the_warm_up():
    """Loshchilov & Hutter's cosine (SGDR without restarts), as torch.optim.lr_scheduler.CosineAnnealingLR computes it
    from the step; ours takes the progress through the budget instead, which for a step budget is step / max_steps,
    and the group's base rate as an argument."""
    cfg = TrainConfig(out="x", lr=3e-4, lr_end=1e-5, warmup_steps=100, max_steps=1000)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_steps, eta_min=cfg.lr_end)
    for step in range(cfg.max_steps + 1):
        torch_lr = optimizer.param_groups[0]["lr"]
        ours = learning_rate(cfg, step, step / cfg.max_steps, cfg.lr)
        if step >= cfg.warmup_steps:
            assert math.isclose(ours, torch_lr, rel_tol=1e-6), f"step {step}: {ours} vs {torch_lr}"
        else:
            assert math.isclose(ours, torch_lr * step / cfg.warmup_steps, rel_tol=1e-6), "linear warm-up on top"
        optimizer.step()
        scheduler.step()


# ---------- inputs.DinoTokens vs DINOv2's own forward (dinov2/models/vision_transformer.py) ----------

@needs_dinov2
def test_running_every_block_ourselves_gives_dinov2_s_own_patch_features():
    """DinoTokens runs prepare_tokens_with_masks and the blocks in a loop. With all 12 blocks and DINOv2's final norm
    applied after, that is forward_features' x_norm_patchtokens."""
    from table_image_icl.inputs import DINOV2, DINOV2_WEIGHTS, DinoTokens, NormalisedPixels
    model = torch.hub.load(DINOV2, DINOV2_WEIGHTS).eval()
    raw = np.random.default_rng(0).integers(0, 255, (2, 224, 224, 3), dtype=np.uint8)
    ours = DinoTokens("cpu", block=12, model=model)(raw)
    with torch.no_grad():
        theirs = model.forward_features(NormalisedPixels("cpu")(raw))["x_norm_patchtokens"]
        ours = model.norm(ours)
    assert ours.shape == theirs.shape == (2, 256, 768)
    assert torch.allclose(ours, theirs, atol=1e-4)


@needs_dinov2
def test_the_summary_tokens_are_dinov2_s_own_normed_cls_and_registers():
    """Encoder cls: DinoTokens(summary=True) is forward_features' x_norm_clstoken and x_norm_regtokens, in that order."""
    from table_image_icl.inputs import DINOV2, DINOV2_WEIGHTS, DinoTokens, NormalisedPixels
    model = torch.hub.load(DINOV2, DINOV2_WEIGHTS).eval()
    raw = np.random.default_rng(0).integers(0, 255, (2, 224, 224, 3), dtype=np.uint8)
    ours = DinoTokens("cpu", summary=True, model=model)(raw)
    with torch.no_grad():
        features = model.forward_features(NormalisedPixels("cpu")(raw))
    theirs = torch.cat([features["x_norm_clstoken"][:, None], features["x_norm_regtokens"]], 1)
    assert ours.shape == theirs.shape == (2, 5, 768) and torch.allclose(ours, theirs, atol=1e-4)
