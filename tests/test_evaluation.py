"""scripts/evaluate.py: scores by photo role and view, and the share of the floor-to-ceiling gap the photo closes."""
import importlib.util
import math
import os

import numpy as np
import pytest
import torch

from table_image_icl.inputs import NormalisedPixels
from table_image_icl.model import TableImageICL, ModelConfig
from table_image_icl.tasks import TaskConfig, Tasks
from table_image_icl.training import fixed_batches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = ModelConfig(encoder="pixels", encoder_layers=1, patch_layers=1, image_size=28)


def script():
    spec = importlib.util.spec_from_file_location("evaluate", os.path.join(ROOT, "scripts", "evaluate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate = script()
gap_closed, score_by_role = evaluate.gap_closed, evaluate.score_by_role


def tasks():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(6), 12)
    collection = {"images": rng.integers(0, 256, (len(labels), 28, 28, 3), dtype=np.uint8), "labels": labels, "kind": "class"}
    cfg = TaskConfig(batch_size=2, rows=(32, 48), features=(3, 6), max_classes=4, max_bins=4)
    return Tasks(cfg, {"c": collection}, ["c"], ["c"], NormalisedPixels("cpu"))


def test_gap_closed_is_zero_at_the_floor_and_one_at_the_ceiling():
    views = {"no image": {"loss": 1.0, "accuracy": 0.5}, "column visible": {"loss": 0.8, "accuracy": 0.7}, "image": {"loss": 0.95, "accuracy": 0.55}}
    assert gap_closed(views) == {"loss": pytest.approx(0.25), "accuracy": pytest.approx(0.25)}
    views["image"] = views["no image"]
    assert gap_closed(views) == {"loss": 0.0, "accuracy": 0.0}
    views["image"] = views["column visible"]
    assert gap_closed(views) == {"loss": pytest.approx(1.0), "accuracy": pytest.approx(1.0)}
    views["column visible"] = views["no image"]
    assert all(math.isnan(v) for v in gap_closed(views).values()), "no gap to close: undefined"


def test_scores_come_per_role_and_per_view():
    pytest.importorskip("tabicl")
    t = tasks()
    stream = t.validation_batches(seed=0)
    batches = []
    while {b.image_role for b in batches} != {"feature", "label"} or len(batches) < 4:
        batches.append(next(stream))
    torch.manual_seed(0)
    scores = score_by_role(TableImageICL(TINY, pretrained=False), t, batches, "cpu", amp=False)
    assert set(scores) == {"feature", "label"}
    for r in scores.values():
        assert set(r["views"]) == set(t.VIEWS) and set(r["gap_closed"]) == {"loss", "accuracy"}
        assert all(math.isfinite(v["loss"]) and 0 <= v["accuracy"] <= 1 for v in r["views"].values())
        gain = r["image gain"]
        assert math.isclose(gain["loss_mean"], r["views"]["no image"]["loss"] - r["views"]["image"]["loss"], abs_tol=1e-6)
        assert gain["batches"] * 2 == r["views"]["image"]["tasks"] and 0 <= gain["share_of_batches_helped"] <= 1
    assert sum(r["views"]["image"]["tasks"] for r in scores.values()) == 2 * len(batches)


def test_evaluation_tasks_are_made_once_and_read_back_the_same(tmp_path):
    pytest.importorskip("tabicl")
    t = tasks()
    path = str(tmp_path / "tasks.pt")
    first = fixed_batches(path, t, seed=20_000, n_batches=3)
    assert len(first) == 3 and os.path.exists(path)
    second = fixed_batches(path, t, seed=20_000, n_batches=3)
    for a, b in zip(first, second):
        assert torch.equal(a.x, b.x) and torch.equal(a.image_index, b.image_index) and a.image_role == b.image_role and a.collection == b.collection
