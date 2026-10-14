"""The training loop: the loss, the schedule, checkpoints, the record of a run, and a run that resumes."""
import json
import math
import os
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from table_image_icl.inputs import NormalisedPixels
from table_image_icl.model import TableImageICL, ModelConfig
from table_image_icl.tasks import TaskConfig, Tasks
from table_image_icl.training import TrainConfig, learning_rate, load_checkpoint, make_optimizer, save_checkpoint, task_loss, train, validate

TINY = ModelConfig(encoder_layers=1, patch_layers=1, image_size=28, patch_size=14)


def tiny_model():
    torch.manual_seed(0)
    return TableImageICL(TINY, "pixels", pretrained=False)


def tiny_tasks():
    rng = np.random.default_rng(0)

    def collection(n_classes):
        labels = np.repeat(np.arange(n_classes), 12)
        return {"images": rng.integers(0, 256, (len(labels), 28, 28, 3), dtype=np.uint8), "labels": labels, "kind": "class"}

    cfg = TaskConfig(batch_size=2, rows=(32, 48), features=(3, 6), max_classes=4, max_bins=4)
    return Tasks(cfg, {"seen": collection(8), "unseen": collection(6)}, ["seen"], ["unseen"], NormalisedPixels("cpu"))


def short_run(out, max_steps):
    return TrainConfig(out=out, max_steps=max_steps, validate_every=2, n_validation_batches=2, workers=0, warmup_steps=2, amp=False)


def metrics_of(out):
    return [m for m in map(json.loads, open(f"{out}/metrics.jsonl")) if "step" in m]


def test_loss_ignores_test_rows_whose_class_is_not_in_the_context():
    log_probs = torch.log_softmax(torch.randn(1, 3, 4), -1)
    y_context = torch.tensor([[0, 1, 0, 1]])
    seen, unseen = torch.tensor([[0, 1, 1]]), torch.tensor([[0, 1, 3]])
    loss_seen, _ = task_loss(log_probs, y_context, seen)
    loss_unseen, _ = task_loss(log_probs, y_context, unseen)
    two_rows, _ = task_loss(log_probs[:, :2], y_context, seen[:, :2])
    assert torch.allclose(loss_unseen, two_rows) and not torch.allclose(loss_seen, two_rows)


def test_accuracy_counts_only_rows_that_can_be_predicted():
    log_probs = torch.log(torch.tensor([[[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]]))
    _, accuracy = task_loss(log_probs, torch.tensor([[0, 1]]), torch.tensor([[0, 1, 1]]))
    assert math.isclose(accuracy, 2 / 3, rel_tol=1e-5)


def test_learning_rate_warms_up_then_decays_and_is_a_factor_of_each_groups_base_rate():
    cfg = TrainConfig(out="unused", lr=1e-5, image_lr=1e-4, warmup_steps=10)
    assert learning_rate(cfg, 1, 0.0, 1e-4) < learning_rate(cfg, 5, 0.0, 1e-4) < 1e-4
    assert math.isclose(learning_rate(cfg, 10, 0.0, 1e-4), 1e-4)
    assert learning_rate(cfg, 100, 1.0, 1e-4) < 1e-9
    optimizer = make_optimizer(tiny_model(), cfg)
    assert [g["base_lr"] for g in optimizer.param_groups] == [1e-5, 1e-4], "the table stages and the image side"
    assert all(len(g["params"]) > 0 for g in optimizer.param_groups)


def test_checkpoint_round_trip(tmp_path):
    model = tiny_model()
    optimizer = make_optimizer(model, TrainConfig(out="unused"))
    path = str(tmp_path / "checkpoint.pt")
    save_checkpoint(path, model, optimizer, step=7, elapsed=12.5, best_loss=0.4)
    assert os.listdir(tmp_path) == ["checkpoint.pt"], "written to a temporary file, then renamed"
    other = tiny_model()
    other.read.label.weight.data.zero_()
    assert load_checkpoint(path, other, make_optimizer(other, TrainConfig(out="unused"))) == (7, 12.5, 0.4)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), other.state_dict().values()))


def test_a_short_run_writes_its_record_logs_every_view_and_resumes(tmp_path):
    pytest.importorskip("tabicl")
    out = str(tmp_path / "run")
    settings = {"model": asdict(TINY), "arm": "pixels"}
    train(short_run(out, 2), tiny_model(), tiny_tasks(), settings)
    record = json.load(open(f"{out}/config.json"))
    assert {"train", "code_version", "started", "parameters", "machine"} <= set(record) and record["arm"] == "pixels"
    first = metrics_of(out)[0]
    assert set(first["views"]) == {"image", "no image", "column visible", "images shuffled"}
    assert all(math.isfinite(v["loss"]) and 0 <= v["accuracy"] <= 1 for v in first["views"].values())
    assert first["validation_loss"] == first["views"]["image"]["loss"], "the best weights are chosen with photos"
    assert first["grad_norm"] > 0 and first["elapsed_hours"] > 0
    assert os.path.exists(f"{out}/model.pt") and os.path.exists(f"{out}/model-final.pt") and os.path.exists(f"{out}/validation.pt")

    train(short_run(out, 4), tiny_model(), tiny_tasks(), settings)        # same folder: resumes at step 2
    assert [m["step"] for m in metrics_of(out)] == [2, 4]
    assert {"resumed_at": 2} in [json.loads(line) for line in open(f"{out}/metrics.jsonl")]


def test_a_resumed_run_continues_from_its_checkpoint_not_from_the_model_it_is_given(tmp_path):
    pytest.importorskip("tabicl")
    out = str(tmp_path / "run")
    train(short_run(out, 2), tiny_model(), tiny_tasks(), {})
    trained = torch.load(f"{out}/checkpoint.pt", map_location="cpu", weights_only=True)["model"]
    torch.manual_seed(5)
    fresh = TableImageICL(TINY, "pixels", pretrained=False)
    train(short_run(out, 2), fresh, tiny_tasks(), {})                      # the budget is used up: loads, trains no step
    assert all(torch.equal(fresh.state_dict()[k].cpu(), v) for k, v in trained.items())


def test_a_resume_with_other_settings_is_refused(tmp_path):
    pytest.importorskip("tabicl")
    out = str(tmp_path / "run")
    train(short_run(out, 2), tiny_model(), tiny_tasks(), {"arm": "pixels"})
    with pytest.raises(AssertionError, match="other settings"):
        train(replace(short_run(out, 4), lr=5e-4), tiny_model(), tiny_tasks(), {"arm": "pixels"})
    with pytest.raises(AssertionError, match="other settings"):
        train(short_run(out, 4), tiny_model(), tiny_tasks(), {"arm": "dino"})


def test_validation_leaves_the_model_and_the_random_state_as_they_were():
    pytest.importorskip("tabicl")
    tasks = tiny_tasks()
    stream = tasks.validation_batches(0)
    batches = [next(stream) for _ in range(2)]
    model = tiny_model().train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()
    views = validate(model, tasks, batches, "cpu", amp=False)
    assert set(views) == set(tasks.VIEWS)
    assert model.training and torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
