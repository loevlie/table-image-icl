"""The training script end to end on a tiny decoded photo source, on the CPU, a few steps."""
import json
import os
import subprocess
import sys

import importlib.util

import pytest

from table_image_icl import photos

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("test_photos", os.path.join(ROOT, "tests", "test_photos.py"))
test_photos = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_photos)
make_shard = test_photos.make_shard


def test_train_runs_on_a_decoded_source_and_records_it(tmp_path):
    pytest.importorskip("tabicl")
    cache = tmp_path / "cache"
    classes = [f"c{i:03d}" for i in range(120)]                    # a tenth of the classes validates: 12, enough for a 10-class task
    shards = [make_shard(str(tmp_path / f"s{i}.tar"), n=600, seed=i, classes=classes) for i in range(2)]
    photos.decode(shards, photos.decoded_folder("imagenet1k", str(cache), shards=2), workers=2)     # the model's 224 px
    out = tmp_path / "run"
    cmd = [sys.executable, "scripts/train.py", "--out", str(out), "--encoder", "pixels", "--photos", "imagenet1k", "--shards", "2",
           "--photo-cache", str(cache), "--max-steps", "2", "--workers", "0", "--n-validation-batches", "1", "--validate-every", "1",
           "--batch-size", "2", "--encoder-layers", "1", "--patch-layers", "1",
           "--importance-sampling", "--distractor-share", "0.2", "--ordinal-confusion", "0.2", "--without-replacement", "--augment-test", "--skip-unlearnable"]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT}, timeout=900)
    assert result.returncode == 0, result.stderr[-3000:]
    record = json.load(open(out / "config.json"))
    assert record["photos"]["source"] == "imagenet1k" and record["photos"]["shards"] == 2 and record["photos"]["decoded"]["photos"] == 1200
    assert os.path.exists(out / "model.pt") and os.path.exists(out / "validation.pt")
    tasks = record["tasks"]
    assert tasks["importance_sampling"] and tasks["distractor_share"] == 0.2 and tasks["augment_test"] and tasks["skip_unlearnable"], "the flags are recorded"
    assert tasks["fine_grained_share"] == 0 and tasks["sibling_noise"] == 0
