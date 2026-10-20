"""Published weights: fetched once from the release, checked by sha256, loaded like any run folder."""
import json
import os

import pytest
import torch

from table_image_icl import pretrained
from table_image_icl.model import ModelConfig, TableImageICL


def fake_release(tmp_path, monkeypatch, name="tiny"):
    """A release served from a local folder (file:// URLs) holding a small run's config.json and model.pt."""
    cfg = ModelConfig(encoder="pixels", encoder_layers=1, patch_layers=1)
    model = TableImageICL(cfg, pretrained=False)
    served = tmp_path / "served" / "releases" / "download" / "v9"
    served.mkdir(parents=True)
    record = {"model": cfg.__dict__, "tasks": {}, "train": {}}
    (served / f"{name}-config.json").write_text(json.dumps(record))
    torch.save(model.state_dict(), served / f"{name}-model.pt")
    digests = {f: pretrained.sha256(str(served / f"{name}-{f}")) for f in pretrained.FILES}
    monkeypatch.setitem(pretrained.RELEASES, name, {"tag": "v9", "sha256": digests})
    return model, "file://" + str(tmp_path / "served")


def test_fetch_downloads_checks_and_reuses(tmp_path, monkeypatch):
    model, base = fake_release(tmp_path, monkeypatch)
    folder = pretrained.fetch("tiny", str(tmp_path / "checkpoints"), base)
    assert sorted(os.listdir(folder)) == ["config.json", "model.pt"]
    stamp = os.path.getmtime(os.path.join(folder, "model.pt"))
    assert pretrained.fetch("tiny", str(tmp_path / "checkpoints"), base) == folder
    assert os.path.getmtime(os.path.join(folder, "model.pt")) == stamp, "present and matching: not fetched again"
    loaded, record = pretrained.load_pretrained("tiny", str(tmp_path / "checkpoints"), base=base)
    assert all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), model.state_dict().values()))
    assert not loaded.training and record["model"]["encoder"] == "pixels"


def test_a_wrong_checksum_is_refused(tmp_path, monkeypatch):
    _, base = fake_release(tmp_path, monkeypatch)
    pretrained.RELEASES["tiny"]["sha256"]["model.pt"] = "0" * 64
    with pytest.raises(AssertionError, match="sha256"):
        pretrained.fetch("tiny", str(tmp_path / "checkpoints"), base)
    assert not os.path.exists(tmp_path / "checkpoints" / "tiny" / "model.pt"), "nothing half-written is kept under the final name"


def test_unknown_names_are_refused():
    with pytest.raises(AssertionError, match="no release named"):
        pretrained.fetch("nothing-of-the-sort")
