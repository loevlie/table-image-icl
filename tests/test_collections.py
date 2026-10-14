"""Which collections train and which validate; the cache; no downloads."""
import numpy as np

from table_image_icl.collections import COLLECTIONS, TRAIN_NAMES, VALIDATION_NAMES, cached, split_attributes


def test_training_and_validation_collections_do_not_overlap():
    assert set(VALIDATION_NAMES).isdisjoint(TRAIN_NAMES)
    assert set(TRAIN_NAMES) | set(VALIDATION_NAMES) <= set(COLLECTIONS) | {"celeba-heldout"}


def test_split_attributes_shares_photos_but_no_attribute():
    labels = np.arange(60).reshape(6, 10) % 2 == 0
    collection = {"images": np.zeros((6, 4, 4, 3), np.uint8), "labels": labels, "kind": "attr", "names": np.array([f"a{i}" for i in range(10)])}
    train, heldout = split_attributes(collection, n_heldout=3)
    assert train["labels"].shape == (6, 7) and heldout["labels"].shape == (6, 3)
    assert list(heldout["names"]) == ["a7", "a8", "a9"] and train["images"] is collection["images"]


def test_a_cached_collection_is_decoded_once_and_read_back_unchanged(tmp_path):
    calls = []

    def build():
        calls.append(1)
        return {"images": np.full((3, 4, 4, 3), 7, np.uint8), "labels": np.array([0, 1, 1]), "kind": "class"}

    path = str(tmp_path / "things-3.npz")
    first, second = cached(path, build), cached(path, build)
    assert len(calls) == 1 and [f.name for f in tmp_path.iterdir()] == ["things-3.npz"]
    assert np.array_equal(first["images"], second["images"]) and second["kind"] == "class"
    assert first["sha256"] == second["sha256"] and len(first["sha256"]) == 64, "the file's checksum travels with the data"
    assert "sha256" not in cached(None, build) and len(calls) == 2, "no path: nothing is cached"
