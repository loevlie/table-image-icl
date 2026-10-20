"""One photo source for the generated tasks: webdataset shards decoded once into an array, rare classes left out,
classes split into training and validation. The tests build a tiny shard themselves; nothing is downloaded."""
import io
import json
import os
import tarfile

import numpy as np
import pytest
from PIL import Image

from table_image_icl import photos

CLASSES = [f"n{1440764 + 7 * i:08d}" for i in range(12)]          # a table shows up to 10 classes, so the fixture has more


def make_shard(path, n=24, seed=0, suffix=".webp", classes=CLASSES):
    """A webdataset tar as timm publishes ImageNet: per sample a photo, a .cls class index and a .json with the name."""
    rng = np.random.default_rng(seed)
    with tarfile.open(path, "w") as tar:
        for i in range(n):
            label = i % len(classes)
            photo = Image.fromarray(rng.integers(0, 255, (40 + 10 * i, 60, 3), dtype=np.uint8))
            buf = io.BytesIO()
            photo.save(buf, format="WEBP" if suffix == ".webp" else "JPEG")
            for member, data in [(".cls", str(label).encode()), (".json", json.dumps({"label": label, "class_name": classes[label]}).encode()), (suffix, buf.getvalue())]:
                info = tarfile.TarInfo(f"sample{i:04d}{member}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return path


def test_read_shard_yields_each_samples_photo_class_and_name(tmp_path):
    samples = list(photos.read_shard(make_shard(str(tmp_path / "s.tar"), n=5)))
    assert [s[2] for s in samples] == [0, 1, 2, 3, 4] and samples[3][3] == CLASSES[3]
    assert Image.open(io.BytesIO(samples[0][1])).size == (60, 40)


def test_decode_writes_a_disk_backed_collection_in_shard_order(tmp_path):
    shards = [make_shard(str(tmp_path / f"s{i}.tar"), n=24, seed=i) for i in range(2)]
    out = photos.decode(shards, str(tmp_path / "decoded"), size=32, workers=2)
    p = photos.load_photos(out)
    assert p["images"].shape == (48, 32, 32, 3) and p["images"].dtype == np.uint8 and isinstance(p["images"], np.memmap)
    assert p["labels"].tolist() == [i % 12 for i in range(24)] * 2 and list(p["names"]) == CLASSES
    assert p["record"]["photos"] == 48 and p["record"]["shards"] == ["s0.tar", "s1.tar"]
    first = photos.letterbox(Image.open(io.BytesIO(next(photos.read_shard(shards[0]))[1])), 32)
    assert np.array_equal(p["images"][0], first), "the same letterbox as inference"
    with pytest.raises(FileExistsError):
        photos.decode(shards, out, size=32, workers=1)


def test_the_wordnet_id_is_read_from_the_key_or_filename_when_the_metadata_has_no_name(tmp_path):
    """timm's ImageNet-1k shards: no class_name, but the key and the filename start with the WordNet id."""
    path = str(tmp_path / "s.tar")
    with tarfile.open(path, "w") as tar:
        for i, (wnid, label) in enumerate([("n02097047", 196), ("n01682714", 40)]):
            buf = io.BytesIO()
            Image.fromarray(np.zeros((20, 20, 3), np.uint8)).save(buf, format="JPEG")
            for suffix, data in [(".cls", str(label).encode()), (".json", json.dumps({"label": label, "filename": f"{wnid}/{wnid}_{i}.JPEG"}).encode()), (".jpg", buf.getvalue())]:
                info = tarfile.TarInfo(f"{wnid}_{i}{suffix}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    assert [s[3] for s in photos.read_shard(path)] == ["n02097047", "n01682714"]
    assert photos.wnid_of("n02097047_2079") == "n02097047" and photos.wnid_of("sample0001") is None


def test_jpeg_shards_decode_too(tmp_path):
    out = photos.decode([make_shard(str(tmp_path / "j.tar"), n=6, suffix=".jpg")], str(tmp_path / "d"), size=16, workers=1)
    assert photos.load_photos(out)["images"].shape == (6, 16, 16, 3)


def test_photos_can_be_read_into_memory_and_capped(tmp_path):
    out = photos.decode([make_shard(str(tmp_path / "s.tar"), n=30)], str(tmp_path / "d"), size=16, workers=1)
    on_disk, in_memory = photos.load_photos(out), photos.load_photos(out, in_memory=True)
    assert not isinstance(in_memory["images"], np.memmap) and np.array_equal(on_disk["images"], in_memory["images"])
    assert len(photos.load_photos(out, max_photos=7)["labels"]) == 7


def test_rare_classes_are_never_shown_and_the_class_split_is_disjoint():
    labels = np.array([0] * 20 + [1] * 20 + [2] * 3 + [3] * 20 + [4] * 20 + [5] * 20)
    p = {"images": np.zeros((103, 4, 4, 3), np.uint8), "labels": labels, "names": np.array([str(i) for i in range(6)], dtype=object)}
    train, validation = photos.split_classes(p, validation_share=0.4, min_photos=10, seed=0)
    assert train["images"] is p["images"] and validation["images"] is p["images"], "the photos are shared, only the labels differ"
    seen_train, seen_val = set(train["labels"][train["labels"] >= 0]), set(validation["labels"][validation["labels"] >= 0])
    assert seen_train.isdisjoint(seen_val) and seen_train | seen_val == {0, 1, 3, 4, 5}, "class 2 has 3 photos: left out of both"
    assert len(seen_val) == 2
    assert (train["labels"] == -1).sum() + (validation["labels"] == -1).sum() == 103 + 3, "every photo is masked on exactly one side, class 2 on both"
    again, _ = photos.split_classes(p, validation_share=0.4, min_photos=10, seed=0)
    assert np.array_equal(again["labels"], train["labels"]), "the split is a function of the seed"


def test_sources_are_pinned_to_a_revision():
    for name, source in photos.SOURCES.items():
        assert len(source.revision) == 40, name
        assert source.shards >= 1


# ---------- the WordNet hierarchy: fine-grained tasks need to know which classes are near ----------

PATHS = {"a": ["root", "animal", "dog"], "b": ["root", "animal", "dog"], "c": ["root", "animal", "dog"],
         "d": ["root", "animal", "cat"], "e": ["root", "animal", "cat"], "f": ["root", "artifact"], "g": ["root", "artifact"], "h": []}


def hierarchy_collection(masked=()):
    labels = np.repeat(np.arange(8), 20)
    labels = np.where(np.isin(labels, list(masked)), -1, labels)
    names = np.array(list("abcdefgh"), dtype=object)
    return {"images": np.zeros((160, 4, 4, 3), np.uint8), "labels": labels, "names": names, "hierarchy": PATHS}


def test_class_paths_and_groups_follow_the_hierarchy():
    c = hierarchy_collection(masked=(2,))
    paths = photos.class_paths(c["labels"], c["names"], c["hierarchy"])
    assert set(paths) == {0, 1, 3, 4, 5, 6}, "masked classes and classes without a path are left out"
    assert paths[0] == ("root", "animal", "dog") and paths[5] == ("root", "artifact")
    groups = photos.class_groups(paths, depths=range(1, 4))
    assert groups[1] == {"animal": [0, 1, 3, 4], "artifact": [5, 6]}
    assert groups[2] == {"dog": [0, 1], "cat": [3, 4]}, "a group of one class is dropped"
    assert 3 not in groups, "no class has a path that deep"


def test_hierarchy_json_is_read_and_travels_with_the_split(tmp_path):
    out = photos.decode([make_shard(str(tmp_path / "s.tar"), n=240)], str(tmp_path / "d"), size=16, workers=1)
    with open(os.path.join(out, "hierarchy.json"), "w") as f:
        json.dump({CLASSES[i]: ["root", "animal" if i < 6 else "artifact", f"kind{i // 2}"] for i in range(12)}, f)
    loaded = photos.load_photos(out)
    assert loaded["hierarchy"][CLASSES[0]] == ["root", "animal", "kind0"]
    train, validation = photos.split_classes(loaded, validation_share=0.25, min_photos=10, seed=0)
    for split in (train, validation):
        shown = set(split["labels"][split["labels"] >= 0])
        assert set(split["paths"]) == shown, "paths only for the classes this split shows"
        assert all(set(members) <= shown for groups in split["groups"].values() for members in groups.values())
    assert photos.load_photos(str(tmp_path / "d")).get("hierarchy") is not None


def test_a_folder_without_a_hierarchy_still_loads(tmp_path):
    out = photos.decode([make_shard(str(tmp_path / "s.tar"), n=24)], str(tmp_path / "d"), size=16, workers=1)
    train, _ = photos.split_classes(photos.load_photos(out))
    assert train["paths"] == {} and train["groups"] == {}


def test_wordnet_paths_come_from_nltk_and_tolerate_unknown_ids():
    nltk = pytest.importorskip("nltk")
    try:
        from nltk.corpus import wordnet
        wordnet.synsets("dog")
    except LookupError:
        pytest.skip("WordNet corpus not downloaded")
    paths = photos.wordnet_paths(["n02084071", "c003", "n99999999"])
    assert paths["n02084071"][0] == "n00001740" and "n02084071" not in paths["n02084071"], "ancestors root-first, self excluded"
    assert paths["c003"] == [] and paths["n99999999"] == []
