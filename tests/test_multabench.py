"""MulTaBench loading and splits, on a small fake dataset folder: no downloads."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from table_image_icl import multabench


@pytest.fixture
def folder(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"age": rng.normal(50, 10, n), "side": rng.choice(["LEFT", "RIGHT"], n),
                       "report": [f"a long free text report number {i} " * 3 for i in range(n)],
                       "image": [f"images/{i}.png" for i in range(n)],
                       "target": rng.choice(["benign", "malignant", "normal"], n), "score": rng.normal(size=n)})
    df.loc[3, "age"] = np.nan
    df.to_csv(tmp_path / "data.csv", index=False)

    def with_target(target, task_type):
        json.dump({"target": target, "image_col": "image", "task_type": task_type}, open(tmp_path / "metadata.json", "w"))
        return str(tmp_path)

    return with_target


def test_a_classification_table(folder):
    data = multabench.load(folder("target", "cls"))
    assert data.is_classification and data.columns == ["age", "side", "score"], "free text and the image column are dropped"
    assert set(np.unique(data.y)) == {0, 1, 2} and data.y.dtype == np.int64
    assert np.isnan(data.table[3, 0]) and set(np.unique(data.table[:, 1])) == {0.0, 1.0}, "missing stays missing, categories become codes"
    assert data.paths[5].endswith("images/5.png")


def test_a_regression_table(folder):
    data = multabench.load(folder("score", "reg"))
    assert not data.is_classification and data.y.dtype == np.float64 and "score" not in data.columns


@pytest.mark.parametrize("k", range(6))
def test_splits_are_disjoint_capped_and_repeatable(k):
    y = np.random.default_rng(0).integers(0, 3, 40_000)
    train, test = multabench.split(y, True, k)
    assert len(set(train) & set(test)) == 0 and len(train) == 10_000 and len(test) == 5_000
    again = multabench.split(y, True, k)
    assert np.array_equal(train, again[0]) and np.array_equal(test, again[1])
    assert abs(np.mean(y[train] == 0) - np.mean(y == 0)) < 0.01, "class balance kept"


def test_the_three_folds_of_a_repeat_cover_every_row_once():
    y = np.random.default_rng(0).integers(0, 2, 900)
    assert sorted(np.concatenate([multabench.split(y, True, k)[1] for k in range(3)]).tolist()) == list(range(900))


def test_regression_splits_need_no_classes():
    train, test = multabench.split(np.random.default_rng(0).normal(size=600), False, 0)
    assert len(train) == 400 and len(test) == 200


def test_the_validation_sets_are_extended_datasets_and_the_rest_are_core():
    assert len(multabench.VALIDATION_SETS) == 4 and len(multabench.DEVELOPMENT_SETS) == 3 and len(multabench.TEST_SETS) == 8
    assert all(slug.startswith("multabench-full-") for slug in multabench.VALIDATION_SETS.values()), "the extended slice"
    core = list(multabench.DEVELOPMENT_SETS.values()) + list(multabench.TEST_SETS.values())
    assert all(slug.startswith("multabench-") and not slug.startswith("multabench-full-") for slug in core)
    assert len(multabench.ALL_SETS) == 15 and len(set(multabench.ALL_SETS.values())) == 15, "no dataset is in two sets"
    assert list(multabench.DATASET_SETS) == ["validation", "development", "test"]


def test_group_keys_read_patients_listings_and_fruits_off_the_benchmark_files():
    key = multabench.GROUP_KEYS
    pad = pd.DataFrame({"img_id": ["images/imgs_part_3_imgs_part_3_PAT_1516_1765_530.png", "images/imgs_part_1_PAT_46_881_939.png", "images/x_PAT_1516_1766_531.png"]})
    assert key["multabench-full-pad-ufes-lesion"](pad, "img_id").tolist() == ["PAT_1516", "PAT_46", "PAT_1516"]
    socal = pd.DataFrame({"street": ["755 Brawley Avenue", "755 brawley avenue ", "124 C Street W"], "citi": ["Brawley, CA"] * 3})
    assert key["multabench-full-socal-houses"](socal, "image_id").nunique() == 2
    chex = pd.DataFrame({"i": ["images/train_patient00005_study1_view1_frontal.jpg", "images/train_patient00005_study2_view1_frontal.jpg"]})
    assert key["multabench-chexpert"](chex, "i").tolist() == ["patient00005"] * 2
    assert key["multabench-mango-mass"](pd.DataFrame({"i": ["images/12a.jpg", "images/12b.jpg", "images/3a.jpg"]}), "i").tolist() == ["12", "12", "3"]
    assert key["multabench-mammography-cmmd"](pd.DataFrame({"i": ["images/Cancer_D1-0033_1-2.png", "images/Cancer_D1-0033_1-1.png"]}), "i").tolist() == ["D1-0033"] * 2
    assert key["multabench-petfinder"](pd.DataFrame({"i": ["images/86e1089a3-1.jpg", "images/86e1089a3-2.jpg"]}), "i").tolist() == ["86e1089a3"] * 2
    assert key["multabench-khaadi-clothes"](pd.DataFrame({"i": ["images/ALK231009_image_0.jpg", "images/ALK231009_image_1.jpg"]}), "i").tolist() == ["ALK231009"] * 2
    cbis = pd.DataFrame({"i": ["images/1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515_1-211.jpg"]})
    assert key["multabench-cbis-ddsm"](cbis, "i").tolist() == ["P_00001"]


def test_group_ids_are_none_without_a_key_and_fail_on_an_unmatched_row():
    assert multabench.group_ids("multabench-full-ham10000-lesion", pd.DataFrame({"i": ["a"]}), "i") is None
    assert multabench.group_ids("/cache/chico89/multabench-mango-mass/versions/1", pd.DataFrame({"i": ["images/7a.jpg"]}), "i").tolist() == ["7"]
    with pytest.raises(AssertionError):
        multabench.group_ids("multabench-chexpert", pd.DataFrame({"i": ["images/no-patient.jpg"]}), "i")


def test_a_grouped_split_keeps_every_group_on_one_side():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 8)
    y = rng.integers(0, 3, len(groups))
    for k in range(6):
        train, test = multabench.split(y, True, k, groups=groups)
        assert not set(groups[train]) & set(groups[test]) and len(train) + len(test) == len(y)
    a, b = multabench.split(y, True, 0, groups=groups), multabench.split(y, True, 0, groups=groups)
    assert np.array_equal(a[0], b[0])


def test_load_grouped_uses_the_dataset_s_group_key(folder, monkeypatch):
    path = folder("target", "cls")
    assert multabench.load(path).groups is None
    monkeypatch.setitem(multabench.GROUP_KEYS, os.path.basename(path), lambda df, c: df[c].str.extract(r"(\d)")[0])
    groups = multabench.load(path, grouped=True).groups
    assert len(groups) == 300 and groups[0] == "0" and groups[123] == "1"
