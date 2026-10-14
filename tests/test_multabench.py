"""MulTaBench loading and splits, on a small fake dataset folder: no downloads."""
import json

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


def test_the_validation_and_test_sets_do_not_overlap():
    assert not set(multabench.VALIDATION_SETS) & set(multabench.TEST_SETS)
    assert len(multabench.VALIDATION_SETS) == 4 and len(multabench.TEST_SETS) == 8
