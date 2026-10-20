"""The two scores, the predictions files, and the paired bootstrap over the test rows."""
import numpy as np
import pytest

from table_image_icl.scoring import auroc, bootstrap, load_predictions, markdown, merge, report, rmse, save_predictions, score


def test_auroc_is_one_for_perfect_and_half_for_uninformative():
    y = np.array([0, 1, 1, 0])
    assert auroc(y, np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3]]), np.array([0, 1])) == 1.0
    assert auroc(y, np.full((4, 2), 0.5), np.array([0, 1])) == 0.5


def test_auroc_leaves_out_classes_the_test_rows_never_show():
    y = np.array([0, 2, 2, 0])
    p = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.2, 0.1, 0.7], [0.7, 0.2, 0.1]])
    assert auroc(y, p, np.array([0, 1, 2])) == 1.0


def test_rmse():
    assert rmse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 0.0
    assert rmse(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(np.sqrt(12.5))
    assert score(np.array([1.0, 2.0]), np.array([1.0, 2.0]), False) == 0.0


def test_predictions_round_trip(tmp_path):
    records = {("CBIS-DDSM", 0): {"y": np.array([0, 1, 1]), "rows": np.array([5, 7, 9]),
                                  "methods": {"ours, table + image": (np.array([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]]), np.array([0, 1])),
                                              "TabICL v2, table only": (np.full((3, 2), 0.5), np.array([0, 1]))}},
               ("Mango Mass", 2): {"y": np.array([1.5, 2.5]), "rows": np.array([1, 3]), "methods": {"ours, table + image": np.array([1.4, 2.7])}}}
    path = str(tmp_path / "p.npz")
    save_predictions(path, records)
    back = load_predictions(path)
    assert set(back) == set(records) and np.array_equal(back[("CBIS-DDSM", 0)]["rows"], [5, 7, 9])
    p, classes = back[("CBIS-DDSM", 0)]["methods"]["ours, table + image"]
    assert np.allclose(p, [[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]]) and list(classes) == [0, 1]
    assert np.allclose(back[("Mango Mass", 2)]["methods"]["ours, table + image"], [1.4, 2.7])


def classification_splits(seed=0, n=200):
    rng = np.random.default_rng(seed)
    splits = []
    for _ in range(3):
        y = rng.integers(0, 3, n)
        perfect = np.eye(3)[y] * 0.8 + 0.1
        noisy = np.clip(perfect + rng.normal(0, 0.25, perfect.shape), 0.01, None)
        noisy = noisy / noisy.sum(1, keepdims=True)
        splits.append((y, {"perfect": (perfect, np.arange(3)), "noisy": (noisy, np.arange(3)),
                           "random": (rng.dirichlet(np.ones(3), n), np.arange(3)), "same as noisy": (noisy, np.arange(3))}))
    return splits


def test_bootstrap_intervals_and_paired_differences():
    out = bootstrap(classification_splits(), True, reference="random", n_resamples=300, seed=0)
    assert out["perfect"]["point"] == 1.0 and out["perfect"]["ci"] == (1.0, 1.0)
    assert out["random"]["ci"][0] < 0.5 < out["random"]["ci"][1]
    assert out["perfect"]["difference"]["ci"][0] > 0 and out["perfect"]["difference"]["p_better"] == 1.0
    assert out["random"]["difference"] == {"point": 0.0, "ci": (0.0, 0.0), "p_better": 0.0}, "the reference against itself"
    assert out["same as noisy"]["point"] == out["noisy"]["point"], "the same resamples for every method"


def test_bootstrap_resamples_within_each_class_so_every_class_stays():
    rng = np.random.default_rng(1)
    y = np.array([0] * 97 + [1] * 3)
    p = np.stack([1 - y + rng.normal(0, 0.1, 100), y + rng.normal(0, 0.1, 100)], 1)
    out = bootstrap([(y, {"m": (p, np.array([0, 1]))})], True, reference="m", n_resamples=200, seed=0)
    assert 0.9 < out["m"]["ci"][0] <= out["m"]["ci"][1] <= 1.0


def test_bootstrap_for_regression_counts_lower_rmse_as_better():
    rng = np.random.default_rng(2)
    splits = []
    for _ in range(3):
        y = rng.normal(size=150)
        splits.append((y, {"good": y + rng.normal(0, 0.1, 150), "bad": y + rng.normal(0, 1.0, 150)}))
    out = bootstrap(splits, False, reference="bad", n_resamples=300, seed=0)
    assert out["good"]["difference"]["ci"][1] < 0 and out["good"]["difference"]["p_better"] == 1.0


def test_merge_labels_our_methods_per_file_and_keeps_the_references_once(tmp_path):
    y, rows = np.array([0, 1, 1, 0]), np.array([1, 2, 3, 4])
    good = (np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.8, 0.2]]), np.array([0, 1]))
    flat = (np.full((4, 2), 0.5), np.array([0, 1]))
    for name, ours in [("a", good), ("b", flat)]:
        save_predictions(str(tmp_path / f"{name}.npz"), {("D", 0): {"y": y, "rows": rows, "methods": {"ours, table + image": ours, "TabICL v2, table only": flat}}})
    files = {"run a": str(tmp_path / "a.npz"), "run b": str(tmp_path / "b.npz")}
    assert set(merge(files)[("D", 0)][1]) == {"run a: ours, table + image", "run b: ours, table + image", "TabICL v2, table only"}
    result = report(files, "TabICL v2, table only", n_resamples=50)
    assert result["D"]["metric"] == "AUROC" and result["D"]["methods"]["run a: ours, table + image"]["point"] == 1.0
    assert "| run a: ours, table + image | 1.000" in markdown(result, "TabICL v2, table only")
