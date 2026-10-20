"""Generated tasks: a hidden variable shown only through each row's photo, and the four ways a batch is shown."""
import numpy as np
import pytest
import torch

from table_image_icl.tasks import Batch, TaskConfig, Tasks, can_show, hide_column, most_important_column, noisy_label, pick_images, place_on_slots, task_batches, values_to_bins, with_column


def test_hiding_removes_the_given_column_of_each_table():
    x = torch.arange(2 * 6 * 5, dtype=torch.float32).reshape(2, 6, 5)
    x[1, :, 3:] = 0                                              # table 1 has 3 real columns, 2 of padding
    which = torch.tensor([4, 1])
    kept, hidden, n_kept = hide_column(x, torch.tensor([5, 3]), which)
    assert kept.shape == (2, 6, 4) and n_kept.tolist() == [4, 2]
    for t in range(2):
        assert torch.equal(hidden[t], x[t, :, which[t]])
        assert torch.equal(kept[t], x[t][:, [c for c in range(5) if c != which[t]]]), "the other columns keep their order"


def test_bins_come_from_the_context_rows_and_keep_the_order():
    rng = np.random.default_rng(0)
    values = rng.normal(size=120)
    bins = values_to_bins(values, n_context=80, n_bins=5)
    assert set(bins[:80]) == set(range(5))
    assert (np.diff(bins[np.argsort(values)]) >= 0).all()
    changed = values.copy()
    changed[80:] *= 50
    assert np.array_equal(bins[:80], values_to_bins(changed, 80, 5)[:80])


def test_a_column_with_few_distinct_values_gets_one_bin_per_value():
    values = np.random.default_rng(0).choice([0.0, 1.0, 5.0], 90)
    bins = values_to_bins(values, n_context=60, n_bins=8)
    assert len(set(bins)) == 3 and all(len(set(values[bins == b])) == 1 for b in range(3))


def test_each_bin_shows_one_class_of_the_collection():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(12), 25)
    bins = rng.integers(0, 4, 200)
    index = pick_images(rng, {"labels": labels}, bins)
    shown = labels[index]
    class_of_bin = {b: set(shown[bins == b]) for b in range(4)}
    assert all(len(c) == 1 for c in class_of_bin.values()), "one class per bin"
    assert len(set.union(*class_of_bin.values())) == 4, "different bins show different classes"
    assert len(set(index[bins == 0])) > 1, "rows of a bin get different photos of its class"


def test_photos_labelled_minus_one_are_never_shown():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(8), 30)
    labels[labels % 2 == 1] = -1                                 # the odd classes belong to the other split
    collection = {"labels": labels}
    assert can_show(collection, 4) and not can_show(collection, 5)
    index = pick_images(rng, collection, rng.integers(0, 4, 100))
    assert (labels[index] >= 0).all() and set(labels[index]) <= {0, 2, 4, 6}


def test_the_most_important_column_is_the_one_the_label_depends_on():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 6)).astype(np.float32)
    y = (x[:, 4] > 0).astype(int)
    assert most_important_column(x, y, n_context=80) == 4


def test_a_noisy_label_shows_the_class_on_a_share_of_the_rows():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 400)
    shown = noisy_label(rng, y, n_context=300)
    assert set(shown) <= set(range(3))
    assert 0.3 + 0.7 / 3 - 0.08 < (shown == y).mean() < 1.0, "the class is shown on 30-100% of the rows, a random one otherwise"


def test_classes_land_on_distinct_random_slots_and_every_slot_gets_used():
    g = torch.Generator().manual_seed(0)
    y = torch.tensor([[0, 1, 2, 1, 0], [1, 1, 0, 0, 1]])
    slots = place_on_slots(y, n_slots=10, generator=g)
    for row, original in zip(slots, y):
        mapping = {int(o): int(s) for o, s in zip(original, row)}
        assert len(set(mapping.values())) == len(mapping) and all(int(s) == mapping[int(o)] for o, s in zip(original, row))
    many = torch.zeros(300, 4, dtype=torch.long)
    many[:, 2:] = 1
    assert set(place_on_slots(many, n_slots=10, generator=g).flatten().tolist()) == set(range(10))


def test_the_share_of_photo_as_label_batches_is_a_setting():
    pytest.importorskip("tabicl")
    collections = {"things": {"labels": np.repeat(np.arange(15), 40)}}
    for share, role in [(1.0, "label"), (0.0, "feature")]:
        cfg = TaskConfig(batch_size=2, rows=(32, 48), features=(3, 6), max_classes=4, max_bins=5, image_as_label_share=share)
        stream = task_batches(cfg, collections, seed=0)
        assert {next(stream).image_role for _ in range(6)} == {role}


def test_batches_tie_the_hidden_variable_to_the_photos():
    pytest.importorskip("tabicl")
    rng = np.random.default_rng(0)
    collections = {"things": {"labels": np.repeat(np.arange(15), 40)}, "more": {"labels": np.repeat(np.arange(9), 50)}}
    cfg = TaskConfig(batch_size=3, rows=(48, 64), features=(3, 8), max_classes=4, max_bins=5)
    stream = task_batches(cfg, collections, seed=0)
    roles = set()
    for _ in range(40):
        if roles == {"feature", "label"}:
            break
        b = next(stream)
        roles.add(b.image_role)
        n, n_ctx = b.x.shape[1], b.y_context.shape[1]
        assert b.image_index.shape == (3, n) and len(b.collection) == 3 and b.hidden.shape == (3, n)
        assert (b.n_columns >= 1).all() and b.x.shape[2] == int(b.n_columns.max())
        if b.image_role == "label":
            assert (b.bins.float() == b.hidden).all(), "the shown class is the hidden variable"
            continue
        for t in range(3):
            labels = collections[b.collection[t]]["labels"]
            bins = b.bins[t].numpy()
            assert set(bins[:n_ctx]) == set(bins), "every bin appears in the context"
            shown = labels[b.image_index[t].numpy()]
            assert all(len(set(shown[bins == k])) == 1 for k in set(bins)), "rows of one bin show one class"
    assert roles == {"feature", "label"}


def test_a_column_is_put_back_after_each_tables_real_columns():
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    x[1, :, 2:] = 0
    values = torch.tensor([[100.0, 101, 102], [200, 201, 202]])
    wider, n_columns = with_column(x, torch.tensor([4, 2]), values)
    assert wider.shape == (2, 3, 5) and n_columns.tolist() == [5, 3]
    assert torch.equal(wider[0, :, 4], values[0]) and torch.equal(wider[1, :, 2], values[1])
    assert (wider[1, :, 3:] == 0).all(), "padding stays padding"


def numbered_collection(n_images, labels):
    """Photo k is filled with the value k, so a looked-up photo says which one it is."""
    images = np.broadcast_to(np.arange(n_images, dtype=np.uint8)[:, None, None, None], (n_images, 28, 28, 3)).copy()
    return {"images": images, "labels": labels}


COLLECTIONS = {"seen": numbered_collection(120, np.repeat(np.arange(12), 10)), "unseen": numbered_collection(100, np.repeat(np.arange(10), 10))}


def tasks(p_no_image=0.15):
    cfg = TaskConfig(batch_size=2, rows=(24, 32), features=(3, 6), max_classes=3, max_bins=4)
    return Tasks(cfg, COLLECTIONS, ["seen"], ["unseen"], to_tokens=lambda images: torch.from_numpy(images).float(), p_no_image=p_no_image)


def fake_batch():
    x = torch.randn(2, 6, 3, generator=torch.Generator().manual_seed(0))
    index = torch.tensor([[0, 1, 2, 3, 4, 5], [10, 20, 30, 40, 50, 60]])
    return Batch(x, torch.tensor([[1, 2, 1, 2], [5, 5, 7, 7]]), torch.tensor([[1, 2], [5, 7]]), torch.tensor([3, 2]),
                 ["seen", "unseen"], index, hidden=torch.randn(2, 6), bins=torch.zeros(2, 6, dtype=torch.long), image_role="feature")


def test_the_image_view_looks_up_each_rows_photo_in_its_tables_collection():
    b = fake_batch()
    inputs = tasks().inputs(b, "cpu", "image")
    assert inputs["images"].shape == (2, 6, 28, 28, 3)
    assert torch.equal(inputs["images"][:, :, 0, 0, 0], b.image_index.float()) and torch.equal(inputs["x"], b.x)


def test_the_no_image_view_is_the_table_alone():
    inputs = tasks().inputs(fake_batch(), "cpu", "no image")
    assert "images" not in inputs and inputs["x"].shape == (2, 6, 3)


def test_the_column_visible_view_gives_the_hidden_column_back_instead_of_the_photo():
    b = fake_batch()
    inputs = tasks().inputs(b, "cpu", "column visible")
    assert "images" not in inputs and inputs["x"].shape == (2, 6, 4)
    assert torch.equal(inputs["x"][0, :, 3], b.hidden[0]) and torch.equal(inputs["x"][1, :, 2], b.hidden[1])


def test_the_shuffled_view_gives_rows_the_photos_of_other_rows():
    b, t = fake_batch(), tasks()
    shown = t.inputs(b, "cpu", "images shuffled")["images"][:, :, 0, 0, 0]
    assert not torch.equal(shown, b.image_index.float())
    assert torch.equal(shown.sort(1).values, b.image_index.float().sort(1).values), "the same photos, on other rows"
    assert torch.equal(shown, t.inputs(b, "cpu", "images shuffled")["images"][:, :, 0, 0, 0]), "the same shuffle every time"


def test_training_drops_the_photo_in_the_given_share_of_steps():
    t = tasks(p_no_image=0.25)
    views = [t.training_view() for _ in range(2000)]
    assert 0.21 < views.count("no image") / 2000 < 0.29
    assert {tasks(p_no_image=0.0).training_view() for _ in range(50)} == {"image"}


def test_training_and_validation_tasks_use_different_collections():
    pytest.importorskip("tabicl")
    t = tasks()
    assert set(next(iter(t.stream(seed=0))).collection) == {"seen"}
    assert set(next(t.validation_batches(seed=0)).collection) == {"unseen"}


def test_a_task_config_reads_a_record_and_ignores_what_it_does_not_know():
    cfg = TaskConfig.from_record({"batch_size": 2, "rows": [64, 256], "image_as_label_share": 1.0, "something_else": 3})
    assert cfg.batch_size == 2 and cfg.image_as_label_share == 1.0 and cfg.max_bins == 10


# ---------- additions to the prior, each behind a flag that is off by default ----------

from table_image_icl.tasks import augment, confuse_bins, distractor_bins, learnable_bins, learnable_label, pick_classes, siblings  # noqa: E402

PATHS = {0: ("root", "animal", "dog"), 1: ("root", "animal", "dog"), 2: ("root", "animal", "dog"),
         3: ("root", "animal", "cat"), 4: ("root", "animal", "cat"), 5: ("root", "artifact"), 6: ("root", "artifact")}
GROUPS = {1: {"animal": [0, 1, 2, 3, 4], "artifact": [5, 6]}, 2: {"dog": [0, 1, 2], "cat": [3, 4]}}


def hierarchy_collection():
    return {"labels": np.repeat(np.arange(8), 25), "paths": PATHS, "groups": GROUPS}


def test_a_default_task_config_has_every_addition_off():
    cfg = TaskConfig()
    assert not cfg.importance_sampling and cfg.distractor_share == 0 and cfg.fine_grained_share == 0
    assert cfg.sibling_noise == 0 and cfg.ordinal_confusion == 0
    assert not cfg.without_replacement and not cfg.augment_test and not cfg.skip_unlearnable


def test_same_seed_same_batches():
    pytest.importorskip("tabicl")
    cfg = TaskConfig(batch_size=2, rows=(32, 48), features=(3, 6), max_classes=4, max_bins=5)
    collections = {"things": {"labels": np.repeat(np.arange(15), 40)}}
    def first(n):                                                # one stream at a time: the prior draws from the global generators
        stream = task_batches(cfg, collections, seed=3)
        return [next(stream) for _ in range(n)]
    for x, y in zip(first(3), first(3)):
        assert torch.equal(x.image_index, y.image_index) and torch.equal(x.bins, y.bins) and x.image_role == y.image_role


def test_importance_sampling_draws_columns_in_proportion():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 6)).astype(np.float32)
    y = ((x[:, 4] + 0.4 * x[:, 1]) > 0).astype(int)
    assert most_important_column(x, y, n_context=150) == 4, "without a generator: the argmax, as before"
    draws = [most_important_column(x, y, 150, rng=np.random.default_rng(i)) for i in range(60)]
    assert max(set(draws), key=draws.count) == 4 and 1 in draws, "column 4 is modal, column 1 is drawn sometimes"


def test_distractor_bins_keep_the_count_and_lose_the_link():
    rng = np.random.default_rng(0)
    bins = np.repeat(np.arange(4), 50)
    shown = distractor_bins(rng, bins)
    assert shown.shape == bins.shape and set(shown) <= set(range(4))
    assert (shown != bins).mean() > 0.5, "the photo's class no longer follows the bin"


def test_distractor_photos_are_independent_of_the_bins():
    pytest.importorskip("tabicl")
    cfg = TaskConfig(batch_size=2, rows=(48, 64), features=(3, 8), max_classes=4, max_bins=5, image_as_label_share=0.0, distractor_share=1.0)
    collections = {"things": {"labels": np.repeat(np.arange(15), 40)}}
    b = next(task_batches(cfg, collections, seed=0))
    for t in range(2):
        shown = collections["things"]["labels"][b.image_index[t].numpy()]
        bins = b.bins[t].numpy()
        assert all(len(set(shown[bins == k])) > 1 for k in set(bins) if (bins == k).sum() > 8), "rows of one bin show several classes"
        assert len(set(bins)) == len(set(b.hidden[t].numpy().round(6))) or b.image_role == "feature"


def test_pick_classes_defaults_to_a_random_draw_and_takes_one_group_when_fine_grained():
    c = hierarchy_collection()
    plain = pick_classes(np.random.default_rng(0), c, 3)
    assert len(set(plain.tolist())) == 3 and set(plain.tolist()) <= set(range(8))
    for seed in range(10):
        fine = pick_classes(np.random.default_rng(seed), c, 3, fine_grained_share=1.0)
        assert set(fine.tolist()) in ({0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 2, 3}, {0, 2, 4}, {0, 3, 4}, {1, 2, 3}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4}), "three classes under one ancestor"
    assert len(set(pick_classes(np.random.default_rng(0), c, 6, fine_grained_share=1.0).tolist())) == 6, "no group has 6 classes: the random draw"
    assert len(set(pick_classes(np.random.default_rng(0), {"labels": c["labels"]}, 3, fine_grained_share=1.0).tolist())) == 3, "no hierarchy at all"


def test_pick_images_takes_the_classes_it_is_given():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(12), 25)
    bins = rng.integers(0, 3, 100)
    index = pick_images(rng, {"labels": labels}, bins, classes=np.array([7, 2, 9]))
    assert all(set(labels[index][bins == b]) == {c} for b, c in enumerate([7, 2, 9]))


def test_siblings_share_the_deepest_ancestor():
    classes = np.array([0, 1, 3, 5, 6])
    assert siblings(0, classes, PATHS) == [1], "the other dog"
    assert siblings(3, classes, PATHS) == [0, 1], "no other cat: the nearest are the animals"
    assert siblings(5, classes, PATHS) == [6]
    assert siblings(5, np.array([5, 0]), PATHS) == [], "nothing shares more than the root"
    assert siblings(0, classes, {}) == [], "no hierarchy: nothing"


def test_sibling_noise_confuses_only_nearest_classes():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    near = {0: [1], 1: [0], 2: [3], 3: [2]}
    shown = noisy_label(rng, y, n_context=300, share=0.5, siblings=near, sibling_noise=1.0)
    wrong = shown != y
    assert 0.3 < wrong.mean() < 0.7 and all(shown[i] in near[y[i]] for i in np.flatnonzero(wrong))
    uniform = noisy_label(np.random.default_rng(0), y, 300, share=0.5)
    assert set(uniform) <= set(range(4))


def test_confuse_bins_moves_rows_to_a_neighbouring_bin():
    rng = np.random.default_rng(0)
    bins = rng.integers(0, 5, 1000)
    moved = confuse_bins(rng, bins, 0.3)
    assert np.abs(moved - bins).max() <= 1 and moved.min() >= 0 and moved.max() <= 4
    assert 0.2 < (moved != bins).mean() < 0.35
    assert np.array_equal(confuse_bins(rng, bins, 0.0), bins)


def test_photos_are_drawn_without_replacement_when_there_are_enough():
    labels = np.repeat(np.arange(4), 25)
    few = pick_images(np.random.default_rng(0), {"labels": labels}, np.zeros(20, dtype=int), classes=np.array([1]), replace=False)
    assert len(set(few.tolist())) == 20
    many = pick_images(np.random.default_rng(0), {"labels": labels}, np.zeros(30, dtype=int), classes=np.array([1]), replace=False)
    assert len(set(many.tolist())) == 25, "more rows than photos: repeats, but every photo is used"


def test_augment_flips_or_crops_and_keeps_the_shape():
    rng = np.random.default_rng(0)
    photos = np.random.default_rng(1).integers(0, 256, (6, 32, 32, 3), dtype=np.uint8)
    out = augment(rng, photos)
    assert out.shape == photos.shape and out.dtype == np.uint8
    assert any(not np.array_equal(out[i], photos[i]) for i in range(6))


def test_test_rows_are_augmented_only_in_the_training_image_view():
    t = tasks()
    t.cfg = TaskConfig(batch_size=2, rows=(24, 32), features=(3, 6), max_classes=3, max_bins=4, augment_test=True)
    for c in t.collections.values():                             # photos with structure, so a flip or crop shows
        c["images"] = np.broadcast_to(np.arange(28, dtype=np.uint8)[None, None, :, None], c["images"].shape).copy()
    b = fake_batch()
    plain = t.inputs(b, "cpu", "image")["images"]
    trained = t.inputs(b, "cpu", "image", training=True)["images"]
    n_context = b.y_context.shape[1]
    assert torch.equal(trained[:, :n_context], plain[:, :n_context]), "context photos are never touched"
    assert not torch.equal(trained[:, n_context:], plain[:, n_context:]), "test photos are augmented"
    assert torch.equal(t.inputs(b, "cpu", "image")["images"], plain), "validation sees the photos as they are"


def test_the_learnability_rules():
    assert not learnable_bins(np.zeros(10, dtype=int)) and learnable_bins(np.array([0, 1, 1, 0]))
    assert not learnable_label(0.3, 10) and learnable_label(0.9, 2) and learnable_label(0.45, 10) and not learnable_label(0.4, 10)


def test_unlearnable_batches_are_skipped():
    pytest.importorskip("tabicl")
    cfg = TaskConfig(batch_size=2, rows=(32, 48), features=(3, 6), max_classes=4, max_bins=5, skip_unlearnable=True)
    collections = {"things": {"labels": np.repeat(np.arange(15), 40)}}
    stream = task_batches(cfg, collections, seed=0)
    for _ in range(10):
        b = next(stream)
        assert all(learnable_bins(bins) for bins in b.bins.numpy())


def test_task_stream_carries_class_level_extras_only():
    from table_image_icl.tasks import TaskStream
    c = {"things": {**hierarchy_collection(), "images": np.zeros((200, 4, 4, 3), np.uint8)}}
    s = TaskStream(TaskConfig(), c, seed=0)
    assert set(s.collections["things"]) == {"labels", "paths", "groups"}
