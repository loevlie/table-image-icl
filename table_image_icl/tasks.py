"""Training tasks: synthetic tables from TabICL v2's prior, where every row also gets a photo. The photo stands for
a hidden variable, in one of two roles drawn per batch:

    photo as feature   the column a random forest leans on most is removed from the table and cut into bins, and
                       the photo shows the bin;
    photo as label     the table stays whole and the photo shows the row's class for a share of the rows
                       (30-100%, drawn per task) and a random class for the rest.

Each bin (or shown class) is given one class of the photo collection and each row gets a photo of that class. A
collection is {"images", "labels"}; a label of -1 marks a photo the collection never shows (photos.py masks the
other split's classes and rare classes that way). A batch carries photo indices, not pixels; the pixels are looked
up when the batch is shown to the model.

Additions, each behind a TaskConfig flag that is off by default (the default generator is unchanged):
    importance_sampling   the hidden column is drawn in proportion to its importance, not always the top one
    distractor_share      a share of tables whose photos say nothing about their rows
    fine_grained_share    a share of tables whose classes are neighbours in the WordNet hierarchy (photos.py)
    sibling_noise         a wrong photo in the label role comes from the nearest other class of the task
    ordinal_confusion     a row in the feature role may show its neighbouring bin, so bin order shows in the photos
    without_replacement   a bin's photos are distinct when the class has enough
    augment_test          test-row photos are flipped and cropped a little during training
    skip_unlearnable      tables whose photo says nothing learnable are skipped"""
import random
from dataclasses import dataclass
from typing import Callable, Iterator, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from table_image_icl.model import N_SLOTS
from table_image_icl.predict import bin_targets

MIN_AGREEMENT = 0.5                             # skip_unlearnable: the photo must show the true class more often than not


@dataclass(frozen=True)
class TaskConfig:
    batch_size: int = 4
    rows: tuple[int, int] = (64, 256)
    features: tuple[int, int] = (3, 33)         # before hiding a column
    max_classes: int = 10
    context_share: tuple[float, float] = (0.3, 0.9)
    max_bins: int = 10                          # bins of the hidden column = photo classes shown in a task
    image_as_label_share: float = 0.5           # share of batches where the photo stands for the label
    prior_type: str = "graph_scm"               # TabICL v2's prior
    importance_sampling: bool = False
    distractor_share: float = 0.0
    fine_grained_share: float = 0.0
    sibling_noise: float = 0.0
    ordinal_confusion: float = 0.0              # the largest per-table rate; the rate is drawn from 0 to it
    without_replacement: bool = False
    augment_test: bool = False
    skip_unlearnable: bool = False

    @classmethod
    def from_record(cls, record: dict) -> "TaskConfig":
        return cls(**{k: v for k, v in record.items() if k in cls.__dataclass_fields__})


class Batch(NamedTuple):
    x: Tensor              # (batch, rows, columns) the table without the hidden column, zero-padded to the widest
    y_context: Tensor      # (batch, n_context) label slots
    y_test: Tensor         # (batch, rows - n_context)
    n_columns: Tensor      # (batch,) real columns per table
    collection: list       # per table: the photo collection its photos come from
    image_index: Tensor    # (batch, rows) index of each row's photo in that collection
    hidden: Tensor         # (batch, rows) the hidden variable as a number (the column's values, or the shown class)
    bins: Tensor           # (batch, rows) the bin of each row's hidden variable = which photo class it was shown as
    image_role: str        # "feature" or "label"


# ---------- tables from the prior ----------

def place_on_slots(y: Tensor, n_slots: int, generator: torch.Generator | None = None) -> Tensor:
    """y (batch, rows) with classes 0..K-1 -> the same labels on K distinct random slots, per task, so that no slot
    comes to mean anything."""
    slots = torch.stack([torch.randperm(n_slots, generator=generator) for _ in range(len(y))])
    return slots.gather(1, y)


def prior_tables(cfg: TaskConfig, seed: int = 0) -> Iterator[tuple[Tensor, Tensor, Tensor, Tensor]]:
    """An endless stream of (x, y_context, y_test, n_columns) from TabICL's prior. All tables of a batch share
    their numbers of rows and context rows."""
    from tabicl.prior import PriorDataset

    torch.manual_seed(seed)                # the prior draws from the global generators
    np.random.seed(seed)
    random.seed(seed)
    prior = PriorDataset(batch_size=cfg.batch_size, batch_size_per_gp=cfg.batch_size, min_features=cfg.features[0],
                         max_features=cfg.features[1], max_classes=cfg.max_classes, min_seq_len=cfg.rows[0], max_seq_len=cfg.rows[1],
                         min_train_size=cfg.context_share[0], max_train_size=cfg.context_share[1], prior_type=cfg.prior_type, n_jobs=1)
    generator = torch.Generator().manual_seed(seed)
    while True:
        x, y, n_columns, _, train_sizes = prior.get_batch()
        n_context = int(train_sizes[0])
        slots = place_on_slots(y.long(), N_SLOTS, generator)
        yield x[:, :, : int(n_columns.max())].float(), slots[:, :n_context], slots[:, n_context:], n_columns


# ---------- the hidden variable and its photos ----------

def most_important_column(x: np.ndarray, y: np.ndarray, n_context: int, rng: np.random.Generator | None = None) -> int:
    """The column a random forest fitted on the context rows relies on most; with `rng`, a column drawn in proportion
    to its importance instead (importance_sampling), so the photo can stand for a marginal column too."""
    from sklearn.ensemble import RandomForestClassifier
    importance = RandomForestClassifier(n_estimators=50, random_state=0).fit(x[:n_context], y[:n_context]).feature_importances_
    if rng is None:
        return int(importance.argmax())
    return int(rng.choice(len(importance), p=importance / importance.sum() if importance.sum() > 0 else None))


def hide_column(x: Tensor, n_columns: Tensor, which: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Remove column `which` (batch,) from every table: the tables one column narrower, the removed values
    (batch, rows), and the new numbers of real columns."""
    batch, rows, columns = x.shape
    keep = torch.arange(columns).expand(batch, -1) != which[:, None]
    kept = x.transpose(1, 2)[keep].reshape(batch, columns - 1, rows).transpose(1, 2)
    return kept, x[torch.arange(batch), :, which], n_columns - 1


def noisy_label(rng: np.random.Generator, y: np.ndarray, n_context: int, share: float | None = None,
                siblings: dict | None = None, sibling_noise: float = 0.0) -> np.ndarray:
    """What the photo shows when it stands for the label: the row's class for a share of the rows, a random class of
    the task for the others. The share is drawn per task from 0.3-1 unless given. With `siblings` (class -> the
    nearest other classes of the task), a wrong row shows one of them with probability sibling_noise."""
    share = rng.uniform(0.3, 1.0) if share is None else share
    classes = np.unique(y[:n_context])
    shown = np.where(rng.random(len(y)) < share, y, rng.choice(classes, len(y)))
    if siblings and sibling_noise > 0:
        for i in np.flatnonzero((shown != y) & (rng.random(len(y)) < sibling_noise)):
            near = [c for c in siblings.get(int(y[i]), []) if c in classes]
            if near:
                shown[i] = rng.choice(near)
    return shown


def values_to_bins(values: np.ndarray, n_context: int, n_bins: int) -> np.ndarray:
    """A column cut into at most n_bins bins at the quantiles of its context rows."""
    return bin_targets(values, n_context, n_bins)[0]


def confuse_bins(rng: np.random.Generator, bins: np.ndarray, rate: float) -> np.ndarray:
    """The bins whose photos the rows show: each row moves to a neighbouring bin with probability `rate`, so adjacent
    bins become confusable and the order of the bins shows in the photos (ordinal_confusion)."""
    if rate <= 0:
        return bins
    step = rng.choice([-1, 1], len(bins))
    moved = np.clip(bins + step, 0, bins.max())
    return np.where(rng.random(len(bins)) < rate, moved, bins)


def distractor_bins(rng: np.random.Generator, bins: np.ndarray) -> np.ndarray:
    """Bins drawn independently of the rows (distractor_share): the photos then say nothing about their rows."""
    return rng.integers(0, bins.max() + 1, len(bins))


def shown_classes(collection: dict) -> np.ndarray:
    """The classes a collection shows: every label but -1."""
    labels = collection["labels"]
    return np.unique(labels[labels >= 0])


def pick_classes(rng: np.random.Generator, collection: dict, k: int, fine_grained_share: float = 0.0) -> np.ndarray:
    """k distinct classes for a task: random ones, or, with probability fine_grained_share, k classes under one
    WordNet ancestor (collection["groups"]: depth -> ancestor -> classes, from photos.class_groups), the depth drawn
    among those with a group large enough; random when there is none."""
    if fine_grained_share > 0 and rng.random() < fine_grained_share:
        candidates = {d: [g for g, members in groups.items() if len(members) >= k] for d, groups in collection.get("groups", {}).items()}
        depths = [d for d, names in candidates.items() if names]
        if depths:
            depth = int(rng.choice(depths))
            group = str(rng.choice(candidates[depth]))
            return rng.choice(collection["groups"][depth][group], k, replace=False)
    return rng.permutation(shown_classes(collection))[:k]


def pick_images(rng: np.random.Generator, collection: dict, bins: np.ndarray, classes: np.ndarray | None = None, replace: bool = True) -> np.ndarray:
    """One photo per row: bin b shows class classes[b] (random distinct classes when not given) and rows draw photos
    of it, with replacement unless there are enough photos and replace is False."""
    labels = collection["labels"]
    if classes is None:
        classes = rng.permutation(shown_classes(collection))[: bins.max() + 1]
    membership = np.full(len(labels), -1)
    for b, c in enumerate(classes):
        membership[labels == c] = b
    index = np.empty(len(bins), dtype=np.int64)
    for b in np.unique(bins):
        pool, n = np.flatnonzero(membership == b), int((bins == b).sum())
        if replace or n <= len(pool):
            index[bins == b] = rng.choice(pool, size=n, replace=replace)
        else:                                                    # more rows than photos: every photo once, then repeats
            index[bins == b] = rng.permutation(np.concatenate([pool, rng.choice(pool, n - len(pool))]))
    return index


def siblings(c: int, classes: np.ndarray, paths: dict) -> list[int]:
    """The other classes of a task nearest to c in the hierarchy: those sharing the longest ancestor path with it,
    beyond the root. Empty when nothing is nearer than the root, or there is no hierarchy."""
    mine = paths.get(int(c), ())
    shared = {}
    for other in classes:
        if int(other) == int(c):
            continue
        theirs = paths.get(int(other), ())
        n = 0
        while n < min(len(mine), len(theirs)) and mine[n] == theirs[n]:
            n += 1
        if n >= 2:
            shared[int(other)] = n
    if not shared:
        return []
    best = max(shared.values())
    return [other for other, n in shared.items() if n == best]


def can_show(collection: dict, n_bins: int) -> bool:
    return len(shown_classes(collection)) >= n_bins


def learnable_bins(bins: np.ndarray) -> bool:
    return int(bins.max()) >= 1


def learnable_label(share: float, k: int) -> bool:
    """The photo shows the true class more often than not (a wrong photo lands on the true class 1/k of the time)."""
    return share + (1 - share) / k >= MIN_AGREEMENT


def task_batches(cfg: TaskConfig, collections: dict, seed: int = 0) -> Iterator[Batch]:
    """collections: name -> {"labels": class ids (photos,), -1 for photos never shown; optional "paths", "groups"}."""
    rng = np.random.default_rng(seed)
    names = sorted(collections)
    for x, y_context, y_test, n_columns in prior_tables(cfg, seed):
        role = "label" if rng.random() < cfg.image_as_label_share else "feature"
        if role == "feature" and (n_columns < 2).any():
            continue                                             # a table needs a column to hide and one to keep
        n_context = y_context.shape[1]
        y = torch.cat([y_context, y_test], 1).numpy()
        chosen = [None] * len(x)                                 # per table: the photo classes, when drawn before the bins
        if role == "feature":
            which = [most_important_column(x[t, :, : int(n_columns[t])].numpy(), y[t], n_context, rng if cfg.importance_sampling else None) for t in range(len(x))]
            x, hidden, n_columns = hide_column(x, n_columns, torch.tensor(which))
            all_bins = [values_to_bins(values, n_context, int(rng.integers(2, cfg.max_bins + 1))) for values in hidden.numpy()]
            if cfg.skip_unlearnable and not all(learnable_bins(b) for b in all_bins):
                continue
        else:
            all_bins, shares = [], []
            for t, labels in enumerate(y):
                share = rng.uniform(0.3, 1.0)
                near = None
                classes = np.unique(labels)
                able = [n for n in names if can_show(collections[n], len(classes))]
                if cfg.sibling_noise > 0 and able:               # the photo classes first, so that nearness is defined
                    name = str(rng.choice(able))
                    photo_classes = pick_classes(rng, collections[name], len(classes), cfg.fine_grained_share)
                    by_photo = {int(p): int(c) for c, p in zip(classes, photo_classes)}
                    paths = collections[name].get("paths", {})
                    near = {int(c): [by_photo[s] for s in siblings(p, photo_classes, paths)] for c, p in zip(classes, photo_classes)}
                noisy = noisy_label(rng, labels, n_context, share, near, cfg.sibling_noise)
                shown, bins = np.unique(noisy, return_inverse=True)
                if near is not None:                             # bin b shows the photo class of the b-th shown label
                    chosen[t] = (name, photo_classes[np.searchsorted(classes, shown)])
                all_bins.append(bins)
                shares.append(share)
            hidden = torch.tensor(np.stack(all_bins)).float()
            if cfg.skip_unlearnable and not all(learnable_label(s, len(np.unique(labels[:n_context]))) for s, labels in zip(shares, y)):
                continue
        if any(not any(can_show(collections[n], int(bins.max()) + 1) for n in names) for bins in all_bins):
            continue                                             # no collection has enough classes for this table
        picked, index = [], []
        for t, bins in enumerate(all_bins):
            if chosen[t] is not None:                            # label role with sibling noise: classes already drawn
                name, photos_classes = chosen[t]
            else:
                name = str(rng.choice([n for n in names if can_show(collections[n], int(bins.max()) + 1)]))
                photos_classes = pick_classes(rng, collections[name], int(bins.max()) + 1, cfg.fine_grained_share) if cfg.fine_grained_share > 0 else None
            shown_bins = bins
            if cfg.ordinal_confusion > 0 and role == "feature":
                shown_bins = confuse_bins(rng, bins, rng.uniform(0, cfg.ordinal_confusion))
            if cfg.distractor_share > 0 and rng.random() < cfg.distractor_share:
                shown_bins = distractor_bins(rng, bins)
            picked.append(name)
            index.append(pick_images(rng, collections[name], shown_bins, photos_classes, replace=not cfg.without_replacement))
        yield Batch(x[:, :, : int(n_columns.max())], y_context, y_test, n_columns, picked,
                    torch.tensor(np.stack(index)), hidden, torch.tensor(np.stack(all_bins)), role)


class TaskStream(IterableDataset):
    """task_batches for a DataLoader (batch_size=None); only class-level arrays travel to the workers, no pixels."""

    def __init__(self, cfg: TaskConfig, collections: dict, seed: int):
        self.cfg, self.seed = cfg, seed
        self.collections = {name: {k: c[k] for k in ("labels", "paths", "groups") if k in c} for name, c in collections.items()}

    def __iter__(self) -> Iterator[Batch]:
        worker = get_worker_info()
        return task_batches(self.cfg, self.collections, seed=self.seed + (worker.id if worker else 0))


# ---------- showing a batch to the model ----------

def with_column(x: Tensor, n_columns: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
    """Put a column back: values (batch, rows) become each table's next column after its real ones."""
    x = F.pad(x, (0, 1))
    x[torch.arange(len(x)), :, n_columns] = values
    return x, n_columns + 1


def augment(rng: np.random.Generator, photos: np.ndarray, min_crop: float = 0.85) -> np.ndarray:
    """uint8 photos (n, H, W, 3): each flipped left-right with probability 0.5 and cropped to a random square of
    min_crop-100% of its side at a random place, resized back. Test-row photos only, during training (augment_test)."""
    out = np.empty_like(photos)
    size = photos.shape[1]
    for i, photo in enumerate(photos):
        if rng.random() < 0.5:
            photo = photo[:, ::-1]
        side = int(round(size * rng.uniform(min_crop, 1.0)))
        top, left = rng.integers(0, size - side + 1, 2)
        crop = torch.from_numpy(np.ascontiguousarray(photo[top:top + side, left:left + side])).permute(2, 0, 1)[None].float()
        out[i] = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()
    return out


class Tasks:
    """The tasks as training and evaluation take them. A batch can be shown to the model in four ways:

        image             the table and each row's photo
        no image          the table alone: the floor
        column visible    the table with the hidden column given back, no photo: the ceiling
        images shuffled   the table and the photo of another row: a photo that says nothing about its row
    """
    VIEWS = ("image", "no image", "column visible", "images shuffled")

    def __init__(self, cfg: TaskConfig, collections: dict, train_names: list[str], validation_names: list[str],
                 to_tokens: Callable[[np.ndarray], Tensor], p_no_image: float = 0.15):
        """collections: name -> {"images": uint8 (photos, H, W, 3), "labels"}; to_tokens turns uint8 photos
        into the model's image input (inputs.py); p_no_image: the share of training steps without photos, which keeps
        the table-only path trained."""
        self.cfg, self.collections, self.to_tokens, self.p_no_image = cfg, collections, to_tokens, p_no_image
        self.train_names, self.validation_names = train_names, validation_names
        self.coin = torch.Generator().manual_seed(0)
        self.augment_rng = np.random.default_rng(0)

    def stream(self, seed: int) -> TaskStream:
        return TaskStream(self.cfg, {name: self.collections[name] for name in self.train_names}, seed)

    def validation_batches(self, seed: int) -> Iterator[Batch]:
        return task_batches(self.cfg, {name: self.collections[name] for name in self.validation_names}, seed)

    def training_view(self) -> str:
        return "no image" if torch.rand((), generator=self.coin) < self.p_no_image else "image"

    def images(self, batch: Batch, image_index: Tensor, augment_from: int | None = None) -> Tensor:
        """The photos of the given rows as model input; rows from augment_from on (the test rows) augmented."""
        pixels = np.concatenate([self.collections[name]["images"][rows] for name, rows in zip(batch.collection, image_index.numpy())])
        if augment_from is not None:
            pixels = pixels.reshape(image_index.shape + pixels.shape[1:])
            pixels[:, augment_from:] = augment(self.augment_rng, pixels[:, augment_from:].reshape((-1,) + pixels.shape[2:])).reshape(pixels[:, augment_from:].shape)
            pixels = pixels.reshape((-1,) + pixels.shape[2:])
        return self.to_tokens(pixels).unflatten(0, image_index.shape)

    def inputs(self, batch: Batch, device: str, view: str, training: bool = False) -> dict:
        """The model's keyword arguments for one batch, shown in the given way; `training` turns on augment_test."""
        assert view in self.VIEWS, view
        x = batch.x
        if view == "column visible":
            x, _ = with_column(x, batch.n_columns, batch.hidden)
        inputs = {"x": x.to(device), "y_context": batch.y_context.to(device)}
        if view == "image":
            augment_from = batch.y_context.shape[1] if training and self.cfg.augment_test else None
            inputs["images"] = self.images(batch, batch.image_index, augment_from).to(device)
        elif view == "images shuffled":
            other_row = torch.randperm(batch.image_index.shape[1], generator=torch.Generator().manual_seed(0))
            inputs["images"] = self.images(batch, batch.image_index[:, other_row]).to(device)
        return inputs
