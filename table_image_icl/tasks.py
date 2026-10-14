"""Training tasks: synthetic tables from TabICL v2's prior, where every row also gets a photo. The photo stands for
a hidden variable, in one of two roles drawn per batch:

    photo as feature   the column a random forest leans on most is removed from the table and cut into bins, and
                       the photo shows the bin;
    photo as label     the table stays whole and the photo shows the row's class for a share of the rows
                       (30-100%, drawn per task) and a random class for the rest.

Each bin (or shown class) is given one class of a photo collection, or the two values of a face attribute, and each
row gets a photo of its bin's class. A batch carries photo indices, not pixels; the pixels are looked up when the
batch is shown to the model."""
import random
from dataclasses import dataclass
from typing import Callable, Iterator, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from table_image_icl.predict import bin_targets
from table_image_icl.tables import N_SLOTS


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

def most_important_column(x: np.ndarray, y: np.ndarray, n_context: int) -> int:
    """The column a random forest fitted on the context rows relies on most."""
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=50, random_state=0).fit(x[:n_context], y[:n_context])
    return int(forest.feature_importances_.argmax())


def hide_column(x: Tensor, n_columns: Tensor, which: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Remove column `which` (batch,) from every table: the tables one column narrower, the removed values
    (batch, rows), and the new numbers of real columns."""
    batch, rows, columns = x.shape
    keep = torch.arange(columns).expand(batch, -1) != which[:, None]
    kept = x.transpose(1, 2)[keep].reshape(batch, columns - 1, rows).transpose(1, 2)
    return kept, x[torch.arange(batch), :, which], n_columns - 1


def noisy_label(rng: np.random.Generator, y: np.ndarray, n_context: int) -> np.ndarray:
    """What the photo shows when it stands for the label: the row's class for a share of the rows, a random class of
    the task for the others. The share is drawn per task from 0.3-1."""
    share = rng.uniform(0.3, 1.0)
    classes = np.unique(y[:n_context])
    return np.where(rng.random(len(y)) < share, y, rng.choice(classes, len(y)))


def values_to_bins(values: np.ndarray, n_context: int, n_bins: int) -> np.ndarray:
    """A column cut into at most n_bins bins at the quantiles of its context rows."""
    return bin_targets(values, n_context, n_bins)[0]


def pick_images(rng: np.random.Generator, collection: dict, bins: np.ndarray) -> np.ndarray:
    """One photo per row. Class collection: every bin gets its own random class and rows draw photos of it.
    Attribute collection (two bins only): one random attribute, false for one bin and true for the other."""
    labels = collection["labels"]
    if collection["kind"] == "attr":
        membership = labels[:, rng.integers(labels.shape[1])].astype(int)
        if rng.random() < 0.5:
            membership = 1 - membership
    else:
        membership = np.full(len(labels), -1)
        classes = rng.permutation(np.unique(labels))[: bins.max() + 1]
        for b, c in enumerate(classes):
            membership[labels == c] = b
    index = np.empty(len(bins), dtype=np.int64)
    for b in np.unique(bins):
        index[bins == b] = rng.choice(np.flatnonzero(membership == b), size=int((bins == b).sum()))
    return index


def can_show(collection: dict, n_bins: int) -> bool:
    if collection["kind"] == "attr":
        return n_bins == 2
    return len(np.unique(collection["labels"])) >= n_bins


def task_batches(cfg: TaskConfig, collections: dict, seed: int = 0) -> Iterator[Batch]:
    """collections: name -> {"labels": class ids (photos,) or attributes (photos, attributes), "kind": "class"|"attr"}."""
    rng = np.random.default_rng(seed)
    names = sorted(collections)
    for x, y_context, y_test, n_columns in prior_tables(cfg, seed):
        role = "label" if rng.random() < cfg.image_as_label_share else "feature"
        if role == "feature" and (n_columns < 2).any():
            continue                                             # a table needs a column to hide and one to keep
        n_context = y_context.shape[1]
        y = torch.cat([y_context, y_test], 1).numpy()
        if role == "feature":
            which = [most_important_column(x[t, :, : int(n_columns[t])].numpy(), y[t], n_context) for t in range(len(x))]
            x, hidden, n_columns = hide_column(x, n_columns, torch.tensor(which))
            all_bins = [values_to_bins(values, n_context, int(rng.integers(2, cfg.max_bins + 1))) for values in hidden.numpy()]
        else:
            all_bins = [np.unique(noisy_label(rng, labels, n_context), return_inverse=True)[1] for labels in y]
            hidden = torch.tensor(np.stack(all_bins)).float()
        picked, index = [], []
        for bins in all_bins:
            name = str(rng.choice([n for n in names if can_show(collections[n], int(bins.max()) + 1)]))
            picked.append(name)
            index.append(pick_images(rng, collections[name], bins))
        yield Batch(x[:, :, : int(n_columns.max())], y_context, y_test, n_columns, picked,
                    torch.tensor(np.stack(index)), hidden, torch.tensor(np.stack(all_bins)), role)


class TaskStream(IterableDataset):
    """task_batches for a DataLoader (batch_size=None); only label arrays travel to the workers, no pixels."""

    def __init__(self, cfg: TaskConfig, collections: dict, seed: int):
        self.cfg, self.seed = cfg, seed
        self.collections = {name: {"labels": c["labels"], "kind": c["kind"]} for name, c in collections.items()}

    def __iter__(self) -> Iterator[Batch]:
        worker = get_worker_info()
        return task_batches(self.cfg, self.collections, seed=self.seed + (worker.id if worker else 0))


# ---------- showing a batch to the model ----------

def with_column(x: Tensor, n_columns: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
    """Put a column back: values (batch, rows) become each table's next column after its real ones."""
    x = F.pad(x, (0, 1))
    x[torch.arange(len(x)), :, n_columns] = values
    return x, n_columns + 1


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
        """collections: name -> {"images": uint8 (photos, H, W, 3), "labels", "kind"}; to_tokens turns uint8 photos
        into the model's image input (inputs.py); p_no_image: the share of training steps without photos, which keeps
        the table-only path trained."""
        self.cfg, self.collections, self.to_tokens, self.p_no_image = cfg, collections, to_tokens, p_no_image
        self.train_names, self.validation_names = train_names, validation_names
        self.coin = torch.Generator().manual_seed(0)

    def stream(self, seed: int) -> TaskStream:
        return TaskStream(self.cfg, {name: self.collections[name] for name in self.train_names}, seed)

    def validation_batches(self, seed: int) -> Iterator[Batch]:
        return task_batches(self.cfg, {name: self.collections[name] for name in self.validation_names}, seed)

    def training_view(self) -> str:
        return "no image" if torch.rand((), generator=self.coin) < self.p_no_image else "image"

    def images(self, batch: Batch, image_index: Tensor) -> Tensor:
        pixels = np.concatenate([self.collections[name]["images"][rows] for name, rows in zip(batch.collection, image_index.numpy())])
        return self.to_tokens(pixels).unflatten(0, image_index.shape)

    def inputs(self, batch: Batch, device: str, view: str) -> dict:
        """The model's keyword arguments for one batch, shown in the given way."""
        assert view in self.VIEWS, view
        x = batch.x
        if view == "column visible":
            x, _ = with_column(x, batch.n_columns, batch.hidden)
        inputs = {"x": x.to(device), "y_context": batch.y_context.to(device)}
        if view == "image":
            inputs["images"] = self.images(batch, batch.image_index).to(device)
        elif view == "images shuffled":
            other_row = torch.randperm(batch.image_index.shape[1], generator=torch.Generator().manual_seed(0))
            inputs["images"] = self.images(batch, batch.image_index[:, other_row]).to(device)
        return inputs
