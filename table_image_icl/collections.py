"""Labelled photo collections from Hugging Face, kept in memory as 224 px uint8 images. A training task borrows
one collection and shows each row a photo of the class its hidden value falls in; the validation collections are
never trained on."""
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from table_image_icl.inputs import letterbox

COLLECTIONS = {  # name: (Hugging Face id, config, kind). kind "class": one label column; "attr": boolean attributes
    "mini-imagenet": ("timm/mini-imagenet", None, "class"),
    "food101": ("ethz/food101", None, "class"),
    "cifar100": ("uoft-cs/cifar100", None, "class"),
    "cars": ("tanganke/stanford_cars", None, "class"),
    "flowers": ("dpdl-benchmark/oxford_flowers102", None, "class"),
    "pets": ("timm/oxford-iiit-pet", None, "class"),
    "gtsrb": ("tanganke/gtsrb", None, "class"),
    "eurosat": ("blanchon/EuroSAT_RGB", None, "class"),
    "celeba": ("flwrlabs/celeba", "img_align+identity+attr", "attr"),
    "dtd": ("tanganke/dtd", None, "class"),
    "resisc45": ("timm/resisc45", None, "class"),
}
TRAIN_NAMES = ["celeba", "cars", "flowers", "gtsrb", "food101", "mini-imagenet", "cifar100", "eurosat"]
VALIDATION_NAMES = ["dtd", "resisc45", "pets", "celeba-heldout"]     # celeba-heldout: CelebA's photos with attributes training never uses
N_HELDOUT_ATTRIBUTES = 10


def load_all(max_images: int = 4000, cache_dir: str | None = None) -> dict:
    """Every collection, with CelebA split into training attributes and held-out attributes. With a cache_dir, each
    collection's decoded photos are saved there once and read back afterwards."""
    collections = {name: load_collection(name, max_images, cache_dir) for name in COLLECTIONS}
    collections["celeba"], collections["celeba-heldout"] = split_attributes(collections["celeba"], N_HELDOUT_ATTRIBUTES)
    return collections


def split_attributes(collection: dict, n_heldout: int) -> tuple[dict, dict]:
    """An attribute collection as two: the first attributes for training, the last n_heldout for validation."""
    labels = collection["labels"]
    train = {**collection, "labels": labels[:, :-n_heldout]}
    heldout = {**collection, "labels": labels[:, -n_heldout:]}
    if "names" in collection:
        train["names"], heldout["names"] = collection["names"][:-n_heldout], collection["names"][-n_heldout:]
    return train, heldout


def load_collection(name: str, max_images: int = 4000, cache_dir: str | None = None, threads: int = 16) -> dict:
    path = cache_dir and os.path.join(cache_dir, f"{name}-{max_images}.npz")
    return cached(path, lambda: decode_collection(name, max_images, threads))


def cached(path: str | None, build) -> dict:
    """The dict of arrays build() returns, saved at path the first time and read from there after. The collection
    carries the file's sha256, so a run records exactly which photos it saw."""
    if not (path and os.path.exists(path)):
        collection = build()
        if not path:
            return collection
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path + ".tmp.npz", **collection)
        os.replace(path + ".tmp.npz", path)
    with np.load(path) as saved:
        collection = {"images": saved["images"], "labels": saved["labels"], "kind": str(saved["kind"]), "sha256": sha256(path)}
        if "names" in saved:
            collection["names"] = saved["names"].astype(object)
        return collection


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_collection(name: str, max_images: int, threads: int) -> dict:
    import datasets
    hf_id, config, kind = COLLECTIONS[name]
    ds = datasets.load_dataset(hf_id, config, split="train").shuffle(seed=0)
    ds = ds.select(range(min(len(ds), max_images)))
    image_col = next(k for k, v in ds.features.items() if isinstance(v, datasets.Image))
    if kind == "class":
        label_col = next((k for k, v in ds.features.items() if isinstance(v, datasets.ClassLabel)), None)
        label_col = label_col or next(k for k in ["label", "labels", "fine_label", "class"] if k in ds.features)
        original, labels = np.unique(np.asarray(ds[label_col]), return_inverse=True)
        feature = ds.features[label_col]
        names = [feature.names[int(i)] if isinstance(feature, datasets.ClassLabel) else str(i) for i in original]
    else:
        names = [k for k, v in ds.features.items() if getattr(v, "dtype", "") == "bool"]
        labels = np.stack([np.asarray(ds[a], dtype=bool) for a in names], 1)
    with ThreadPoolExecutor(threads) as pool:
        images = np.stack(list(pool.map(letterbox, ds[image_col])))
    return {"images": images, "labels": labels, "kind": kind, "names": np.array(names, dtype=object)}
