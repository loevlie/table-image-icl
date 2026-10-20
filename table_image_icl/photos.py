"""The photos the generated tasks show: one labelled photo source, read from timm's webdataset shards on the Hub at
a pinned revision and decoded once into an array on disk. Two sources:

    imagenet1k    ImageNet-1k, the full training set: 1.28M photos of 1,000 classes in 1,024 shards, 181 GB decoded.
                  Read it into memory on a machine that has it (in_memory=True): random reads of a memmap on a network
                  file system make training wait on the disk.
    imagenet21k   ImageNet-21k (Winter 2021), the full set: 13.2M photos of 19,167 classes in 1,024 shards, 2 TB decoded.

For a laptop, decode a few shards of either (scripts/photos.py imagenet1k --shards 16: about 20,000 photos, 2 GB) and
train with the same --shards.

Classes with fewer than MIN_PHOTOS photos are never shown, and the rest are split into training and validation
classes, so validation tasks show photos of classes never trained on. A hierarchy.json next to names.json (WordNet
ancestors per class, from write_hierarchy) lets tasks draw classes that are near each other (tasks.pick_classes).

    scripts/photos.py imagenet1k --cache cache        # download and decode (needs the Hub terms accepted, hf auth login)
    train, validation = prepare("imagenet1k", "cache")  # what scripts/train.py calls"""
import io
import json
import os
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from PIL import Image

from table_image_icl.inputs import letterbox

MIN_PHOTOS = 10              # a class with fewer photos would repeat the same few pictures
VALIDATION_SHARE = 0.1       # share of the classes kept for validation tasks
IMAGE_SUFFIXES = (".webp", ".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class Source:
    repo: str                # Hugging Face dataset repo
    revision: str            # the commit the shards are read at, so a decode is reproducible
    shards: int              # how many of the repo's training shards, in the repo's own order


SOURCES = {
    "imagenet1k": Source("timm/imagenet-1k-wds", "cdf403ce12f01022a0c36e584e588c0b9cebc4af", 1024),
    "imagenet21k": Source("timm/imagenet-w21-webp-wds", "27e886b015251f2cbe809833445c307321b748ac", 1024),
}


# ---------- download ----------

def download(name: str, dest: str, shards: int | None = None) -> list[str]:
    """The source's first `shards` training shards (default: the source's number) into dest, at the pinned revision;
    files already there are not fetched again. Writes dest/MANIFEST.json and returns the shard paths in order."""
    from huggingface_hub import hf_hub_download
    source = SOURCES[name]
    shards = shards or source.shards
    os.makedirs(dest, exist_ok=True)
    info_path = hf_hub_download(source.repo, "_info.json", repo_type="dataset", revision=source.revision, local_dir=dest)
    names = json.load(open(info_path))["splits"]["train"]["filenames"][:shards]
    paths = [hf_hub_download(source.repo, f, repo_type="dataset", revision=source.revision, local_dir=dest) for f in names]
    manifest = {"source": name, "repo": source.repo, "revision": source.revision, "fetched": time.strftime("%Y-%m-%d %H:%M:%S"),
                "files": {f: os.path.getsize(p) for f, p in zip(names, paths)}}
    with open(os.path.join(dest, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return paths


# ---------- decode ----------

def read_shard(path: str):
    """Yields (key, photo bytes, class index, class name) per sample. A sample's members share a key: the class is its
    .cls member, the name (ImageNet's WordNet id) the class_name of its .json member when there is one."""
    def complete(key, sample):
        if "image" in sample and "label" in sample:
            return key, sample["image"], sample["label"], sample.get("name")

    with tarfile.open(path, "r") as tar:
        current, sample = None, {}
        for member in tar:
            if not member.isfile():
                continue
            key, suffix = os.path.splitext(member.name)
            if key != current:
                if current is not None and (done := complete(current, sample)):
                    yield done
                current, sample = key, {}
            if suffix.lower() in IMAGE_SUFFIXES:
                sample["image"] = tar.extractfile(member).read()
            elif suffix == ".cls":
                sample["label"] = int(tar.extractfile(member).read().decode().strip())
            elif suffix == ".json":
                meta = json.loads(tar.extractfile(member).read().decode())
                sample["name"] = meta.get("class_name") or meta.get("wnid")
        if current is not None and (done := complete(current, sample)):
            yield done


def count_samples(path: str) -> int:
    return sum(1 for _ in read_shard(path))


def _decode_into(args) -> tuple[np.ndarray, dict]:
    """One shard, written straight into its slice of the shared array; memory stays at one photo."""
    path, size, offset, images_path = args
    images = np.load(images_path, mmap_mode="r+")
    labels, names = [], {}
    for i, (_, photo, label, name) in enumerate(read_shard(path)):
        images[offset + i] = letterbox(Image.open(io.BytesIO(photo)), size)
        labels.append(label)
        if name:
            names[label] = name
    images.flush()
    return np.array(labels, dtype=np.int64), names


def decode(shards: list[str], out: str, size: int = 224, workers: int = 8, manifest: str | None = None) -> str:
    """The shards as out/images.npy (photos, size, size, 3) uint8, labels.npy, names.json and done.json. Shards are
    decoded in parallel, each into its own slice in shard order, so the result does not depend on the workers."""
    if os.path.exists(os.path.join(out, "done.json")):
        raise FileExistsError(f"{out} is already decoded; remove it to decode again")
    os.makedirs(out, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool:
        counts = list(pool.map(count_samples, shards))
    images_path = os.path.join(out, "images.npy")
    np.lib.format.open_memmap(images_path, mode="w+", dtype=np.uint8, shape=(sum(counts), size, size, 3)).flush()
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_decode_into, [(s, size, int(o), images_path) for s, o in zip(shards, offsets, strict=True)]))
    labels = np.concatenate([p[0] for p in parts])
    seen = {}
    for _, names in parts:
        seen.update(names)
    n_classes = int(labels.max()) + 1
    names = [seen.get(i, str(i)) for i in range(n_classes)]
    np.save(os.path.join(out, "labels.npy"), labels)
    with open(os.path.join(out, "names.json"), "w") as f:
        json.dump(names, f)
    record = {"shards": [os.path.basename(s) for s in shards], "photos": int(len(labels)), "size": size, "classes": n_classes}
    if manifest and os.path.exists(manifest):
        record["source"] = {k: v for k, v in json.load(open(manifest)).items() if k != "files"}
    with open(os.path.join(out, "done.json"), "w") as f:
        json.dump(record, f, indent=1)
    return out


# ---------- load ----------

def read_into_memory(images: np.ndarray, chunk: int = 10_000) -> np.ndarray:
    out = np.empty(images.shape, images.dtype)
    for i in range(0, len(images), chunk):
        out[i:i + chunk] = images[i:i + chunk]
    return out


def load_photos(folder: str, in_memory: bool = False, max_photos: int = 0) -> dict:
    """A decoded folder: {"images" (memmap, or an array in RAM), "labels", "names", "record", "hierarchy"}; the
    first max_photos photos when capped (0 = all). The hierarchy is {} when the folder has no hierarchy.json."""
    images = np.load(os.path.join(folder, "images.npy"), mmap_mode="r")
    labels = np.load(os.path.join(folder, "labels.npy"))
    if max_photos:
        images, labels = images[:max_photos], labels[:max_photos]
    if in_memory:
        images = read_into_memory(images)
    names = np.array(json.load(open(os.path.join(folder, "names.json"))), dtype=object)
    hierarchy = json.load(open(os.path.join(folder, "hierarchy.json"))) if os.path.exists(os.path.join(folder, "hierarchy.json")) else {}
    return {"images": images, "labels": labels, "names": names, "record": json.load(open(os.path.join(folder, "done.json"))), "hierarchy": hierarchy}


# ---------- the WordNet hierarchy ----------

def wordnet_paths(wnids: list[str]) -> dict[str, list[str]]:
    """wnid -> its ancestors in WordNet, root first, itself excluded (the shortest hypernym path); [] for a name that
    is not a wnid or not in WordNet. Needs nltk and its WordNet corpus (nltk.download("wordnet")), once."""
    from nltk.corpus import wordnet
    out = {}
    for wnid in wnids:
        try:
            synset = wordnet.synset_from_pos_and_offset("n", int(wnid[1:])) if wnid.startswith("n") and wnid[1:].isdigit() else None
            path = min(synset.hypernym_paths(), key=len) if synset else []
            out[wnid] = [f"n{s.offset():08d}" for s in path[:-1]]
        except Exception:
            out[wnid] = []
    return out


def write_hierarchy(folder: str) -> str:
    """hierarchy.json for a decoded folder, from its names.json."""
    names = json.load(open(os.path.join(folder, "names.json")))
    path = os.path.join(folder, "hierarchy.json")
    with open(path, "w") as f:
        json.dump(wordnet_paths(names), f)
    return path


def class_paths(labels: np.ndarray, names: np.ndarray, hierarchy: dict) -> dict[int, tuple]:
    """class index -> its ancestor path, for the classes shown (label >= 0) that have a path."""
    out = {}
    for c in np.unique(labels[labels >= 0]):
        path = hierarchy.get(str(names[c]), [])
        if path:
            out[int(c)] = tuple(path)
    return out


def class_groups(paths: dict, depths=range(2, 11)) -> dict[int, dict[str, list[int]]]:
    """depth -> ancestor at that depth -> the classes under it (groups of one class dropped)."""
    out = {}
    for depth in depths:
        groups = {}
        for c, path in paths.items():
            if len(path) > depth:
                groups.setdefault(path[depth], []).append(c)
        groups = {g: members for g, members in groups.items() if len(members) > 1}
        if groups:
            out[depth] = groups
    return out


def split_classes(photos: dict, validation_share: float = VALIDATION_SHARE, min_photos: int = MIN_PHOTOS, seed: int = 0) -> tuple[dict, dict]:
    """Two collections over the same photos: training classes and validation classes, disjoint. A collection's labels
    are -1 for every photo outside its classes, and classes with fewer than min_photos photos are -1 in both."""
    labels = photos["labels"]
    classes, counts = np.unique(labels, return_counts=True)
    usable = classes[counts >= min_photos]
    shuffled = np.random.default_rng(seed).permutation(usable)
    n_validation = max(1, int(round(validation_share * len(usable)))) if len(usable) > 1 else 0
    validation_classes = set(shuffled[:n_validation].tolist())

    def masked(keep):
        kept = np.array([c in keep for c in classes])
        allowed = np.isin(labels, classes[kept])
        split = {**photos, "labels": np.where(allowed, labels, -1)}
        split["paths"] = class_paths(split["labels"], photos["names"], photos.get("hierarchy", {}))
        split["groups"] = class_groups(split["paths"])
        return split

    return masked(set(usable.tolist()) - validation_classes), masked(validation_classes)


def prepare(name: str, cache: str, shards: int | None = None, in_memory: bool = False, max_photos: int = 0) -> tuple[dict, dict]:
    """The training and validation collections of a decoded source under cache/<name>-<shards>/ (scripts/photos.py
    makes it)."""
    folder = decoded_folder(name, cache, shards)
    if not os.path.exists(os.path.join(folder, "done.json")):
        raise FileNotFoundError(f"{folder} is not decoded yet: python scripts/photos.py {name} --cache {cache}" + (f" --shards {shards}" if shards else ""))
    return split_classes(load_photos(folder, in_memory, max_photos))


def decoded_folder(name: str, cache: str, shards: int | None = None) -> str:
    return os.path.join(cache, f"{name}-{shards or SOURCES[name].shards}")
