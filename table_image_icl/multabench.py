"""MulTaBench's image-table datasets (Kaggle, owner chico89) and its train/test splits. The split and the row caps
follow get_split of https://github.com/alanarazi7/MulTaBench, so scores are on the benchmark's rows."""
import json
import os
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, train_test_split

MAX_TRAIN, MAX_TEST = 10_000, 5_000

# MulTaBench has 20 image-table datasets ("core") and released 40 more alongside them ("extended", slugs multabench-full-*)
# that pass its joint-signal filter but not its encoder-tuning filter, and whose scores it keeps apart from the core 40.
# The model's design is chosen on four extended datasets, so that every core dataset can be a test set. Three core
# datasets were scored during development before the validation set moved to the extended slice; they are kept apart
# from the eight test sets, which are untouched. Nothing is trained on any of them.
VALIDATION_SETS = {"OASIS Alzheimer": "multabench-full-oasis-alzheimers", "HAM10000": "multabench-full-ham10000-lesion",
                   "PAD-UFES-20": "multabench-full-pad-ufes-lesion", "SoCal Houses": "multabench-full-socal-houses"}
DEVELOPMENT_SETS = {"CBIS-DDSM": "multabench-cbis-ddsm", "CheXpert": "multabench-chexpert", "Mango Mass": "multabench-mango-mass"}
TEST_SETS = {"Khaadi Clothes": "multabench-khaadi-clothes", "Amazon Packages": "multabench-amazon-packages",
             "Glaucoma SMDG": "multabench-glaucoma-smdg", "Mammography CMMD": "multabench-mammography-cmmd",
             "PetFinder": "multabench-petfinder", "Hateful Meme": "multabench-hateful-meme",
             "Flower Bouquets": "multabench-flower-bouquets", "Painting Price": "multabench-painting-price"}
DATASET_SETS = {"validation": VALIDATION_SETS, "development": DEVELOPMENT_SETS, "test": TEST_SETS}
ALL_SETS = {**VALIDATION_SETS, **DEVELOPMENT_SETS, **TEST_SETS}


class Dataset(NamedTuple):
    table: np.ndarray            # (rows, columns) floats; categories as codes; NaN = missing
    columns: list[str]
    y: np.ndarray                # class codes (int64) or regression targets (float64)
    paths: np.ndarray            # one photo path per row
    is_classification: bool
    version: str                 # the Kaggle dataset version the files came from


def load(slug: str) -> Dataset:
    """The table without its target and image columns; free-text columns dropped, categories as codes.
    `slug` is a chico89 Kaggle slug, or a local folder in the same layout (data.csv, metadata.json, images)."""
    if os.path.isdir(slug):
        folder = os.path.abspath(slug)
    else:
        import kagglehub
        folder = kagglehub.dataset_download(f"chico89/{slug}")
    with open(os.path.join(folder, "metadata.json")) as f:
        meta = json.load(f)
    df = pd.read_csv(os.path.join(folder, "data.csv"))
    is_classification = meta.get("task_type", "cls") != "reg"
    target = df[meta["target"]]
    y = target.astype("category").cat.codes.to_numpy(np.int64) if is_classification else target.to_numpy(np.float64)
    paths = np.array([os.path.join(folder, str(p)) for p in df[meta["image_col"]]])

    table = df.drop(columns=[meta["target"], meta["image_col"]])
    for name in list(table.columns):
        column = table[name]
        if pd.api.types.is_numeric_dtype(column):
            continue
        if column.nunique() > 50 and column.astype(str).str.len().mean() > 20:      # free text
            table = table.drop(columns=[name])
        else:
            table[name] = column.astype("category").cat.codes.replace(-1, np.nan)
    return Dataset(table.to_numpy(np.float64), list(table.columns), y, paths, is_classification, os.path.basename(folder))


def split(y: np.ndarray, is_classification: bool, k: int) -> tuple[np.ndarray, np.ndarray]:
    """MulTaBench's get_split: split k of a repeated 3-fold (2 repeats, seed 42), stratified when every class has
    3+ rows; train and test capped at 10,000 and 5,000 rows, keeping the class balance."""
    stratified = is_classification and np.unique(y, return_counts=True)[1].min() >= 3
    folds = (RepeatedStratifiedKFold if stratified else RepeatedKFold)(n_splits=3, n_repeats=2, random_state=42)
    train, test = list(folds.split(np.zeros(len(y)), y))[k]

    def cap(rows, size, seed):
        if len(rows) <= size:
            return rows
        keep_balance = is_classification and np.unique(y[rows], return_counts=True)[1].min() >= 2
        return train_test_split(rows, train_size=size, random_state=seed, stratify=y[rows] if keep_balance else None)[0]

    return np.sort(cap(train, MAX_TRAIN, 42 + k)), np.sort(cap(test, MAX_TEST, 42 + k))
