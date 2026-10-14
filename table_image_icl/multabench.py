"""MulTaBench's image-table datasets (Kaggle, owner chico89) and its train/test splits. The split and the row caps
follow get_split of https://github.com/alanarazi7/MulTaBench, so scores are on the benchmark's rows."""
import json
import os
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, StratifiedGroupKFold, train_test_split

MAX_TRAIN, MAX_TEST = 10_000, 5_000

# MulTaBench has 20 image-table datasets ("core") and released 40 more alongside them ("extended", slugs multabench-full-*)
# that pass its joint-signal filter but not its encoder-tuning filter, and whose scores it keeps apart from the core 40.
# The model's design is chosen on four extended datasets, so that every core dataset can be a test set. Three core
# datasets were scored during development before the validation set moved to the extended slice; they are kept apart
# from the eight test sets, which are untouched. Nothing is trained on any of them.
VALIDATION_SETS = {"HAM10000": "multabench-full-ham10000-lesion", "PAD-UFES-20": "multabench-full-pad-ufes-lesion",
                   "SoCal Houses": "multabench-full-socal-houses", "DVM-Car": "multabench-full-dvm-car"}
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
    groups: np.ndarray | None = None   # one group id per row (patient, listing, fruit) when the rows are grouped


def load(slug: str, grouped: bool = False) -> Dataset:
    """The table without its target and image columns; free-text columns dropped, categories as codes.
    `slug` is a chico89 Kaggle slug, or a local folder in the same layout (data.csv, metadata.json, images).
    With `grouped`, rows from the same patient, listing or fruit (GROUP_KEYS) get a group id, so that `split` keeps
    them on one side; without it the benchmark's own row-wise split is used."""
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
    groups = group_ids(slug, df, meta["image_col"]) if grouped else None
    return Dataset(table.to_numpy(np.float64), list(table.columns), y, paths, is_classification, os.path.basename(folder), groups)


def _name(df: pd.DataFrame, image_col: str) -> pd.Series:
    return df[image_col].astype(str).str.rsplit("/", n=1).str[-1]


def _patient_from_cbis(df: pd.DataFrame, image_col: str) -> pd.Series:
    """CBIS-DDSM names its files by DICOM series UID; the patient of each file comes from the collection's own
    description tables, listed in data/cbis_ddsm_patients.csv."""
    key = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "cbis_ddsm_patients.csv")).set_index("image")["patient"]
    return key.reindex(df[image_col]).reset_index(drop=True)


# Rows of the same patient, listing or fruit, read off the benchmark's files. MulTaBench splits rows i.i.d. and its
# README says the splits are not audited for group structure, so a patient's other lesion or a listing's duplicate can
# sit in the context of a test row. HAM10000 needs no key (the benchmark kept one image per lesion), nor does DVM-Car
# (one photo per advert).
GROUP_KEYS = {
    "multabench-full-pad-ufes-lesion": lambda df, c: _name(df, c).str.extract(r"(PAT_\d+)_")[0],                 # patient
    "multabench-full-socal-houses": lambda df, c: df["street"].str.lower().str.strip() + " | " + df["citi"].str.lower().str.strip(),   # listing
    "multabench-cbis-ddsm": _patient_from_cbis,                                                                   # patient
    "multabench-chexpert": lambda df, c: _name(df, c).str.extract(r"(patient\d+)")[0],                             # patient
    "multabench-mango-mass": lambda df, c: _name(df, c).str.extract(r"^(\d+)")[0],                                 # fruit
    "multabench-mammography-cmmd": lambda df, c: _name(df, c).str.extract(r"(D\d-\d+)")[0],                        # patient
    "multabench-petfinder": lambda df, c: _name(df, c).str.split("-").str[0],                                     # pet listing
    "multabench-khaadi-clothes": lambda df, c: _name(df, c).str.split("_image").str[0],                           # product
}


def group_ids(slug: str, df: pd.DataFrame, image_col: str) -> np.ndarray | None:
    """One group id per row for datasets with a known group key, else None (every row its own group)."""
    parts = os.path.normpath(slug).split(os.sep)                 # a slug, or a path that contains one
    key = next((GROUP_KEYS[part] for part in reversed(parts) if part in GROUP_KEYS), None)
    if key is None:
        return None
    groups = key(df, image_col)
    assert groups.notna().all(), f"{slug}: {groups.isna().sum()} rows without a group id"
    return groups.to_numpy()


def split(y: np.ndarray, is_classification: bool, k: int, groups: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """MulTaBench's get_split: split k of a repeated 3-fold (2 repeats, seed 42), stratified when every class has
    3+ rows; train and test capped at 10,000 and 5,000 rows, keeping the class balance. With `groups`, a stratified
    group 3-fold instead (one repeat per seed), so a group's rows all land on one side."""
    stratified = is_classification and np.unique(y, return_counts=True)[1].min() >= 3
    if groups is not None:
        folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42 + k // 3)
        train, test = list(folds.split(np.zeros(len(y)), y if is_classification else np.zeros(len(y), int), groups))[k % 3]
    else:
        folds = (RepeatedStratifiedKFold if stratified else RepeatedKFold)(n_splits=3, n_repeats=2, random_state=42)
        train, test = list(folds.split(np.zeros(len(y)), y))[k]

    def cap(rows, size, seed):
        if len(rows) <= size:
            return rows
        keep_balance = is_classification and np.unique(y[rows], return_counts=True)[1].min() >= 2
        return train_test_split(rows, train_size=size, random_state=seed, stratify=y[rows] if keep_balance else None)[0]

    return np.sort(cap(train, MAX_TRAIN, 42 + k)), np.sort(cap(test, MAX_TEST, 42 + k))
