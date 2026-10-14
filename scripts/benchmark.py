"""The model on real image-table data: MulTaBench datasets, MulTaBench's splits, no training on them.

    python scripts/benchmark.py --run runs/block4 --out results/block4/multabench.json [--set test] [--max-context 4096]

On each dataset and split the model is scored three ways on the same rows: table only, table + photo, and table +
the photo of another test row. Against: TabPFN v2 on the table, and MulTaBench's frozen baseline, TabPFN v2 on the
table plus 30 PCA components of frozen DINOv2 features. The context is at most --max-context rows, and the model
reads labels from at most --max-sources context photos; a context sweep raises both."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from table_image_icl import multabench
from table_image_icl.inputs import inputs_for, letterbox
from table_image_icl.model import load_run
from table_image_icl.predict import predict_classes, predict_values
from table_image_icl.references import FrozenDinoFeatures, auroc, pca_columns, r2, tabpfn_v2
from table_image_icl.training import provenance

DATASETS = {**multabench.VALIDATION_SETS, **multabench.TEST_SETS}


def read_photos(paths, threads=16) -> np.ndarray:
    with ThreadPoolExecutor(threads) as pool:
        return np.stack(list(pool.map(letterbox, paths)))


def predictions(model, to_tokens, dino, data, context, test, device) -> dict:
    """Every method's prediction for one split: (probabilities, classes) for classification, values for regression."""
    x, y = data.table, data.y
    photos = read_photos(data.paths[context]), read_photos(data.paths[test])
    shuffled = photos[0], photos[1][np.random.default_rng(0).permutation(len(test))]
    ours = predict_classes if data.is_classification else predict_values
    out = {"ours, table only": ours(model, x[context], y[context], x[test], device=device, chunk=512),
           "ours, table + image": ours(model, x[context], y[context], x[test], device=device, chunk=512, photos=photos, to_tokens=to_tokens),
           "ours, image of another row": ours(model, x[context], y[context], x[test], device=device, chunk=512, photos=shuffled, to_tokens=to_tokens)}
    features = dino(np.concatenate(photos))
    plain = np.vstack([x[context], x[test]])
    frozen = np.hstack([plain, pca_columns(features, len(context))])
    for name, table in [("TabPFN v2, table only", plain), ("TabPFN v2, table + frozen DINOv2 PCA", frozen)]:
        reference = tabpfn_v2(data.is_classification, device).fit(table[: len(context)], y[context])
        out[name] = (reference.predict_proba(table[len(context):]), reference.classes_) if data.is_classification else reference.predict(table[len(context):])
    return out


def mean_and_se(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan"), "n": len(values)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", choices=list(multabench.DATASET_SETS), default="validation")
    ap.add_argument("--datasets", nargs="+", help="specific datasets (benchmark names or local folders) instead of a whole set")
    ap.add_argument("--max-context", type=int, default=256)
    ap.add_argument("--max-sources", type=int, help="context photos the model reads labels from (default: as trained)")
    ap.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, record = load_run(args.run, device)
    if args.max_sources is not None:
        model.read.max_sources = args.max_sources
    to_tokens = inputs_for(record["arm"], device, block=record["dino_block"])
    dino = FrozenDinoFeatures(device)
    report = {"run": args.run, "arm": record["arm"], "max_context": args.max_context, "max_sources": model.read.max_sources,
              "set": args.set, **provenance(), "datasets": {}}
    for name in args.datasets or list(multabench.DATASET_SETS[args.set]):
        data = multabench.load(DATASETS.get(name, name))
        name = name if name in DATASETS else os.path.basename(os.path.normpath(name))
        scores = {}
        for k in args.splits:
            train, test = multabench.split(data.y, data.is_classification, k)
            context = train
            if len(train) > args.max_context:
                stratify = data.y[train] if data.is_classification else None
                context = train_test_split(train, train_size=args.max_context, random_state=k, stratify=stratify)[0]
            if data.is_classification:
                test = test[np.isin(data.y[test], data.y[context])]
            for method, result in predictions(model, to_tokens, dino, data, context, test, device).items():
                score = auroc(data.y[test], *result) if data.is_classification else r2(data.y[test], result)
                scores.setdefault(method, []).append(score)
            print(f"{name} split {k}: " + " | ".join(f"{m} {s[-1]:.3f}" for m, s in scores.items()), flush=True)
        report["datasets"][name] = {"metric": "AUROC" if data.is_classification else "R2", "kaggle_version": data.version,
                                    "context_rows": int(len(context)), "test_rows": int(len(test)), **{m: mean_and_se(s) for m, s in scores.items()}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
