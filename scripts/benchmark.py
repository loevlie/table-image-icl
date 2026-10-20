"""The model on real image-table data: MulTaBench datasets, MulTaBench's splits, no training on them.

    python scripts/benchmark.py --run checkpoints/imagenet21k --out results/imagenet21k/multabench.json [--set validation|no-harm|test] [--grouped]

On each dataset and split every method predicts the same test rows from the same context rows (table_image_icl/
references.py): ours with the photo, without it, and with two useless-photo controls; TabPFN v2 and TabICL v2 on the
table and on the table plus MulTaBench's frozen DINOv2 PCA columns. The report holds AUROC (classification) or RMSE
(regression) per method, mean and standard error over the splits, and every method's per-row predictions go to a
file next to it, for scripts/bootstrap.py. The context is at most --max-context rows; the model reads labels from at
most --max-sources context photos."""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from table_image_icl import multabench
from table_image_icl.inputs import inputs_for
from table_image_icl.model import load_run
from table_image_icl.references import FrozenDinoFeatures, predictions
from table_image_icl.scoring import save_predictions, score
from table_image_icl.training import provenance


def mean_and_se(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan"), "n": len(values)}


def predictions_path(out: str) -> str:
    """results/x/multabench.json -> results/x/multabench-predictions.npz"""
    return os.path.splitext(out)[0] + "-predictions.npz"


def context_and_test(data, k: int, max_context: int) -> tuple[np.ndarray, np.ndarray]:
    """Split k of the benchmark: at most max_context context rows (stratified for classification); test rows whose
    class the context never shows are dropped."""
    train, test = multabench.split(data.y, data.is_classification, k, groups=data.groups)
    context = train
    if len(train) > max_context:
        context = train_test_split(train, train_size=max_context, random_state=k, stratify=data.y[train] if data.is_classification else None)[0]
    if data.is_classification:
        test = test[np.isin(data.y[test], data.y[context])]
    return context, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a run folder, or checkpoints/<name> from table_image_icl.pretrained")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", choices=list(multabench.DATASET_SETS), default="validation")
    ap.add_argument("--datasets", nargs="+", help="specific datasets (benchmark names, Kaggle slugs or local folders) instead of a whole set")
    ap.add_argument("--max-context", type=int, default=256)
    ap.add_argument("--max-sources", type=int, help="context photos the model reads labels from (default: as trained)")
    ap.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--grouped", action="store_true", help="keep each patient, listing or fruit on one side of the split (multabench.GROUP_KEYS) instead of the benchmark's row-wise split")
    ap.add_argument("--no-references", action="store_true", help="only our model (no DINOv2 features, TabPFN v2 or TabICL v2): quick")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, record = load_run(args.run, device)
    if args.max_sources is not None:
        model.read.max_sources = args.max_sources
    to_tokens = inputs_for(model.cfg.encoder, device)
    dino = None if args.no_references else FrozenDinoFeatures(device)
    report = {"run": args.run, "encoder": model.cfg.encoder, "max_context": args.max_context, "max_sources": model.read.max_sources,
              "set": args.set, "grouped": args.grouped, **provenance(), "predictions": predictions_path(args.out), "datasets": {}}
    kept = {}
    for name in args.datasets or list(multabench.DATASET_SETS[args.set]):
        data = multabench.load(multabench.ALL_SETS.get(name, name), grouped=args.grouped)
        name = name if name in multabench.ALL_SETS else os.path.basename(os.path.normpath(name))
        scores = {}
        for k in args.splits:
            context, test = context_and_test(data, k, args.max_context)
            results = predictions(model, to_tokens, dino, data, context, test, device)
            kept[(name, k)] = {"y": data.y[test], "rows": test, "methods": results}
            for method, result in results.items():
                scores.setdefault(method, []).append(score(data.y[test], result, data.is_classification))
            print(f"{name} split {k}: " + " | ".join(f"{m} {s[-1]:.3f}" for m, s in scores.items()), flush=True)
        report["datasets"][name] = {"metric": "AUROC" if data.is_classification else "RMSE", "kaggle_version": data.version,
                                    "context_rows": int(len(context)), "test_rows": int(len(test)), **{m: mean_and_se(s) for m, s in scores.items()}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_predictions(report["predictions"], kept)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("saved", args.out, "and", report["predictions"])


if __name__ == "__main__":
    main()
