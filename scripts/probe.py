"""Probe a published model on one benchmark dataset: a few labelled rows as context, the rest predicted, scored
against the frozen-feature reference with paired bootstrap intervals; optionally one row's prediction in detail.

    python scripts/probe.py --dataset "Mango Mass"                        # the default model, 256 context rows, 3 splits
    python scripts/probe.py --model block4 --dataset HAM10000 --row 17    # one test row: its cells, photo, prediction with and without the photo
    python scripts/probe.py --dataset "Mango Mass" --no-references        # ours only: no DINOv2, TabPFN v2 or TabICL v2 (fast)

Datasets: a name from table_image_icl/multabench.py, a chico89 Kaggle slug, or a local folder in the same layout."""
import argparse
import os

import numpy as np
import torch

from table_image_icl import multabench
from table_image_icl.inputs import inputs_for
from table_image_icl.pretrained import RELEASES, load_pretrained
from table_image_icl.references import FROZEN, OURS, FrozenDinoFeatures, predictions
from table_image_icl.scoring import bootstrap, markdown, score

from benchmark import context_and_test  # noqa: E402  (scripts/benchmark.py)


def show_row(data, context, test, results, row: int) -> None:
    """One test row: its table cells, its photo's path, its label, and what each method predicts for it."""
    r = test[row]
    print(f"\nrow {r}: photo {data.paths[r]}")
    for name, value in zip(data.columns, data.table[r]):
        print(f"  {name:30s} {value}")
    print(f"  {'label':30s} {data.y[r]}")
    for method, result in results.items():
        if data.is_classification:
            probabilities, classes = result
            top = np.argsort(-probabilities[row])[:3]
            print(f"  {method:48s} " + ", ".join(f"{classes[i]}: {probabilities[row, i]:.2f}" for i in top))
        else:
            print(f"  {method:48s} {result[row]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=next(iter(RELEASES), "imagenet1k"), help=f"a published model: {list(RELEASES)}")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-context", type=int, default=256)
    ap.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--grouped", action="store_true", help="patients, listings and fruits kept on one side of the split")
    ap.add_argument("--no-references", action="store_true")
    ap.add_argument("--row", type=int, help="also print this test row of the first split in detail")
    ap.add_argument("--resamples", type=int, default=1000)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = load_pretrained(args.model, device=device)
    to_tokens = inputs_for(model.cfg.encoder, device)
    dino = None if args.no_references else FrozenDinoFeatures(device)
    data = multabench.load(multabench.ALL_SETS.get(args.dataset, args.dataset), grouped=args.grouped)
    name = args.dataset if args.dataset in multabench.ALL_SETS else os.path.basename(os.path.normpath(args.dataset))
    splits = []
    for k in args.splits:
        context, test = context_and_test(data, k, args.max_context)
        results = predictions(model, to_tokens, dino, data, context, test, device)
        splits.append((data.y[test], results))
        print(f"{name} split {k}: " + " | ".join(f"{m} {score(data.y[test], r, data.is_classification):.3f}" for m, r in results.items()), flush=True)
        if args.row is not None and k == args.splits[0]:
            show_row(data, context, test, results, args.row)
    reference = OURS if args.no_references else FROZEN
    result = {name: {"metric": "AUROC" if data.is_classification else "RMSE", "splits": len(splits),
                     "methods": bootstrap(splits, data.is_classification, reference, args.resamples)}}
    print("\n" + markdown(result, reference))


if __name__ == "__main__":
    main()
