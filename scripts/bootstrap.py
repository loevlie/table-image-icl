"""Which method is best, with paired bootstrap intervals over the test rows (table_image_icl/scoring.py), from the
predictions files scripts/benchmark.py writes. Several runs can go into one table.

    python scripts/bootstrap.py --predictions "from scratch=results/imagenet21k/multabench-predictions.npz" \\
        "block4=results/block4/multabench-predictions.npz" --out results/bootstrap/validation.json"""
import argparse
import json
import os

from table_image_icl.references import FROZEN
from table_image_icl.scoring import markdown, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", nargs="+", required=True, help="label=file, one per run")
    ap.add_argument("--reference", default=FROZEN)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--datasets", nargs="+", help="only these datasets")
    ap.add_argument("--out", help="JSON record; the table is written next to it as .md")
    args = ap.parse_args()
    files = dict(item.split("=", 1) for item in args.predictions)
    result = report(files, args.reference, args.resamples, datasets=args.datasets)
    text = markdown(result, args.reference)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"files": files, "reference": args.reference, "resamples": args.resamples, "datasets": result}, f, indent=1)
        with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
