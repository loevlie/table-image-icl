"""Score a trained run on held-out generated tasks (photos from collections never trained on), by what the photo
stands for and by view: table only (floor), table + photo, hidden column given back (ceiling), photo of another row.

    python scripts/evaluate.py --run runs/block4 --tasks-file evaluation-tasks.pt --out results/block4/evaluation.json

Every run is scored on the same fixed tasks: the tasks file is made once and shared."""
import argparse
import json
import os

import torch

from table_image_icl.collections import load_all, sha256
from table_image_icl.evaluation import score_by_role
from table_image_icl.inputs import inputs_for
from table_image_icl.model import load_run
from table_image_icl.tasks import TaskConfig, Tasks
from table_image_icl.training import EVALUATION_SEED, fixed_batches, provenance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks-file", required=True, help="the fixed evaluation tasks, shared by every run (made on first use)")
    ap.add_argument("--batches", type=int, default=100, help="evaluation batches of 4 tasks")
    ap.add_argument("--image-cache")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, record = load_run(args.run, device)
    collections = load_all(record["images_per_collection"], args.image_cache)
    tasks = Tasks(TaskConfig.from_record(record["tasks"]), collections, record["train_collections"], record["validation_collections"],
                  inputs_for(record["arm"], device, block=record["dino_block"]))
    batches = fixed_batches(args.tasks_file, tasks, EVALUATION_SEED, args.batches)

    report = {"run": args.run, "arm": record["arm"], **provenance(), "tasks": sum(len(b.x) for b in batches),
              "tasks_file": args.tasks_file, "tasks_sha256": sha256(args.tasks_file), "by_role": score_by_role(model, tasks, batches, device)}
    for role, r in report["by_role"].items():
        views = " | ".join(f"{v} {s['loss']:.3f}/{s['accuracy']:.3f}" for v, s in r["views"].items())
        print(f"photo as {role} ({r['views']['image']['tasks']} tasks): {views} | gap closed: accuracy {r['gap_closed']['accuracy']:.0%}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
