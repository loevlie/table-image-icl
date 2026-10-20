"""Score a trained run on held-out generated tasks (photos from collections never trained on), by what the photo
stands for and by view: table only (floor), table + photo, hidden column given back (ceiling), photo of another row.

    python scripts/evaluate.py --run runs/block4 --tasks-file evaluation-tasks.pt --out results/block4/evaluation.json

Every run is scored on the same fixed tasks: the tasks file is made once and shared. The report is by what the photo
stands for (a feature or the label) and by view, with the share of the floor-to-ceiling gap the photo closes."""
import argparse
import json
import os

import numpy as np
import torch
from torch import nn

from table_image_icl import photos
from table_image_icl.training import sha256
from table_image_icl.inputs import inputs_for
from table_image_icl.model import load_run
from table_image_icl.tasks import TaskConfig, Tasks
from table_image_icl.training import EVALUATION_SEED, fixed_batches, forward_loss, provenance


@torch.no_grad()
def score_by_role(model: nn.Module, tasks, batches: list, device: str, amp: bool = True) -> dict:
    """{role: {"views": {view: {"loss", "accuracy", "tasks"}}, "image gain", "gap_closed"}}. The image gain is the
    per-batch difference in loss between no image and image, as a mean, a standard error and the share of batches
    where the photo helped; every view sees the same tasks, so the batches are paired."""
    model.eval()
    per_batch = {}
    for batch in batches:
        for view in tasks.VIEWS:
            loss, accuracy = forward_loss(model, tasks, batch, view, device, amp)
            per_batch.setdefault(batch.image_role, {}).setdefault(view, []).append((loss.item(), accuracy, len(batch.x)))
    report = {}
    for role, views in per_batch.items():
        summary = {view: {"loss": float(np.mean([loss for loss, _, _ in r])), "accuracy": float(np.mean([a for _, a, _ in r])),
                          "tasks": int(sum(n for _, _, n in r))} for view, r in views.items()}
        gain = np.array([f[0] - i[0] for f, i in zip(views["no image"], views["image"])])
        report[role] = {"views": summary, "gap_closed": gap_closed(summary),
                        "image gain": {"loss_mean": float(gain.mean()), "loss_se": float(gain.std(ddof=1) / np.sqrt(len(gain))) if len(gain) > 1 else float("nan"),
                                       "share_of_batches_helped": float((gain > 0).mean()), "batches": len(gain)}}
    return report


def gap_closed(views: dict) -> dict:
    """How much of the gap between the table alone (floor) and the hidden column given back (ceiling) the photo
    closes, in loss and in accuracy: 0 = the photo adds nothing, 1 = as good as seeing the hidden variable."""
    floor, image, ceiling = views["no image"], views["image"], views["column visible"]

    def share(numerator, denominator):
        return numerator / denominator if denominator else float("nan")

    return {"loss": share(floor["loss"] - image["loss"], floor["loss"] - ceiling["loss"]),
            "accuracy": share(image["accuracy"] - floor["accuracy"], ceiling["accuracy"] - floor["accuracy"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks-file", required=True, help="the fixed evaluation tasks, shared by every run (made on first use)")
    ap.add_argument("--batches", type=int, default=100, help="evaluation batches of 4 tasks")
    ap.add_argument("--photo-cache", default="cache", help="where decoded photo sources live (scripts/photos.py)")
    ap.add_argument("--in-memory", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, record = load_run(args.run, device)
    train_photos, validation_photos = photos.prepare(record["photos"]["source"], args.photo_cache, record["photos"]["shards"], args.in_memory, record["photos"]["max_photos"])
    tasks = Tasks(TaskConfig.from_record(record["tasks"]), {"train": train_photos, "validation": validation_photos}, ["train"], ["validation"],
                  inputs_for(model.cfg.encoder, device))
    batches = fixed_batches(args.tasks_file, tasks, EVALUATION_SEED, args.batches)

    report = {"run": args.run, "encoder": model.cfg.encoder, **provenance(), "tasks": sum(len(b.x) for b in batches),
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
