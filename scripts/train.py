"""Train the model on generated tasks: synthetic tables whose rows carry a photo (table_image_icl/tasks.py).

    python scripts/train.py --out runs/block4 --encoder block4 --max-steps 25000 --hours 5

Running it again with the same folders resumes the run. What the run folder holds: see table_image_icl/training.py."""
import argparse
from dataclasses import asdict

import torch

from table_image_icl.collections import TRAIN_NAMES, VALIDATION_NAMES, load_all
from table_image_icl.inputs import inputs_for
from table_image_icl.model import CHECKPOINT, ENCODERS, TableImageICL, ModelConfig
from table_image_icl.tasks import TaskConfig, Tasks
from table_image_icl.training import TrainConfig, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the run's folder")
    ap.add_argument("--encoder", choices=list(ENCODERS), default="block4", help="where the patch tokens come from (table_image_icl/model.py)")
    ap.add_argument("--checkpoint-dir", help="where checkpoint.pt goes (default: the run's folder)")
    ap.add_argument("--hours", type=float)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--patch-read", choices=["all", "top_k"], default="top_k")
    ap.add_argument("--patch-top-k", type=int, default=32)
    ap.add_argument("--patch-layers", type=int, default=2)
    ap.add_argument("--encoder-layers", type=int, default=4)
    ap.add_argument("--image-as-label-share", type=float, default=0.5, help="share of batches where the photo stands for the label")
    ap.add_argument("--images-per-collection", type=int, default=20_000)
    ap.add_argument("--image-cache", help="folder where decoded collections are kept between runs")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5, help="the pretrained table stages")
    ap.add_argument("--image-lr", type=float, default=1e-4, help="the image side")
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--validate-every", type=int, default=250)
    ap.add_argument("--n-validation-batches", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = ModelConfig(encoder=args.encoder, encoder_layers=args.encoder_layers, patch_layers=args.patch_layers, patch_read=args.patch_read, patch_top_k=args.patch_top_k)
    torch.manual_seed(args.seed)
    model = TableImageICL(model_cfg)

    collections = load_all(args.images_per_collection, args.image_cache)
    task_cfg = TaskConfig(batch_size=args.batch_size, image_as_label_share=args.image_as_label_share)
    tasks = Tasks(task_cfg, collections, TRAIN_NAMES, VALIDATION_NAMES, inputs_for(args.encoder, device))

    run = TrainConfig(out=args.out, checkpoint_dir=args.checkpoint_dir, hours=args.hours, max_steps=args.max_steps, lr=args.lr,
                      image_lr=args.image_lr, warmup_steps=args.warmup_steps, validate_every=args.validate_every,
                      n_validation_batches=args.n_validation_batches, workers=args.workers, seed=args.seed)
    settings = {"model": asdict(model_cfg), "tasks": asdict(task_cfg), "tabicl_checkpoint": CHECKPOINT, "images_per_collection": args.images_per_collection,
                "train_collections": TRAIN_NAMES, "validation_collections": VALIDATION_NAMES, "p_no_image": tasks.p_no_image,
                "collections_sha256": {name: c.get("sha256") for name, c in collections.items()}}
    train(run, model, tasks, settings)


if __name__ == "__main__":
    main()
