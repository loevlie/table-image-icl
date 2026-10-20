"""Train the model on generated tasks: synthetic tables whose rows carry a photo (table_image_icl/tasks.py).

    python scripts/train.py --out runs/pixels --photos imagenet21k --photo-cache cache --in-memory --max-steps 25000
    python scripts/train.py --out runs/try --photos imagenet1k --photo-cache cache --max-steps 200      # a laptop

The photos come from one decoded source (table_image_icl/photos.py; scripts/photos.py downloads and decodes it).
Running it again with the same folders resumes the run. What the run folder holds: see table_image_icl/training.py."""
import argparse
from dataclasses import asdict

import torch

from table_image_icl import photos
from table_image_icl.inputs import inputs_for
from table_image_icl.model import CHECKPOINT, ENCODERS, TableImageICL, ModelConfig
from table_image_icl.tasks import TaskConfig, Tasks
from table_image_icl.training import TrainConfig, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the run's folder")
    ap.add_argument("--encoder", choices=list(ENCODERS), default="pixels", help="where the patch tokens come from (table_image_icl/model.py): pixels = trained from scratch, block4 = frozen DINOv2 blocks 1-4")
    ap.add_argument("--photos", choices=list(photos.SOURCES), default="imagenet21k", help="the photo source (table_image_icl/photos.py); imagenet1k is the small one for a laptop")
    ap.add_argument("--photo-cache", default="cache", help="where decoded photo sources live (scripts/photos.py)")
    ap.add_argument("--shards", type=int, help="shards of the source that were decoded (default: the source's number)")
    ap.add_argument("--in-memory", action="store_true", help="read the decoded photos into RAM (181 GB for imagenet21k): on a network file system a memmap makes every step wait on random reads")
    ap.add_argument("--max-photos", type=int, default=0, help="only the first photos of the source, for a quick test; 0 = all")
    ap.add_argument("--checkpoint-dir", help="where checkpoint.pt goes (default: the run's folder)")
    ap.add_argument("--hours", type=float)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--patch-read", choices=["all", "top_k"], default="top_k")
    ap.add_argument("--patch-top-k", type=int, default=32)
    ap.add_argument("--patch-layers", type=int, default=2)
    ap.add_argument("--encoder-layers", type=int, default=4)
    ap.add_argument("--image-as-label-share", type=float, default=0.5, help="share of batches where the photo stands for the label")
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

    train_photos, validation_photos = photos.prepare(args.photos, args.photo_cache, args.shards, args.in_memory, args.max_photos)
    task_cfg = TaskConfig(batch_size=args.batch_size, image_as_label_share=args.image_as_label_share)
    tasks = Tasks(task_cfg, {"train": train_photos, "validation": validation_photos}, ["train"], ["validation"], inputs_for(args.encoder, device))

    run = TrainConfig(out=args.out, checkpoint_dir=args.checkpoint_dir, hours=args.hours, max_steps=args.max_steps, lr=args.lr,
                      image_lr=args.image_lr, warmup_steps=args.warmup_steps, validate_every=args.validate_every,
                      n_validation_batches=args.n_validation_batches, workers=args.workers, seed=args.seed)
    settings = {"model": asdict(model_cfg), "tasks": asdict(task_cfg), "tabicl_checkpoint": CHECKPOINT, "p_no_image": tasks.p_no_image,
                "photos": {"source": args.photos, "shards": args.shards or photos.SOURCES[args.photos].shards, "max_photos": args.max_photos,
                           "validation_share": photos.VALIDATION_SHARE, "min_photos": photos.MIN_PHOTOS, "decoded": train_photos["record"]}}
    train(run, model, tasks, settings)


if __name__ == "__main__":
    main()
