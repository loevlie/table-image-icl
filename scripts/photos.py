"""Download and decode a photo source for training (table_image_icl/photos.py).

    python scripts/photos.py imagenet1k --cache cache                 # 16 shards, about 20,000 photos, a few minutes
    python scripts/photos.py imagenet21k --cache cache --workers 64   # 100 shards, 1.28M photos, 181 GB decoded

Needs the source's terms accepted on the Hub (timm/imagenet-1k-wds, timm/imagenet-w21-webp-wds) and `hf auth login`.
The shards are fetched at a pinned revision into <cache>/<source>-shards/ with a MANIFEST.json, then decoded into
<cache>/<source>-<shards>/ (images.npy, labels.npy, names.json, done.json). Rerunning skips what is there. On a
cluster whose compute nodes are offline, run --download-only on the login node and --decode-only in a job."""
import argparse
import os

from table_image_icl import photos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=list(photos.SOURCES))
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--shards", type=int, help="how many of the source's training shards (default: the source's number)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--decode-only", action="store_true")
    args = ap.parse_args()
    shards_dir = os.path.join(args.cache, f"{args.source}-shards")
    out = photos.decoded_folder(args.source, args.cache, args.shards)
    if not args.decode_only:
        paths = photos.download(args.source, shards_dir, args.shards)
        print(f"{len(paths)} shards in {shards_dir}")
    if not args.download_only:
        if os.path.exists(os.path.join(out, "done.json")):
            print(f"{out} is already decoded")
            return
        import json
        names = json.load(open(os.path.join(shards_dir, "MANIFEST.json")))["files"]
        paths = [os.path.join(shards_dir, f) for f in list(names)[: args.shards or photos.SOURCES[args.source].shards]]
        photos.decode(paths, out, args.size, args.workers, manifest=os.path.join(shards_dir, "MANIFEST.json"))
        print(f"decoded into {out}: " + str(photos.load_photos(out)["record"]))


if __name__ == "__main__":
    main()
