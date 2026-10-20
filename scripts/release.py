"""Publish a run's weights: print the checksums and the entry for table_image_icl/pretrained.py, and the command that
uploads the two files to a GitHub release under the names load_pretrained expects.

    python scripts/release.py --run runs/imagenet21k --name imagenet21k --tag v0.2"""
import argparse
import json
import os

from table_image_icl.pretrained import FILES, sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the run folder with config.json and model.pt")
    ap.add_argument("--name", required=True, help="the published model's name, e.g. imagenet21k")
    ap.add_argument("--tag", required=True, help="the release tag, e.g. v0.2")
    args = ap.parse_args()
    digests = {f: sha256(os.path.join(args.run, f)) for f in FILES}
    print("RELEASES entry for table_image_icl/pretrained.py:")
    print(f'    "{args.name}": {json.dumps({"tag": args.tag, "sha256": digests})},')
    uploads = " ".join(f'"{os.path.join(args.run, f)}#{args.name}-{f}"' for f in FILES)
    print("\nupload (gh release create makes the release if the tag is new; gh release upload adds to one that exists):")
    print(f"    gh release create {args.tag} {uploads} --title {args.tag} --notes 'model {args.name}'")


if __name__ == "__main__":
    main()
