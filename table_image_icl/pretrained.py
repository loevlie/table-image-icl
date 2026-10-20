"""Trained weights in one call: load_pretrained("imagenet21k") fetches the run's config.json and model.pt from the
repository's GitHub release into checkpoints/<name>/, checks their sha256, and returns the model in eval mode with
its record. A folder that is already there and matches is not fetched again.

    from table_image_icl.pretrained import load_pretrained
    model, record = load_pretrained("imagenet21k")

RELEASES lists every published model: scripts/release.py prints the entry for a run folder once its files are up."""
import hashlib
import os
import urllib.request

from table_image_icl.model import load_run

REPO = "https://github.com/loevlie/table-image-icl"
FILES = ("config.json", "model.pt")

RELEASES = {   # name: {"tag": release tag, "sha256": {file: digest}}; the assets are <tag>/<name>-<file>
}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_url(name: str, file: str, base: str = REPO) -> str:
    return f"{base}/releases/download/{RELEASES[name]['tag']}/{name}-{file}"


def fetch(name: str, folder: str = "checkpoints", base: str = REPO) -> str:
    """The release's files for `name` in folder/<name>/, downloaded unless present with the right checksum."""
    assert name in RELEASES, f"no release named {name}; known: {list(RELEASES)}"
    out = os.path.join(folder, name)
    os.makedirs(out, exist_ok=True)
    for file in FILES:
        path, expected = os.path.join(out, file), RELEASES[name]["sha256"][file]
        if os.path.exists(path) and sha256(path) == expected:
            continue
        urllib.request.urlretrieve(asset_url(name, file, base), path + ".part")
        actual = sha256(path + ".part")
        assert actual == expected, f"{file} of {name}: sha256 {actual}, expected {expected}"
        os.replace(path + ".part", path)
    return out


def load_pretrained(name: str = "imagenet21k", folder: str = "checkpoints", device: str = "cpu", base: str = REPO):
    """The published model `name`, as load_run returns it: (model in eval mode, its record)."""
    return load_run(fetch(name, folder, base), device)
