"""The reference methods the model is scored against on real tables, and one split's predictions for every method.

MulTaBench's frozen baseline: each photo through a frozen DINOv2 ViT-B/14 (final CLS token and mean patch token),
standardised, PCA to 30 columns fitted on the context rows, appended to the table, then a tabular learner. Two
learners: TabPFN v2 (the benchmark's) and TabICL v2, the table model under ours, so that "TabICL v2, table + frozen
DINOv2 PCA" differs from ours only in how the photo comes in. Two controls for ours: the test rows given another test
row's photo, and every row, context included, given a random row's photo."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from sklearn.decomposition import PCA

from table_image_icl.inputs import DINOV2, DINOV2_WEIGHTS, NormalisedPixels, letterbox
from table_image_icl.predict import predict_classes, predict_values

FROZEN = "TabICL v2, table + frozen DINOv2 PCA"       # the reference every comparison is made against
OURS = "ours, table + image"


class FrozenDinoFeatures:
    """DINOv2 ViT-B/14's usual image features: the final CLS token and the mean patch token, (n, 1536)."""

    def __init__(self, device: str, chunk: int = 256):
        self.device, self.chunk = device, chunk
        self.model = torch.hub.load(DINOV2, DINOV2_WEIGHTS).to(device).eval().requires_grad_(False)
        self.pixels = NormalisedPixels(device)

    @torch.no_grad()
    def __call__(self, images: np.ndarray) -> np.ndarray:
        out = []
        for start in range(0, len(images), self.chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                t = self.model.prepare_tokens_with_masks(self.pixels(images[start:start + self.chunk]))
                for block in self.model.blocks:
                    t = block(t)
                t = self.model.norm(t)
            first_patch = 1 + self.model.num_register_tokens
            out.append(torch.cat([t[:, 0], t[:, first_patch:].mean(1)], 1).float().cpu().numpy())
        return np.concatenate(out)


def standardise(a: np.ndarray, n_context: int) -> np.ndarray:
    return (a - a[:n_context].mean(0)) / (a[:n_context].std(0) + 1e-6)


def pca_columns(features: np.ndarray, n_context: int, k: int = 30) -> np.ndarray:
    """MulTaBench's frozen baseline: standardise, then PCA with at most k components, fitted on the context rows."""
    a = standardise(features, n_context)
    k = min(k, n_context - 1, a.shape[1])
    pca = PCA(k, random_state=0).fit(a[:n_context])
    return standardise(pca.transform(a).astype(np.float32), n_context)


def tabpfn_v2(is_classification: bool, device: str):
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion
    cls = TabPFNClassifier if is_classification else TabPFNRegressor
    return cls.create_default_for_version(ModelVersion.V2, device=device, n_estimators=8, ignore_pretraining_limits=True)


def tabicl_v2(is_classification: bool, device: str):
    from tabicl import TabICLClassifier, TabICLRegressor
    return (TabICLClassifier if is_classification else TabICLRegressor)(device=device)


LEARNERS = {"TabPFN v2": tabpfn_v2, "TabICL v2": tabicl_v2}


def read_photos(paths, threads: int = 16) -> np.ndarray:
    with ThreadPoolExecutor(threads) as pool:
        return np.stack(list(pool.map(letterbox, paths)))


def other_rows(n: int, seed: int) -> np.ndarray:
    """A fixed permutation, so a control is the same in every run."""
    return np.random.default_rng(seed).permutation(n)


def predictions(model, to_tokens, dino, data, context, test, device, chunk: int = 512) -> dict:
    """Every method's prediction for one split, on the same test rows: (probabilities, classes) for classification,
    values for regression. `dino`: a FrozenDinoFeatures, or None to skip the reference methods."""
    x, y = data.table, data.y
    photos = read_photos(data.paths[context]), read_photos(data.paths[test])
    ours = predict_classes if data.is_classification else predict_values
    run = lambda **kw: ours(model, x[context], y[context], x[test], device=device, chunk=chunk, **kw)  # noqa: E731
    out = {"ours, table only": run(),
           OURS: run(photos=photos, to_tokens=to_tokens),
           "ours, image of another row": run(photos=(photos[0], photos[1][other_rows(len(test), 0)]), to_tokens=to_tokens)}
    everyone = np.concatenate(photos)[other_rows(len(context) + len(test), 1)]
    out["ours, photos of random rows"] = run(photos=(everyone[: len(context)], everyone[len(context):]), to_tokens=to_tokens)
    if dino is None:
        return out
    features = dino(np.concatenate(photos))
    plain = np.vstack([x[context], x[test]])
    frozen = np.hstack([plain, pca_columns(features, len(context))])
    for learner, build in LEARNERS.items():
        for name, table in [(f"{learner}, table only", plain), (f"{learner}, table + frozen DINOv2 PCA", frozen)]:
            reference = build(data.is_classification, device).fit(table[: len(context)], y[context])
            out[name] = (reference.predict_proba(table[len(context):]), reference.classes_) if data.is_classification else reference.predict(table[len(context):])
    return out
