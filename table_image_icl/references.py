"""The reference methods and scores of the benchmark: TabPFN v2 on the table, TabPFN v2 on the table plus a PCA of
frozen DINOv2 features (MulTaBench's frozen-encoder baseline), AUROC and R2."""
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, roc_auc_score

from table_image_icl.inputs import DINOV2, DINOV2_WEIGHTS, NormalisedPixels


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


def auroc(y: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    """AUROC, one-vs-rest beyond two classes. probabilities (rows, len(classes)) in the order of `classes`; classes no
    test row has are left out and the rest renormalised."""
    present = np.isin(classes, y)
    probabilities = probabilities[:, present] / probabilities[:, present].sum(1, keepdims=True)
    classes = classes[present]
    if len(classes) == 2:
        return float(roc_auc_score(y == classes[1], probabilities[:, 1]))
    return float(roc_auc_score(y, probabilities, multi_class="ovr", labels=classes))


def r2(y: np.ndarray, predictions: np.ndarray) -> float:
    return float(r2_score(y, predictions))
