"""Using a trained model on one real table, with or without a photo per row. Regression goes through the same head
as classification: the context targets are cut into quantile bins, the bins are predicted as classes, and the
prediction is the expected bin mean (TabPFN's recipe, with far fewer bins)."""
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from table_image_icl.model import N_SLOTS


def bin_targets(y: np.ndarray, n_context: int, n_bins: int = N_SLOTS) -> tuple[np.ndarray, np.ndarray]:
    """y (n,), context rows first. Returns each row's bin (n,) and the mean context target of every bin. Edges are
    quantiles of the context targets, taken at actual values so that a column with few distinct values gets one bin
    per value; every bin holds at least one context row."""
    context = y[:n_context]
    edges = np.unique(np.quantile(context, np.linspace(0, 1, n_bins + 1)[1:-1], method="lower"))
    raw = np.searchsorted(edges, y, side="left")
    used = np.unique(raw[:n_context])
    nearest = np.abs(raw[:, None] - used[None, :]).argmin(1)
    means = np.array([context[nearest[:n_context] == b].mean() for b in range(len(used))])
    return nearest, means


def expected_value(log_probabilities: Tensor, bin_means: Tensor) -> Tensor:
    """Point prediction from a distribution over bins: (..., bins) -> (...)."""
    return log_probabilities.softmax(-1) @ bin_means


@torch.no_grad()
def predict_classes(model, x_context: np.ndarray, y_context: np.ndarray, x_test: np.ndarray, device: str = "cpu",
                    chunk: int = 2000, photos: tuple | None = None, to_tokens: Callable | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Class probabilities for the test rows, (n_test, K), and the K classes they refer to (the sorted context labels).
    NaN in x = missing. Test rows are predicted in chunks; each chunk sees the whole context. With photos, a pair
    (context photos, test photos) of uint8 arrays (rows, H, W, 3) and the encoder's to_tokens (inputs.py)."""
    classes, slots = np.unique(y_context, return_inverse=True)
    context = torch.tensor(x_context, dtype=torch.float32, device=device)
    y = torch.tensor(slots, device=device)[None]
    probabilities = []
    for start in range(0, len(x_test), chunk):
        test = torch.tensor(x_test[start:start + chunk], dtype=torch.float32, device=device)
        inputs = {}
        if photos is not None:
            pixels = np.concatenate([photos[0], photos[1][start:start + chunk]])
            inputs["images"] = to_tokens(pixels).to(device)[None]
        log_probs = model(torch.cat([context, test])[None], y, **inputs)[0, :, : len(classes)]
        probabilities.append(log_probs.softmax(-1).cpu().numpy())
    return np.concatenate(probabilities), classes


def predict_values(model, x_context: np.ndarray, y_context: np.ndarray, x_test: np.ndarray, n_bins: int = N_SLOTS,
                   device: str = "cpu", chunk: int = 2000, photos: tuple | None = None, to_tokens: Callable | None = None) -> np.ndarray:
    """Regression through the classifier: (n_test,) predicted values."""
    bins, bin_means = bin_targets(y_context, len(y_context), n_bins)
    probabilities, _ = predict_classes(model, x_context, bins, x_test, device, chunk, photos, to_tokens)
    return probabilities @ bin_means
