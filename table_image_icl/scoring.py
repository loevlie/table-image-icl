"""Scores and the paired bootstrap that says which method is best.

Two scores, used everywhere: AUROC for classification (one-vs-rest beyond two classes) and RMSE for regression. The
benchmark script keeps every method's per-row predictions on the same test rows (save_predictions), and bootstrap()
resamples those rows, within each class so an AUROC always has every class, the same rows for every method
(paired), separately in each split, with the mean over splits as the statistic. For each method it gives the score
interval and the interval of the difference to a reference method, with the share of resamples in which the method
is the better one."""
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score

SEP = "\t"                                       # method names hold commas and spaces, never a tab


def auroc(y: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    """probabilities (rows, len(classes)) in the order of `classes`; classes no test row has are left out and the
    rest renormalised."""
    present = np.isin(classes, y)
    probabilities = probabilities[:, present] / probabilities[:, present].sum(1, keepdims=True)
    classes = classes[present]
    if len(classes) == 2:
        return float(roc_auc_score(y == classes[1], probabilities[:, 1]))
    return float(roc_auc_score(y, probabilities, multi_class="ovr", labels=classes))


def rmse(y: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, predictions)))


def score(y: np.ndarray, prediction, is_classification: bool) -> float:
    """prediction: (probabilities, classes) for classification, values for regression."""
    return auroc(y, *prediction) if is_classification else rmse(y, prediction)


# ---------- predictions files ----------

def save_predictions(path: str, records: dict) -> None:
    """records: {(dataset, split): {"y", "rows", "methods": {method: (probabilities, classes) | values}}} -> one .npz."""
    arrays = {}
    for (dataset, split), record in records.items():
        key = f"{dataset}{SEP}{split}"
        arrays[f"{key}{SEP}y"], arrays[f"{key}{SEP}rows"] = np.asarray(record["y"]), np.asarray(record["rows"])
        for method, prediction in record["methods"].items():
            if isinstance(prediction, tuple):
                arrays[f"{key}{SEP}{method}{SEP}proba"], arrays[f"{key}{SEP}{method}{SEP}classes"] = prediction
            else:
                arrays[f"{key}{SEP}{method}{SEP}values"] = np.asarray(prediction)
    np.savez_compressed(path, **arrays)


def load_predictions(path: str) -> dict:
    records = {}
    with np.load(path) as saved:
        for key in saved.files:
            dataset, split, *rest = key.split(SEP)
            record = records.setdefault((dataset, int(split)), {"methods": {}})
            if rest in (["y"], ["rows"]):
                record[rest[0]] = saved[key]
            elif rest[1] == "values":
                record["methods"][rest[0]] = saved[key]
            elif rest[1] == "proba":
                record["methods"][rest[0]] = (saved[key], saved[f"{dataset}{SEP}{split}{SEP}{rest[0]}{SEP}classes"])
    return records


# ---------- the paired bootstrap ----------

def _take(prediction, rows: np.ndarray):
    return (prediction[0][rows], prediction[1]) if isinstance(prediction, tuple) else prediction[rows]


def _resample(y: np.ndarray, is_classification: bool, rng: np.random.Generator) -> np.ndarray:
    if not is_classification:
        return rng.integers(0, len(y), len(y))
    return np.concatenate([rng.choice(r, len(r), replace=True) for r in (np.flatnonzero(y == c) for c in np.unique(y))])


def bootstrap(splits: list, is_classification: bool, reference: str, n_resamples: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """splits: [(y, {method: prediction}), ...], every method on the same rows of a split. Returns
    {method: {"point", "ci", "difference": {"point", "ci", "p_better"}}}: the difference is method minus reference,
    p_better the share of resamples where the method is better (higher AUROC, lower RMSE)."""
    methods = list(splits[0][1])
    rng = np.random.default_rng(seed)
    point = {m: float(np.mean([score(y, p[m], is_classification) for y, p in splits])) for m in methods}
    draws = np.empty((n_resamples, len(methods)))
    for b in range(n_resamples):
        per_split = []
        for y, predictions in splits:
            rows = _resample(y, is_classification, rng)
            per_split.append([score(y[rows], _take(predictions[m], rows), is_classification) for m in methods])
        draws[b] = np.mean(per_split, 0)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    ref = methods.index(reference)
    out = {}
    for j, m in enumerate(methods):
        diff = draws[:, j] - draws[:, ref]
        better = diff > 0 if is_classification else diff < 0
        out[m] = {"point": point[m], "ci": (float(np.percentile(draws[:, j], lo)), float(np.percentile(draws[:, j], hi))),
                  "difference": {"point": point[m] - point[reference], "ci": (float(np.percentile(diff, lo)), float(np.percentile(diff, hi))),
                                 "p_better": float(better.mean())}}
    return out


def merge(files: dict) -> dict:
    """{label: predictions file} -> {(dataset, split): (y, {method: prediction})} over every file. Our methods
    ("ours, ...") get their file's label in front when there is more than one file; the reference methods are the
    same in every file (same splits) and are kept once."""
    merged = {}
    for label, path in files.items():
        for key, record in load_predictions(path).items():
            y, methods = merged.setdefault(key, (record["y"], {}))
            assert np.array_equal(y, record["y"]), f"{path}: {key} has other test rows than an earlier file"
            for method, prediction in record["methods"].items():
                methods[f"{label}: {method}" if method.startswith("ours") and len(files) > 1 else method] = prediction
    return merged


def report(files: dict, reference: str, n_resamples: int = 2000, seed: int = 0, datasets=None) -> dict:
    """{dataset: {"metric", "splits", "methods": bootstrap(...)}} over the predictions files."""
    merged = merge(files)
    out = {}
    for dataset in dict.fromkeys(d for d, _ in merged):
        if datasets and dataset not in datasets:
            continue
        splits = [merged[(d, k)] for d, k in sorted(merged) if d == dataset]
        is_classification = isinstance(next(iter(splits[0][1].values())), tuple)
        out[dataset] = {"metric": "AUROC" if is_classification else "RMSE", "splits": len(splits),
                        "methods": bootstrap(splits, is_classification, reference, n_resamples, seed)}
    return out


def markdown(result: dict, reference: str) -> str:
    lines = []
    for dataset, r in result.items():
        lines += [f"**{dataset}** ({r['metric']}, {r['splits']} splits; reference: {reference})", "",
                  f"| method | {r['metric']} [95% CI] | vs reference [95% CI] | P(better) |", "|---|---|---|---|"]
        for m, v in r["methods"].items():
            d = v["difference"]
            lines.append(f"| {m} | {v['point']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] | {d['point']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] | {d['p_better']:.2f} |")
        lines.append("")
    return "\n".join(lines)
