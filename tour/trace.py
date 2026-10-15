"""Tensor shapes per executed line, recorded from one small CPU run of the code the tour shows (tour/sections.py).

    python3 tour/trace.py            -> tour/data/traces.json

A trace function follows only frames of files under table_image_icl/ and scripts/. At every line event the frame's
locals are compared with the snapshot taken at the previous event of the same frame: every local that is a tensor or
an array and is a new object since then is recorded, as `name: shape`, under the line that just finished executing.
The first recording per (file, line) is kept. A call records the tensor arguments under the function's `def` line.
Output: {"table_image_icl/model.py": {"213": "queries: (2, 9, 4, 16); keys: (2, 6, 4, 16)", ...}, ...}."""
import contextlib
import importlib.util
import inspect
import io
import json
import os
import random
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACED = (os.path.join(REPO, "table_image_icl") + os.sep, os.path.join(REPO, "scripts") + os.sep)
OUT = os.path.join(REPO, "tour", "data", "traces.json")
DINOV2_CACHE = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8")

sys.path.insert(0, REPO)

from table_image_icl.collections import cached, split_attributes  # noqa: E402
from table_image_icl.inputs import DinoTokens, NormalisedPixels, inputs_for, letterbox  # noqa: E402
from table_image_icl.model import DINO_WIDTH, ModelConfig, TableImageICL, contiguous_codes, prepare_table  # noqa: E402
from table_image_icl.multabench import load, split  # noqa: E402
from table_image_icl.predict import predict_classes, predict_values  # noqa: E402
from table_image_icl.tasks import TaskConfig, Tasks, task_batches  # noqa: E402
from table_image_icl.training import TrainConfig, fixed_batches, forward_loss, learning_rate, make_optimizer, progress, task_loss, train, validate  # noqa: E402

# ----------------------------------------------------------------------------------------------------------- the tracer

records: dict[str, dict[int, str]] = {}          # repo-relative path -> line -> "name: shape; ..."
_state: dict[int, tuple[int | None, dict]] = {}  # id(frame) -> (line of the previous event, tensor locals then)
_lines: dict[str, list[str]] = {}
_def_lines: dict[tuple[str, int], int] = {}


def shape_of(value) -> tuple | None:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return tuple(int(s) for s in value.shape)
    return None


def tensor_locals(frame) -> dict[str, tuple[int, tuple]]:
    """name -> (identity, shape) for the locals that are tensors or arrays."""
    out = {}
    for name, value in frame.f_locals.items():
        shape = shape_of(value)
        if shape is not None:
            out[name] = (id(value), shape)
    return out


def record(path: str, line: int, items: list[tuple[str, tuple]]) -> None:
    if not items:
        return
    per_file = records.setdefault(os.path.relpath(path, REPO), {})
    if line not in per_file:
        per_file[line] = "; ".join(f"{name}: {shape}" for name, shape in items)


def def_line(path: str, first_line: int) -> int:
    """The `def` line of a function whose code starts at first_line (a decorator line, when decorated)."""
    key = (path, first_line)
    if key not in _def_lines:
        if path not in _lines:
            with open(path) as f:
                _lines[path] = f.read().splitlines()
        lines, n = _lines[path], first_line
        while n <= len(lines) and not lines[n - 1].lstrip().startswith(("def ", "async def ", "class ")):
            n += 1
        _def_lines[key] = n if n <= len(lines) else first_line
    return _def_lines[key]


def n_parameters(code) -> int:
    n = code.co_argcount + code.co_kwonlyargcount
    n += bool(code.co_flags & inspect.CO_VARARGS) + bool(code.co_flags & inspect.CO_VARKEYWORDS)
    return n


def local_tracer(frame, event, arg):
    if event not in ("line", "return"):
        return local_tracer
    key = id(frame)
    prev_line, prev = _state.get(key, (None, {}))
    current = tensor_locals(frame)
    if prev_line is not None:
        changed = [(name, shape) for name, (identity, shape) in current.items() if name not in prev or prev[name][0] != identity]
        record(frame.f_code.co_filename, prev_line, changed)
    if event == "line":
        _state[key] = (frame.f_lineno, current)
    else:
        _state.pop(key, None)
    return local_tracer


def global_tracer(frame, event, arg):
    code = frame.f_code
    if event != "call" or not code.co_filename.startswith(TRACED):
        return None
    current = tensor_locals(frame)
    parameters = code.co_varnames[: n_parameters(code)]
    record(code.co_filename, def_line(code.co_filename, code.co_firstlineno), [(name, current[name][1]) for name in parameters if name in current])
    _state[id(frame)] = (frame.f_lineno, current)
    return local_tracer


# ---------------------------------------------------------------------------------------------------- what is run

CFG = ModelConfig(encoder="pixels", encoder_layers=1, patch_layers=1, image_size=28)     # tests/test_model.py
N_CTX, N_TEST, F_COLS, P = 8, 4, 3, 4
Y = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1], [5, 6, 5, 6, 5, 6, 5, 6]])


def table(seed=0):
    return torch.randn(2, N_CTX + N_TEST, F_COLS, generator=torch.Generator().manual_seed(seed))


def photos(seed=0, size=28):
    return torch.randn(2, N_CTX + N_TEST, 3, size, size, generator=torch.Generator().manual_seed(seed))


def script(name: str):
    """One of the scripts under scripts/ as a module (the way tests/test_evaluation.py loads it)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pixels_model():
    """(a) TabICL v2's table stages with the from-scratch patch embedding, with and without photos; the read from
    every context photo and the top-k read with a small k so that the retrieval branch runs."""
    model = TableImageICL(CFG, pretrained=False).eval()
    with torch.no_grad():
        model(table(), Y, images=photos())
        model(table(), Y)
        model.photo_tokens(photos(), Y)
    model.load_weights(model.state_dict())
    for cfg in (ModelConfig(encoder="pixels", encoder_layers=1, patch_layers=1, image_size=28, patch_read="all"),
                ModelConfig(encoder="pixels", encoder_layers=1, patch_layers=1, image_size=28, patch_top_k=2)):
        model = TableImageICL(cfg, pretrained=False).eval()
        with torch.no_grad():
            model(table(), Y, images=photos())
        model.read.train().pick_sources(cfg.max_sources + 1, "cpu")


def dino_models():
    """(b) the same model on frozen DINOv2 tokens: block tokens (encoder block4) and the summary tokens (encoder cls)."""
    block = TableImageICL(ModelConfig(encoder="block4", encoder_layers=1, patch_layers=1, image_size=28), pretrained=False).eval()
    summary = TableImageICL(ModelConfig(encoder="cls", encoder_layers=1, patch_layers=1, image_size=28), pretrained=False).eval()
    with torch.no_grad():
        block(table(), Y, images=torch.randn(2, N_CTX + N_TEST, P, DINO_WIDTH))
        summary(table(), Y, images=torch.randn(2, N_CTX + N_TEST, 5, DINO_WIDTH))


def dino_hub():
    """(c) frozen DINOv2 tokens of two random photos, block and summary, and the pixel input (needs the hub cache)."""
    if not os.path.isdir(DINOV2_CACHE):
        raise RuntimeError("DINOv2 hub cache missing")
    raw = np.random.default_rng(0).integers(0, 256, (2, 224, 224, 3), dtype=np.uint8)
    block = DinoTokens("cpu", block=4)
    block(raw)
    DinoTokens("cpu", summary=True, model=block.model)(raw)
    inputs_for("pixels", "cpu")(raw)
    from PIL import Image
    letterbox(Image.fromarray(raw[0, :100]))


def table_preparation():
    """(d) TabICL's preprocessing and the class renumbering on a small table."""
    x = table()
    x[0, N_CTX + 1, 0] = float("nan")
    x[1, 2, 1] = 1e4
    prepare_table(x, N_CTX)
    contiguous_codes(Y, n_slots=10)


def tiny_tasks(rng: np.random.Generator) -> Tasks:
    def collection(n_classes):
        labels = np.repeat(np.arange(n_classes), 12)
        return {"images": rng.integers(0, 256, (len(labels), 28, 28, 3), dtype=np.uint8), "labels": labels, "kind": "class"}

    cfg = TaskConfig(batch_size=2, rows=(32, 48), features=(3, 6), max_classes=4, max_bins=4)
    return Tasks(cfg, {"seen": collection(8), "unseen": collection(6)}, ["seen"], ["unseen"], NormalisedPixels("cpu"))


def training_pieces():
    """(e) the loss, the schedule, the optimizer groups, one batch's loss and accuracy in every view, a validation."""
    log_probs = torch.randn(2, N_TEST, 10).log_softmax(-1)
    y_test = torch.tensor([[0, 1, 2, 0], [5, 6, 5, 6]])
    task_loss(log_probs, Y, y_test)
    cfg = TrainConfig(out="x", lr=1e-5, image_lr=1e-4, warmup_steps=10, max_steps=100)
    learning_rate(cfg, 5, 0.25, cfg.image_lr)
    progress(cfg, 25, 0.0)
    model = TableImageICL(CFG, pretrained=False)
    make_optimizer(model, cfg)
    tasks = tiny_tasks(np.random.default_rng(0))
    batch = next(tasks.validation_batches(seed=0))
    for view in tasks.VIEWS:
        forward_loss(model, tasks, batch, view, "cpu", amp=False)
    validate(model, tasks, [batch], "cpu", amp=False)


def task_generation():
    """(f) the task generator: one batch with the photo as a feature, one with it as the label, from a class
    collection and from an attribute collection."""
    rng = np.random.default_rng(0)
    collections = {"things": {"labels": np.repeat(np.arange(15), 40), "kind": "class"}, "faces": {"labels": rng.random((500, 8)) < 0.5, "kind": "attr"}}
    for share in (0.0, 1.0):
        cfg = TaskConfig(batch_size=2, rows=(32, 48), features=(3, 6), max_classes=4, max_bins=5, image_as_label_share=share)
        stream = task_batches(cfg, collections, seed=0)
        for _ in range(3):
            next(stream)


def prediction():
    """(g) predict.py on a small real-looking table: classes, classes with photos, regression through the bins."""
    torch.manual_seed(0)
    model = TableImageICL(CFG, pretrained=False).eval()
    rng = np.random.default_rng(0)
    x_context, x_test = rng.normal(size=(40, 5)), rng.normal(size=(9, 5))
    x_context[2, 1] = np.nan
    y = rng.choice([3, 7, 9], 40)
    predict_classes(model, x_context, y, x_test)
    photos_ = (rng.integers(0, 256, (40, 28, 28, 3), dtype=np.uint8), rng.integers(0, 256, (9, 28, 28, 3), dtype=np.uint8))
    predict_classes(model, x_context, y, x_test, chunk=4, photos=photos_, to_tokens=NormalisedPixels("cpu"))
    predict_values(model, x_context, rng.normal(size=40) * 50 + 200, x_test, n_bins=8)


def multabench_pieces():
    """(h) MulTaBench's splits on random labels of 300 rows, the grouped split, and a dataset loaded from a folder
    in the benchmark's layout (tests/test_multabench.py's fixture)."""
    rng = np.random.default_rng(0)
    split(rng.integers(0, 3, 300), is_classification=True, k=0)
    split(rng.normal(size=300), is_classification=False, k=1)
    split(rng.integers(0, 3, 300), is_classification=True, k=2, groups=rng.integers(0, 60, 300))
    folder = tempfile.mkdtemp(prefix="tour-trace-multabench-")
    try:
        n = 300
        df = pd.DataFrame({"age": rng.normal(50, 10, n), "side": rng.choice(["LEFT", "RIGHT"], n),
                           "report": [f"a long free text report number {i} " * 3 for i in range(n)],
                           "image": [f"images/{i}.png" for i in range(n)], "target": rng.choice(["benign", "malignant", "normal"], n)})
        df.loc[3, "age"] = np.nan
        df.to_csv(os.path.join(folder, "data.csv"), index=False)
        with open(os.path.join(folder, "metadata.json"), "w") as f:
            json.dump({"target": "target", "image_col": "image", "task_type": "cls"}, f)
        data = load(folder)
        split(data.y, data.is_classification, 0)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def collection_pieces():
    """(i) an attribute collection split into training and held-out attributes; a collection cached as .npz."""
    labels = np.arange(60).reshape(6, 10) % 2 == 0
    split_attributes({"images": np.zeros((6, 4, 4, 3), np.uint8), "labels": labels, "kind": "attr", "names": np.array([f"a{i}" for i in range(10)])}, n_heldout=3)
    folder = tempfile.mkdtemp(prefix="tour-trace-cache-")
    try:
        build = lambda: {"images": np.full((3, 4, 4, 3), 7, np.uint8), "labels": np.array([0, 1, 1]), "kind": "class"}  # noqa: E731
        cached(os.path.join(folder, "things-3.npz"), build)
        cached(os.path.join(folder, "things-3.npz"), build)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def script_pieces():
    """(j) the scoring in scripts/evaluate.py and the reference methods' pieces in scripts/benchmark.py that need no
    download: AUROC and R2, the PCA columns of the frozen baseline, the mean and standard error."""
    evaluate, benchmark = script("evaluate"), script("benchmark")
    torch.manual_seed(0)
    tasks = tiny_tasks(np.random.default_rng(0))
    stream = tasks.validation_batches(seed=0)
    batches = [next(stream) for _ in range(2)]
    evaluate.score_by_role(TableImageICL(CFG, pretrained=False), tasks, batches, "cpu", amp=False)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 50)
    probabilities = rng.dirichlet(np.ones(3), 50)
    benchmark.auroc(y, probabilities, np.arange(3))
    benchmark.auroc((y > 0).astype(int), probabilities[:, :2] / probabilities[:, :2].sum(1, keepdims=True), np.arange(2))
    benchmark.r2(rng.normal(size=50), rng.normal(size=50))
    benchmark.pca_columns(rng.normal(size=(60, 1536)).astype(np.float32), n_context=40)
    benchmark.mean_and_se([0.7, 0.8, 0.75])


def training_run():
    """(k) two steps of the training loop on tiny collections, in a temporary folder."""
    out = tempfile.mkdtemp(prefix="tour-trace-run-")
    try:
        run = TrainConfig(out=out, max_steps=2, validate_every=2, warmup_steps=2, n_validation_batches=2, workers=0, amp=False)
        torch.manual_seed(0)
        tasks = tiny_tasks(np.random.default_rng(0))
        train(run, TableImageICL(CFG, pretrained=False), tasks, {"model": {}, "tasks": {}})
        fixed_batches(os.path.join(out, "validation.pt"), tasks, 10_000, 2)
    finally:
        shutil.rmtree(out, ignore_errors=True)


SCENARIOS = [pixels_model, dino_models, dino_hub, table_preparation, training_pieces, task_generation, prediction, multabench_pieces,
             collection_pieces, script_pieces, training_run]


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    started = time.time()
    failures = []
    sys.settrace(global_tracer)
    try:
        for scenario in SCENARIOS:
            try:
                with contextlib.redirect_stdout(io.StringIO()):          # the training loop prints its metrics
                    scenario()
            except Exception as e:                                    # a scenario that cannot run here is reported, not fatal
                failures.append(f"{scenario.__name__}: {type(e).__name__}: {e}")
    finally:
        sys.settrace(None)
    out = {path: {str(line): text for line, text in sorted(lines.items())} for path, lines in sorted(records.items())}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    for path, lines in out.items():
        print(f"{len(lines):4d} lines  {path}")
    print(f"{sum(len(v) for v in out.values())} lines in {len(out)} files -> {os.path.relpath(OUT, REPO)} ({time.time() - started:.0f}s)")
    for failure in failures:
        print("skipped:", failure)


if __name__ == "__main__":
    main()
