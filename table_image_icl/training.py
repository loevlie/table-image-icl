"""Training on a stream of generated tasks. A run's folder is its record:

    config.json      every setting, the code version, the machine, the start time
    metrics.jsonl    one line per validation: step, training loss and gradient norm since the last line, loss and
                     accuracy on the validation tasks in every view, the learning rate, hours trained
    validation.pt    the fixed validation batches, the same for the whole run
    model.pt         the weights with the best validation loss so far
    model-final.pt   the weights when the run ended
    checkpoint.pt    everything needed to resume (in checkpoint_dir, which can be a scratch disk)

Starting a run again with the same folders resumes it."""
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from table_image_icl.model import IMAGE_SIDE
from table_image_icl.tasks import Batch

VALIDATION_SEED = 10_000
EVALUATION_SEED = 20_000      # the fixed evaluation tasks every run is scored on; never a training or validation seed
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class TrainConfig:
    out: str                              # the run's folder
    checkpoint_dir: str | None = None     # where checkpoint.pt goes; the run's folder if None
    hours: float | None = None            # stop after this much training time ...
    max_steps: int | None = None          # ... or this many steps, whichever comes first
    lr: float = 1e-5                      # the pretrained table stages
    image_lr: float = 1e-4                # the image side, trained from scratch
    lr_end: float = 0.0                   # the cosine decays to this
    clip: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 500
    validate_every: int = 250
    n_validation_batches: int = 25
    workers: int = 8
    seed: int = 0
    amp: bool = True                      # bfloat16 autocast on a GPU


def task_loss(log_probs: Tensor, y_context: Tensor, y_test: Tensor) -> tuple[Tensor, float]:
    """Mean negative log-likelihood and accuracy over the test rows whose class appears in their task's context;
    a class the context never shows cannot be predicted from it."""
    predictable = (y_test[:, :, None] == y_context[:, None, :]).any(-1)
    loss = F.nll_loss(log_probs[predictable], y_test[predictable])
    accuracy = (log_probs.argmax(-1) == y_test)[predictable].float().mean().item()
    return loss, accuracy


def learning_rate(cfg: TrainConfig, step: int, progress: float, base: float) -> float:
    """Linear warm-up, then a cosine decay from the base rate to lr_end as the budget is used up."""
    warm_up = min(1.0, step / cfg.warmup_steps)
    return warm_up * (cfg.lr_end + (base - cfg.lr_end) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with two groups: the table stages at lr, the image side at image_lr."""
    image = [p for n, p in model.named_parameters() if n.startswith(IMAGE_SIDE)]
    table = [p for n, p in model.named_parameters() if not n.startswith(IMAGE_SIDE)]
    groups = [{"params": table, "base_lr": cfg.lr}, {"params": image, "base_lr": cfg.image_lr}]
    return torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)


def progress(cfg: TrainConfig, step: int, elapsed: float) -> float:
    """The share of the budget (time or steps) used up."""
    shares = [0.0]
    if cfg.hours is not None:
        shares.append(elapsed / (cfg.hours * 3600))
    if cfg.max_steps is not None:
        shares.append(step / cfg.max_steps)
    return max(shares)


def save_checkpoint(path: str, model, optimizer, step: int, elapsed: float, best_loss: float) -> None:
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "elapsed": elapsed, "best_loss": best_loss}
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)                              # never a half-written checkpoint


def load_checkpoint(path: str, model, optimizer) -> tuple[int, float, float]:
    """Restores the weights and the optimizer; returns (step, seconds trained, best validation loss)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["step"], state["elapsed"], state["best_loss"]


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def code_version() -> str:
    try:
        commit = subprocess.check_output(["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", REPO_ROOT, "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        return commit + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def provenance() -> dict:
    """What produced a result: the code version, the command, the machine, the library versions."""
    libraries = {}
    for name in ("torch", "numpy", "scikit-learn", "tabpfn", "tabicl", "datasets"):
        try:
            libraries[name] = version(name)
        except Exception:
            pass
    return {"code_version": code_version(), "command": " ".join(sys.argv), "machine": platform.node(),
            "libraries": libraries, "started": time.strftime("%Y-%m-%d %H:%M:%S")}


FREE_ON_RESUME = ("out", "checkpoint_dir", "hours", "max_steps")


def check_resumed_record(cfg: TrainConfig, settings: dict) -> None:
    """A run resumed with other settings would train under a record that no longer describes it. The budget may
    grow and the folders may move; everything else must match."""
    with open(os.path.join(cfg.out, "config.json")) as f:
        record = json.load(f)
    expected = {"train": {k: v for k, v in asdict(cfg).items() if k not in FREE_ON_RESUME}, **settings}
    recorded = {"train": {k: v for k, v in record["train"].items() if k not in FREE_ON_RESUME}, **{k: record[k] for k in settings}}
    changed = [k for k in expected if json.dumps(expected[k], sort_keys=True) != json.dumps(recorded[k], sort_keys=True)]
    assert not changed, f"resuming {cfg.out} with other settings than its record: {changed}"


def fixed_batches(path: str, tasks, seed: int, n_batches: int) -> list:
    """Validation batches, generated once and saved at path, read from there after: the same tasks every time."""
    if not os.path.exists(path):
        stream = tasks.validation_batches(seed)
        torch.save([tuple(next(stream)) for _ in range(n_batches)], path)
    return [Batch(*b) for b in torch.load(path, weights_only=True)]


def forward_loss(model: nn.Module, tasks, batch, view: str, device: str, amp: bool, training: bool = False) -> tuple[Tensor, float]:
    """Loss and accuracy of one batch, shown to the model in the given way."""
    inputs = tasks.inputs(batch, device, view, training)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda"):
        log_probs = model(**inputs)
    return task_loss(log_probs, inputs["y_context"], batch.y_test.to(device))


@torch.no_grad()
def validate(model: nn.Module, tasks, batches: list, device: str, amp: bool) -> dict:
    """{view: {"loss", "accuracy"}} over the batches. Leaves the model as it was: training mode, no gradients."""
    model.eval()
    views = {}
    for view in tasks.VIEWS:
        results = [forward_loss(model, tasks, b, view, device, amp) for b in batches]
        views[view] = {"loss": sum(r[0].item() for r in results) / len(results), "accuracy": sum(r[1] for r in results) / len(results)}
    model.train()
    return views


def train(cfg: TrainConfig, model: nn.Module, tasks, settings: dict) -> None:
    """Train the model on the tasks' stream within the budget. settings: whatever else describes the run (the model
    and task settings); it goes into config.json. If the run's checkpoint exists, the run continues from it."""
    assert cfg.hours is not None or cfg.max_steps is not None, "give the run a budget: hours or max_steps"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = os.path.join(cfg.checkpoint_dir or cfg.out, "checkpoint.pt")
    os.makedirs(cfg.out, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)

    model.to(device)
    optimizer = make_optimizer(model, cfg)
    step, trained, best_loss = 0, 0.0, math.inf
    if os.path.exists(checkpoint):
        step, trained, best_loss = load_checkpoint(checkpoint, model, optimizer)
        check_resumed_record(cfg, settings)
        with open(os.path.join(cfg.out, "metrics.jsonl"), "a") as f:
            f.write(json.dumps({"resumed_at": step}) + "\n")
        print(f"resumed at step {step} after {trained / 3600:.2f} h", flush=True)
    else:
        record = {"train": asdict(cfg), **settings, "parameters": sum(p.numel() for p in model.parameters()), **provenance()}
        with open(os.path.join(cfg.out, "config.json"), "w") as f:
            json.dump(record, f, indent=1)
        if os.path.exists(os.path.join(cfg.out, "validation.pt")):             # a fresh start keeps no fixed tasks of an
            os.remove(os.path.join(cfg.out, "validation.pt"))                  # earlier launch: they were made under its settings
    validation = fixed_batches(os.path.join(cfg.out, "validation.pt"), tasks, VALIDATION_SEED, cfg.n_validation_batches)

    loader = DataLoader(tasks.stream(seed=cfg.seed + 1 + step), batch_size=None, num_workers=cfg.workers, prefetch_factor=4 if cfg.workers else None)
    losses, norms = [], []
    session_start = time.time()

    def elapsed() -> float:
        return trained + time.time() - session_start

    def log_and_save() -> None:
        nonlocal best_loss
        views = validate(model, tasks, validation, device, cfg.amp)
        main = views[tasks.VIEWS[0]]
        if main["loss"] < best_loss:
            best_loss = main["loss"]
            torch.save(model.state_dict(), os.path.join(cfg.out, "model.pt"))
        entry = {"step": step, "train_loss": sum(losses) / max(len(losses), 1), "validation_loss": main["loss"],
                 "validation_accuracy": main["accuracy"], "best_validation_loss": best_loss, "views": views,
                 "grad_norm": sum(norms) / max(len(norms), 1), "lr": optimizer.param_groups[0]["lr"], "elapsed_hours": elapsed() / 3600}
        with open(os.path.join(cfg.out, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
        losses.clear()
        norms.clear()
        save_checkpoint(checkpoint, model, optimizer, step, elapsed(), best_loss)

    model.train()
    for batch in loader:
        done = progress(cfg, step, elapsed())
        if done >= 1:
            break
        step += 1
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(cfg, step, done, group["base_lr"])
        loss, _ = forward_loss(model, tasks, batch, tasks.training_view(), device, cfg.amp, training=True)
        optimizer.zero_grad()
        loss.backward()
        norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip).item())
        optimizer.step()
        losses.append(loss.item())
        if step % cfg.validate_every == 0:
            log_and_save()
    if losses:
        log_and_save()
    torch.save(model.state_dict(), os.path.join(cfg.out, "model-final.pt"))
