"""Data for the equivalence cards: for every port in tour/provenance.py, our code and the upstream code, both run on
one fixed sample input (as images, with the largest difference and the tolerance the test asserts with), and the
outcome of the Hypothesis differential test, measured here on the same strategies.

    python3 tour/equivalence.py          writes tour/data/equivalence.json (CPU, under three minutes)

The sample inputs are seeded, so running it twice gives the same file."""
import base64
import inspect
import io
import json
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))                       # the test modules import each other by name
from tour.provenance import HUB, PORTS, installed_lines  # noqa: E402

OUT = os.path.join(ROOT, "tour", "data", "equivalence.json")
HYPOTHESIS_EXAMPLES = 200


# ---------- images ----------

def png(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def heatmap(values: np.ndarray, title: str, vmin=None, vmax=None, cmap="viridis") -> str:
    fig, ax = plt.subplots(figsize=(3, 3))
    image = ax.imshow(np.asarray(values, dtype=np.float64), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    return png(fig)


def lines(curves: dict, title: str, xlabel: str = "step") -> str:
    fig, ax = plt.subplots(figsize=(3, 3))
    for label, (x, y) in curves.items():
        ax.plot(x, y, label=label, linewidth=1)
    ax.set_title(title, fontsize=9), ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.yaxis.get_offset_text().set_fontsize(7)
    if len(curves) > 1:
        ax.legend(fontsize=7)
    return png(fig)


def three_heatmaps(ours: np.ndarray, theirs: np.ndarray, what: str) -> dict:
    """ours and theirs on one colour scale, the difference on its own symmetric scale."""
    lo, hi = float(min(ours.min(), theirs.min())), float(max(ours.max(), theirs.max()))
    diff = ours - theirs
    span = float(np.abs(diff).max()) or 1e-12
    return {"ours_png": heatmap(ours, f"ours: {what}", lo, hi), "theirs_png": heatmap(theirs, f"upstream: {what}", lo, hi),
            "diff_png": heatmap(diff, "ours - upstream", -span, span, cmap="RdBu_r"), "max_abs_diff": float(np.abs(diff).max())}


# ---------- the Hypothesis outcome, measured on the test's own strategies ----------

def hypothesis_outcome(test_id: str, strategy, pair, check) -> dict:
    """Runs `pair` on HYPOTHESIS_EXAMPLES draws of `strategy` (no shrinking, no database) and counts the draws on
    which `check(ours, theirs)` raises; max_abs_diff is the largest difference seen over all draws."""
    from hypothesis import HealthCheck, Phase, given, settings
    seen = {"examples": 0, "failures": 0, "max_abs_diff": 0.0}

    @settings(max_examples=HYPOTHESIS_EXAMPLES, deadline=None, database=None, phases=[Phase.generate],
              suppress_health_check=list(HealthCheck), derandomize=True)
    @given(strategy)
    def run(case):
        ours, theirs = pair(*case)
        seen["examples"] += 1
        seen["max_abs_diff"] = max(seen["max_abs_diff"], abs_diff(ours, theirs))
        try:
            check(ours, theirs, case)
        except AssertionError:
            seen["failures"] += 1

    run()
    return {"test": test_id, **seen}


def abs_diff(ours, theirs) -> float:
    if isinstance(ours, tuple):                                 # split: index arrays; 0 when identical, else 1
        return 0.0 if all(np.array_equal(a, b) for a, b in zip(ours, theirs, strict=True)) else 1.0
    return float(np.abs(np.asarray(ours, dtype=np.float64) - np.asarray(theirs, dtype=np.float64)).max())


def check_true(ok: bool) -> None:
    assert ok


# ---------- the sample runs, one per port ----------

def sample_prepare_table(d) -> tuple[dict, dict]:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 12, 5)) * rng.uniform(0.5, 5, (2, 1, 5)) + rng.uniform(-3, 3, (2, 1, 5))
    x[0, 2, 1], x[1, 10, 3] = 40.0, -30.0                       # an outlier in the context, one among the test rows
    n_context = 8
    pairs = [d.prepare_table_pair(x[t], n_context) for t in range(2)]
    ours, theirs = (np.hstack([p[i] for p in pairs]) for i in range(2))
    tolerance = max(d.prepare_table_tolerance(x[0])["atol"], d.prepare_table_tolerance(x[1])["atol"])
    sample = {"description": "Two seeded 12 x 5 tables side by side (first 8 rows are the context), one with an outlier in "
                             "its context and one with an outlier among its test rows; the cells after both preprocessings.",
              **three_heatmaps(ours, theirs, "prepared cells"), "tolerance": tolerance,
              "tolerance_text": f"rtol 1e-6, atol {tolerance:.2e} (1e-6 plus the mean's rounding over the 1e-6 spread floor)"}
    hypothesis = hypothesis_outcome("tests/test_differential.py::test_prepare_table_matches_tabicl_s_two_transformers_on_random_tables",
                                    d.tables(), d.prepare_table_pair,
                                    lambda o, t, case: torch.testing.assert_close(torch.tensor(o), torch.tensor(t), **d.prepare_table_tolerance(case[0])))
    return sample, hypothesis


def sample_split(d) -> tuple[dict, dict]:
    y = np.random.default_rng(0).integers(0, 3, 300)
    (ours_train, ours_test), (theirs_train, theirs_test) = d.split_pair(y, True, 1)
    ours, theirs = np.zeros(300), np.zeros(300)
    ours[ours_test], theirs[theirs_test] = 1, 1
    assert ours[ours_train].sum() == 0 and theirs[theirs_train].sum() == 0
    sample = {"description": "300 rows with 3 classes, split 1 of 6: each row's side, 0 = train, 1 = test, laid out in a 15 x 20 grid.",
              **three_heatmaps(ours.reshape(15, 20), theirs.reshape(15, 20), "test rows"), "tolerance": 0.0,
              "tolerance_text": "exact: the same train and test row indices (np.array_equal)"}
    hypothesis = hypothesis_outcome("tests/test_differential.py::test_split_gives_multabench_s_rows_on_random_labels",
                                    d.labelled_rows(), d.split_pair,
                                    lambda o, t, case: check_true(all(np.array_equal(a, b) for a, b in zip(o, t, strict=True))))
    return sample, hypothesis


def sample_learning_rate(d) -> tuple[dict, dict]:
    from table_image_icl.training import TrainConfig, learning_rate
    lr, lr_end, warmup, max_steps = 3e-4, 1e-5, 100, 1000
    cfg = TrainConfig(out="x", lr=lr, lr_end=lr_end, warmup_steps=warmup, max_steps=max_steps)
    optimizer = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr_end)
    steps = np.arange(max_steps + 1)
    ours, theirs = [], []
    for step in steps:                                           # torch stepped as in training, not the closed form
        theirs.append(optimizer.param_groups[0]["lr"])
        ours.append(learning_rate(cfg, int(step), step / max_steps, lr))
        optimizer.step(), scheduler.step()
    ours, theirs = np.array(ours), np.array(theirs)
    after = steps >= warmup
    diff = ours[after] - theirs[after]
    sample = {"description": "1000 steps at lr 3e-4 down to 1e-5 with a 100-step warm-up; torch's scheduler is stepped as in "
                             "training and the difference is shown from the end of the warm-up (before it ours is lower by design).",
              "ours_png": lines({"ours": (steps, ours)}, "ours: learning rate"),
              "theirs_png": lines({"torch": (steps, theirs)}, "upstream: CosineAnnealingLR"),
              "diff_png": lines({"ours - torch": (steps[after], diff)}, "ours - upstream (after warm-up)"),
              "max_abs_diff": float(np.abs(diff).max()), "tolerance": lr * 1e-6,
              "tolerance_text": "rtol 1e-6 against the stepped scheduler (test_equivalence); rtol 1e-9 against the closed form (test_differential)"}
    hypothesis = hypothesis_outcome("tests/test_differential.py::test_learning_rate_is_torch_s_closed_form_cosine_after_the_warm_up",
                                    d.schedules(), d.learning_rate_pair,
                                    lambda o, t, case: torch.testing.assert_close(torch.tensor(o), torch.tensor(t), **d.LEARNING_RATE_TOLERANCE))
    return sample, hypothesis


def photos() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, (2, 224, 224, 3), dtype=np.uint8)


def sample_dino_blocks(dino) -> tuple[dict, None]:
    from table_image_icl.inputs import DinoTokens, NormalisedPixels
    raw = photos()
    with torch.no_grad():
        ours = dino.norm(DinoTokens("cpu", block=12, model=dino)(raw))
        theirs = dino.forward_features(NormalisedPixels("cpu")(raw))["x_norm_patchtokens"]
    sample = {"description": "Two seeded 224 px noise photos through all 12 blocks of DINOv2 ViT-B/14 (registers) with the final "
                             "norm: the 2 x 256 patch tokens stacked, 768 channels across.",
              **three_heatmaps(ours.flatten(0, 1).numpy(), theirs.flatten(0, 1).numpy(), "patch tokens (512 x 768)"),
              "tolerance": 1e-4, "tolerance_text": "atol 1e-4 (torch.allclose)"}
    return sample, None


def sample_attention(test_model) -> tuple[dict, None]:
    from table_image_icl.model import Attention
    torch.manual_seed(0)
    att = Attention(16, 2).eval()
    g = torch.Generator().manual_seed(0)
    x, keys, values = (torch.randn(1, 6, 16, generator=g) for _ in range(3))
    with torch.no_grad():
        q = att.split(att.q(x))
        assert q.shape == (1, 2, 6, 8)
        ours, theirs = att(x, keys, values), test_model.reference(att, x, keys, values)
    sample = {"description": "Width 16, 2 heads, 6 tokens: the queries, keys and values are (1, 2, 6, 8) after the projections; "
                             "the (6, 16) output of the module next to the hand-written softmax(q k^T / sqrt(8)) v.",
              **three_heatmaps(ours[0].numpy(), theirs[0].numpy(), "attention output (6 x 16)"),
              "tolerance": 1e-5, "tolerance_text": "atol 1e-5 (torch.allclose)"}
    return sample, None


def sample_top_k(test_model) -> tuple[dict, None]:
    """The top-k read with k = every context photo against the plain read, on the test module's tokens and labels."""
    plain, retrieving = test_model.stage(read="all"), test_model.stage(read="top_k", top_k=test_model.N_CTX)
    retrieving.load_state_dict(plain.state_dict())
    with torch.no_grad():
        ours, theirs = retrieving(test_model.tokens(), test_model.Y_READ), plain(test_model.tokens(), test_model.Y_READ)
    sample = {"description": "2 tables of 6 context and 3 test photos, 4 patches of width 16, the same weights on both sides: the "
                             "top-k read with k = 6 (every context photo retrieved) against the plain read; test photos' tokens, "
                             "(2 x 3 x 4) rows of 16.",
              **three_heatmaps(ours[:, test_model.N_CTX:].flatten(0, 2).numpy(), theirs[:, test_model.N_CTX:].flatten(0, 2).numpy(), "read tokens (24 x 16)"),
              "tolerance": 1e-5, "tolerance_text": "atol 1e-5 (torch.allclose)"}
    return sample, None


# ---------- code texts ----------

def our_lines(ours: dict) -> str:
    return repo_lines(ours["path"], ours["start"], ours["end"])


def repo_lines(path: str, start: int, end: int) -> str:
    with open(os.path.join(ROOT, path)) as f:
        return "".join(f.read().splitlines(keepends=True)[start - 1:end])


def reference_body(path: str) -> str:
    with open(os.path.join(ROOT, path)) as f:
        return "".join(f.read().splitlines(keepends=True)[1:])


def upstream_text(p: dict, test_equivalence, test_model) -> tuple[str | None, str]:
    """The upstream code shown on the card and where it comes from."""
    if p["reference"]:
        return reference_body(p["reference"]), f"{p['reference']}, a verbatim copy of the installed upstream lines"
    if p["id"] == "split":
        return inspect.getsource(test_equivalence.multabench_get_split), "multabench_get_split in tests/test_equivalence.py, the benchmark's get_split with its constants inlined"
    if p["id"] == "attention":
        return inspect.getsource(test_model.reference), "reference in tests/test_model.py, the hand-written attention the test compares with"
    if p["id"] == "top_k":
        return repo_lines("table_image_icl/model.py", 212, 217), "table_image_icl/model.py:212-217, the plain read from every context photo in LabelRead.forward"
    return None, "none"


def main() -> None:
    torch.manual_seed(0)
    import test_differential
    import test_equivalence
    import test_model
    dino = None
    if os.path.isdir(HUB):
        from table_image_icl.inputs import DINOV2, DINOV2_WEIGHTS
        dino = torch.hub.load(DINOV2, DINOV2_WEIGHTS).eval()
    cards = []
    for p in PORTS:
        print(p["id"], flush=True)
        if p["id"] == "prepare_table":
            sample, hypothesis = sample_prepare_table(test_differential)
        elif p["id"] == "split":
            sample, hypothesis = sample_split(test_differential)
        elif p["id"] == "learning_rate":
            sample, hypothesis = sample_learning_rate(test_differential)
        elif p["id"] == "dino_blocks":
            if dino is None:
                sample, hypothesis = {"description": "skipped: DINOv2 is not in the torch hub cache", "skipped": True}, None
            else:
                sample, hypothesis = sample_dino_blocks(dino)
        elif p["id"] == "attention":
            sample, hypothesis = sample_attention(test_model)
        elif p["id"] == "top_k":
            sample, hypothesis = sample_top_k(test_model)
        else:
            raise ValueError(p["id"])
        upstream_code, upstream_code_source = upstream_text(p, test_equivalence, test_model)
        upstream = dict(p["upstream"], installed_lines_found=bool(p["upstream"]["local"] and installed_lines(p["upstream"]["local"])))
        cards.append({**p, "upstream": upstream, "sample": sample, "hypothesis": hypothesis, "ours_code": our_lines(p["ours"]),
                      "upstream_code": upstream_code, "upstream_code_source": upstream_code_source})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"hypothesis_examples": HYPOTHESIS_EXAMPLES, "torch": torch.__version__, "ports": cards}, f, indent=1)
    for c in cards:
        h = c["hypothesis"]
        print(f"{c['id']:14s} max_abs_diff {c['sample'].get('max_abs_diff', 'skipped')!s:>12}  tolerance {c['sample'].get('tolerance', '')!s:>10}"
              + (f"  hypothesis {h['examples']} examples, {h['failures']} failures, max {h['max_abs_diff']:.3g}" if h else ""))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
