"""Differential tests: Hypothesis draws random inputs and our port is run next to the upstream code on each of them.
The `*_pair` functions return (ours, upstream) so that tour/equivalence.py can measure the same strategies."""
import warnings

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from test_equivalence import multabench_get_split  # tests/ is on sys.path under pytest (and in tour/equivalence.py)

from table_image_icl.model import OUTLIER_SIGMAS, prepare_table
from table_image_icl.multabench import split
from table_image_icl.training import TrainConfig, learning_rate

pytest.importorskip("tabicl")
from sklearn.impute import SimpleImputer  # noqa: E402
from tabicl._sklearn.preprocessing import CustomStandardScaler, OutlierRemover  # noqa: E402

MAX_EXAMPLES = 200
SCALER_EPSILON = 1e-6                        # CustomStandardScaler adds it to the spread; prepare_table too
LEARNING_RATE_TOLERANCE = {"rtol": 1e-9, "atol": 0.0}


# ---------- prepare_table vs CustomStandardScaler + OutlierRemover ----------

@st.composite
def tables(draw, nan_in_test_rows=False):
    """One table (rows 8..60, columns 1..8) of finite floats across scales, with the number of context rows; with
    `nan_in_test_rows`, some cells of the rows after the context are missing."""
    rows, cols = draw(st.integers(8, 60)), draw(st.integers(1, 8))
    scale = draw(st.sampled_from([1e-3, 1.0, 1e3]))
    x = draw(arrays(np.float64, (rows, cols), elements=st.floats(-1e3, 1e3, allow_subnormal=False))) * scale
    n_context = draw(st.integers(2, rows - 1))
    if nan_in_test_rows:
        missing = draw(arrays(np.bool_, (rows - n_context, cols)))
        x[n_context:][missing] = np.nan
    return x, n_context


def tabicl_preprocess(x: np.ndarray, n_context: int) -> np.ndarray:
    """TabICL's chain fitted on the context rows: mean imputation, CustomStandardScaler, OutlierRemover."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imputer = SimpleImputer(keep_empty_features=True).fit(x[:n_context])
        filled = imputer.transform(x)
        scaler = CustomStandardScaler().fit(filled[:n_context])
        remover = OutlierRemover(threshold=OUTLIER_SIGMAS).fit(scaler.transform(filled[:n_context]))
        return remover.transform(scaler.transform(filled))


def prepare_table_pair(x: np.ndarray, n_context: int) -> tuple[np.ndarray, np.ndarray]:
    ours = prepare_table(torch.tensor(x[None], dtype=torch.float64), n_context)[0].numpy()
    return ours, tabicl_preprocess(x, n_context)


def prepare_table_tolerance(x: np.ndarray) -> dict:
    """rtol 1e-6, plus the rounding the two means can differ by (summation order, at most rows x eps x max|x|),
    amplified by the 1e-6 floor on the spread: in a constant column of values around 1e5 both sides give 0 up to
    ~1e-5 of noise. The term is 1e-6 x rows x max|x| x 2 eps and vanishes for ordinary values."""
    rows = x.shape[0]
    largest = float(np.nanmax(np.abs(x))) if np.isfinite(x).any() else 0.0
    return {"rtol": 1e-6, "atol": 1e-6 + 2 * rows * np.finfo(np.float64).eps * largest / SCALER_EPSILON}


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(tables())
def test_prepare_table_matches_tabicl_s_two_transformers_on_random_tables(table):
    ours, theirs = prepare_table_pair(*table)
    torch.testing.assert_close(torch.tensor(ours), torch.tensor(theirs), **prepare_table_tolerance(table[0]))


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(tables(nan_in_test_rows=True))
def test_prepare_table_fills_missing_test_cells_as_tabicl_s_mean_imputer(table):
    ours, theirs = prepare_table_pair(*table)
    assert np.isfinite(ours).all()
    torch.testing.assert_close(torch.tensor(ours), torch.tensor(theirs), **prepare_table_tolerance(table[0]))


# ---------- split vs MulTaBench's get_split ----------

@st.composite
def labelled_rows(draw):
    """A label array for either task, mostly 30..3000 rows with some 12,000-row cases (above both caps), and a split."""
    is_cls = draw(st.booleans())
    n = draw(st.one_of(st.integers(30, 3000), st.integers(30, 3000), st.integers(30, 3000), st.just(12_000)))
    if is_cls:
        n_classes = draw(st.integers(2, 6))
        y = draw(arrays(np.int64, n, elements=st.integers(0, n_classes - 1)))
    else:
        y = draw(arrays(np.float64, n, elements=st.floats(-10, 10, allow_subnormal=False)))
    return y, is_cls, draw(st.integers(0, 5))


def split_pair(y: np.ndarray, is_cls: bool, k: int) -> tuple[tuple, tuple]:
    return split(y, is_cls, k), multabench_get_split(y, is_cls, k)


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(labelled_rows())
def test_split_gives_multabench_s_rows_on_random_labels(case):
    ours, theirs = split_pair(*case)
    assert all(np.array_equal(a, b) for a, b in zip(ours, theirs, strict=True))


# ---------- learning_rate vs CosineAnnealingLR's closed form ----------

@st.composite
def schedules(draw):
    lr = draw(st.floats(1e-6, 1e-2, allow_subnormal=False))
    lr_end = lr * draw(st.floats(0.0, 0.99))
    warmup = draw(st.integers(1, 200))
    return lr, lr_end, warmup, draw(st.integers(warmup + 1, 2000))


def learning_rate_pair(lr: float, lr_end: float, warmup: int, max_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Our rate and torch's closed-form cosine at every step from the end of the warm-up to max_steps."""
    cfg = TrainConfig(out="x", lr=lr, lr_end=lr_end, warmup_steps=warmup, max_steps=max_steps)
    optimizer = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr_end)
    ours, theirs = [], []
    for step in range(warmup, max_steps + 1):
        scheduler.last_epoch = step
        theirs.append(scheduler._get_closed_form_lr()[0])
        ours.append(learning_rate(cfg, step, step / max_steps, lr))
    return np.array(ours), np.array(theirs)


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(schedules())
def test_learning_rate_is_torch_s_closed_form_cosine_after_the_warm_up(schedule):
    ours, theirs = learning_rate_pair(*schedule)
    torch.testing.assert_close(torch.tensor(ours), torch.tensor(theirs), **LEARNING_RATE_TOLERANCE)
