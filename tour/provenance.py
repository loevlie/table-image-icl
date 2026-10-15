"""Where our code is a port of someone else's: one entry per ported piece, naming our lines, the upstream lines at the
commit we pin (and the same file as installed here, when it is), the verbatim reference copy under tests/reference/,
the tests that show the two agree, and what was changed on purpose. tour/equivalence.py runs both sides on one
sample input and writes tour/data/equivalence.json for the equivalence cards."""
import importlib.util
import os

TABICL = "https://github.com/soda-inria/tabicl/blob/0dbff3e/tabicl/"
DINO = "https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/"
MULTA = "https://github.com/alanarazi7/MulTaBench/blob/eebb15e/"
TORCH = "https://github.com/pytorch/pytorch/blob/v2.6.0/torch/"
HUB = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8")
LICENSES = {"tabicl": "BSD-3-Clause", "dinov2": "Apache-2.0", "torch": "BSD-3-Clause", "multabench": "MIT"}

PORTS = [
    {
        "id": "prepare_table",
        "title": "TabICL's preprocessing: z-score, clip, soften outliers",
        "ours": {"path": "table_image_icl/model.py", "start": 53, "end": 80},
        "upstream": {"label": "tabicl/_sklearn/preprocessing.py: OutlierRemover and CustomStandardScaler",
                     "url": TABICL + "_sklearn/preprocessing.py#L275-L483",
                     "local": ("tabicl", "_sklearn/preprocessing.py", 275, 483)},
        "reference": "tests/reference/prepare_table.py",
        "tests": ["tests/test_equivalence.py::test_prepare_table_is_tabicl_s_standard_scaler_then_its_outlier_remover_fitted_on_the_context",
                  "tests/test_differential.py::test_prepare_table_matches_tabicl_s_two_transformers_on_random_tables",
                  "tests/test_differential.py::test_prepare_table_fills_missing_test_cells_as_tabicl_s_mean_imputer",
                  "tests/test_model.py::test_prepare_table_standardises_on_the_context_fills_missing_cells_and_softens_outliers"],
        "changes": "Batched over tables in torch and fitted on the context rows of each table; the two sklearn classes "
                   "become one function with the bounds in outlier_bounds. Missing cells are filled with the context mean "
                   "(TabICL's SimpleImputer); when the context itself has missing cells, our mean and spread are taken over "
                   "the present cells, whereas TabICL imputes first and divides by every row.",
    },
    {
        "id": "split",
        "title": "MulTaBench's get_split: repeated 3-fold, capped at 10k / 5k",
        "ours": {"path": "table_image_icl/multabench.py", "start": 111, "end": 129},
        "upstream": {"label": "multabench/benchmark/splits.py: get_split (vendored in tests/test_equivalence.py)",
                     "url": MULTA + "multabench/benchmark/splits.py",
                     "local": None},
        "reference": None,
        "tests": ["tests/test_equivalence.py::test_split_gives_the_benchmark_s_rows_for_every_split_and_both_sizes",
                  "tests/test_equivalence.py::test_split_falls_back_to_plain_folds_when_a_class_has_fewer_than_three_rows",
                  "tests/test_differential.py::test_split_gives_multabench_s_rows_on_random_labels"],
        "changes": "The benchmark's constants are inlined (3 folds, 2 repeats, seed 42, caps 10,000 and 5,000) and an optional "
                   "`groups` argument adds a stratified group fold the benchmark does not have, so that a patient's rows stay on "
                   "one side; without groups the rows are the same.",
    },
    {
        "id": "learning_rate",
        "title": "Cosine annealing (torch's CosineAnnealingLR) with a linear warm-up",
        "ours": {"path": "table_image_icl/training.py", "start": 63, "end": 66},
        "upstream": {"label": "torch/optim/lr_scheduler.py: CosineAnnealingLR._get_closed_form_lr",
                     "url": TORCH + "optim/lr_scheduler.py#L1131-L1138",
                     "local": ("torch", "optim/lr_scheduler.py", 1457, 1476),
                     "local_note": "the URL pins torch v2.6.0, the version the reported runs used; the copy under tests/reference/ is "
                                   "the installed torch (2.11.0), where the same method carries a docstring"},
        "reference": "tests/reference/learning_rate.py",
        "tests": ["tests/test_equivalence.py::test_learning_rate_is_cosine_annealing_after_the_warm_up",
                  "tests/test_differential.py::test_learning_rate_is_torch_s_closed_form_cosine_after_the_warm_up",
                  "tests/test_training.py::test_learning_rate_warms_up_then_decays_and_is_a_factor_of_each_groups_base_rate"],
        "changes": "Takes the progress through the budget (step / max_steps for a step budget, elapsed / hours for a time "
                   "budget) instead of the step count, multiplies by a linear warm-up over the first warmup_steps, and takes "
                   "each parameter group's base rate as an argument (the table stages at lr, the image side at image_lr).",
    },
    {
        "id": "dino_blocks",
        "title": "DINOv2's forward_features, stopped after `block` blocks",
        "ours": {"path": "table_image_icl/inputs.py", "start": 43, "end": 66},
        "upstream": {"label": "dinov2/models/vision_transformer.py: prepare_tokens_with_masks and forward_features",
                     "url": DINO + "dinov2/models/vision_transformer.py#L216-L273",
                     "local": ("dinov2", "dinov2/models/vision_transformer.py", 216, 273)},
        "reference": "tests/reference/dino_blocks.py",
        "tests": ["tests/test_equivalence.py::test_running_every_block_ourselves_gives_dinov2_s_own_patch_features",
                  "tests/test_inputs.py::test_real_dino_block_4_gives_256_patch_tokens"],
        "changes": "Only the first `block` blocks are run and the final norm is not applied; the CLS and register tokens are "
                   "dropped. With summary=True (encoder cls) every block and the norm run and the CLS and register tokens are "
                   "kept instead. Photos are processed in chunks under no_grad and, on a GPU, bfloat16 autocast.",
    },
    {
        "id": "attention",
        "title": "Multi-head attention on torch's scaled_dot_product_attention",
        "ours": {"path": "table_image_icl/model.py", "start": 97, "end": 116},
        "upstream": {"label": "torch.nn.functional.scaled_dot_product_attention (docs); the test's hand-written softmax(q k^T / sqrt(d)) v",
                     "url": "https://pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html",
                     "local": None},
        "reference": None,
        "tests": ["tests/test_model.py::test_attention_equals_the_hand_written_reference",
                  "tests/test_model.py::test_queries_do_not_interact"],
        "changes": "Separate inputs for queries, keys and values with their own projections, so one module serves self-attention "
                   "in the transformer layers and the cross-attention of the labelled read; no mask and no dropout.",
    },
    {
        "id": "top_k",
        "title": "The top-k read: each photo reads from its k nearest context photos only",
        "ours": {"path": "table_image_icl/model.py", "start": 231, "end": 240},
        "upstream": {"label": "none: the reference is the plain read from every context photo, the branch above it in LabelRead.forward",
                     "url": None, "local": None},
        "reference": None,
        "tests": ["tests/test_model.py::test_top_k_with_every_context_photo_is_the_plain_read",
                  "tests/test_model.py::test_top_k_reads_only_from_the_nearest_context_photos",
                  "tests/test_model.py::test_the_top_k_read_is_the_same_whether_photos_come_in_chunks_or_one_by_one"],
        "changes": "The keys and label values of each photo's k neighbours are gathered and the photos go through the attention "
                   "in chunks, so the attention a photo faces has k x patches keys whatever the context size; with k equal to "
                   "the number of context photos it is the plain read.",
    },
]


def package_root(name: str) -> str | None:
    """Where the installed copy of `name` lives: the torch hub cache for dinov2, site-packages otherwise; None if absent."""
    if name == "dinov2":
        return HUB if os.path.isdir(HUB) else None
    spec = importlib.util.find_spec(name)
    return os.path.dirname(spec.origin) if spec and spec.origin else None


def installed_file(local: tuple) -> str | None:
    """The absolute path of an upstream file described by a `local` tuple, or None when it is not installed here."""
    root = package_root(local[0])
    path = root and os.path.join(root, local[1])
    return path if path and os.path.isfile(path) else None


def installed_lines(local: tuple) -> list[str] | None:
    """Lines start..end (1-based, inclusive) of the installed upstream file, or None when it is not installed here."""
    path = installed_file(local)
    if path is None:
        return None
    with open(path) as f:
        return f.read().splitlines(keepends=True)[local[2] - 1:local[3]]


def port(port_id: str) -> dict:
    return next(p for p in PORTS if p["id"] == port_id)
