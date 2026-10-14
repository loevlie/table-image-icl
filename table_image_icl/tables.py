"""TabICL v2's table stages (Qu et al., 2025) and what they expect of a table. The weights are the released
checkpoint from Hugging Face (jingang/TabICL, BSD-3), pretrained on about 82M synthetic tables."""
import torch
import torch.nn as nn
from torch import Tensor

REPO = "jingang/TabICL"
CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
WIDTH = 128                      # TabICL's token width; the image side uses the same
N_SLOTS = 10                     # TabICL's class limit: the label slots of a task
TABICL_V2 = {"max_classes": N_SLOTS, "num_quantiles": 999, "embed_dim": WIDTH, "col_num_blocks": 3, "col_nhead": 8, "col_num_inds": 128,
             "col_affine": False, "col_feature_group": "same", "col_feature_group_size": 3, "col_target_aware": True,
             "col_ssmax": "qassmax-mlp-elementwise", "row_num_blocks": 3, "row_nhead": 8, "row_num_cls": 4, "row_rope_base": 100000,
             "row_rope_interleaved": False, "icl_num_blocks": 12, "icl_nhead": 8, "icl_ssmax": "qassmax-mlp-elementwise",
             "ff_factor": 2, "dropout": 0.0, "activation": "gelu", "norm_first": True, "bias_free_ln": False, "recompute": False}
OUTLIER_SIGMAS = 4.0


def load_tabicl(pretrained: bool = True) -> nn.Module:
    """TabICL v2 with its released weights, or the same architecture freshly initialised."""
    from tabicl._model.tabicl import TabICL
    model = TabICL(**TABICL_V2)
    if pretrained:
        from huggingface_hub import hf_hub_download
        state = torch.load(hf_hub_download(REPO, CHECKPOINT), map_location="cpu", weights_only=False)
        assert state["config"] == TABICL_V2, "the checkpoint is not the architecture this module expects"
        model.load_state_dict(state["state_dict"])
    return model


def prepare_table(x: Tensor, n_context: int) -> Tensor:
    """TabICL's preprocessing, fitted on the context rows of each table (batch, rows, columns): standardise, fill
    missing cells with the mean, clip, then soften values beyond OUTLIER_SIGMAS of the cleaned context with a log."""
    context = x[:, :n_context]
    valid = ~torch.isnan(context)
    count = valid.sum(1, keepdim=True).clamp_min(1)
    mean = torch.where(valid, context, 0.0).sum(1, keepdim=True) / count
    std = (torch.where(valid, (context - mean) ** 2, 0.0).sum(1, keepdim=True) / count).sqrt() + 1e-6
    z = ((x - mean) / std).nan_to_num(0.0).clamp(-100, 100)
    lower, upper = outlier_bounds(z[:, :n_context])
    z = torch.maximum(-torch.log1p(z.abs()) + lower, z)
    return torch.minimum(torch.log1p(z.abs()) + upper, z)


def outlier_bounds(context: Tensor) -> tuple[Tensor, Tensor]:
    """Mean and spread of the context without the values beyond OUTLIER_SIGMAS of a first-pass mean and spread,
    as (batch, 1, columns) bounds."""
    def stats(values: Tensor, keep: Tensor) -> tuple[Tensor, Tensor]:
        n = keep.sum(1, keepdim=True)
        m = torch.where(keep, values, 0.0).sum(1, keepdim=True) / n.clamp_min(1)
        var = torch.where(keep, (values - m) ** 2, 0.0).sum(1, keepdim=True) / (n - 1).clamp_min(1)
        return m, var.sqrt().clamp_min(1e-6)

    mean, std = stats(context, torch.ones_like(context, dtype=torch.bool))
    inlier = (context - mean).abs() <= OUTLIER_SIGMAS * std
    mean, std = stats(context, inlier)
    return mean - OUTLIER_SIGMAS * std, mean + OUTLIER_SIGMAS * std


def contiguous_codes(y_context: Tensor, n_slots: int = N_SLOTS) -> tuple[Tensor, Tensor]:
    """TabICL wants classes numbered 0..k-1. Per table, the context labels renumbered in slot order, (batch, n_context),
    and which slot each code stands for, (batch, n_slots) with -1 past the k classes."""
    codes, slots = [], torch.full((len(y_context), n_slots), -1, dtype=torch.long, device=y_context.device)
    for b, y in enumerate(y_context):
        present, inverse = torch.unique(y, return_inverse=True)
        codes.append(inverse)
        slots[b, : len(present)] = present
    return torch.stack(codes), slots
