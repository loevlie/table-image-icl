"""Scoring a trained model on held-out generated tasks: by what the photo stands for (a feature or the label) and by
how the task is shown (the view), with the share of the floor-to-ceiling gap the photo closes."""
import numpy as np
import torch
from torch import nn

from table_image_icl.training import forward_loss


@torch.no_grad()
def score_by_role(model: nn.Module, tasks, batches: list, device: str, amp: bool = True) -> dict:
    """{role: {"views": {view: {"loss", "accuracy", "tasks"}}, "image gain", "gap_closed"}}. The image gain is the
    per-batch difference in loss between no image and image, as a mean, a standard error and the share of batches
    where the photo helped; every view sees the same tasks, so the batches are paired."""
    model.eval()
    per_batch = {}
    for batch in batches:
        for view in tasks.VIEWS:
            loss, accuracy = forward_loss(model, tasks, batch, view, device, amp)
            per_batch.setdefault(batch.image_role, {}).setdefault(view, []).append((loss.item(), accuracy, len(batch.x)))
    report = {}
    for role, views in per_batch.items():
        summary = {view: {"loss": float(np.mean([loss for loss, _, _ in r])), "accuracy": float(np.mean([a for _, a, _ in r])),
                          "tasks": int(sum(n for _, _, n in r))} for view, r in views.items()}
        gain = np.array([f[0] - i[0] for f, i in zip(views["no image"], views["image"])])
        report[role] = {"views": summary, "gap_closed": gap_closed(summary),
                        "image gain": {"loss_mean": float(gain.mean()), "loss_se": float(gain.std(ddof=1) / np.sqrt(len(gain))) if len(gain) > 1 else float("nan"),
                                       "share_of_batches_helped": float((gain > 0).mean()), "batches": len(gain)}}
    return report


def gap_closed(views: dict) -> dict:
    """How much of the gap between the table alone (floor) and the hidden column given back (ceiling) the photo
    closes, in loss and in accuracy: 0 = the photo adds nothing, 1 = as good as seeing the hidden variable."""
    floor, image, ceiling = views["no image"], views["image"], views["column visible"]

    def share(numerator, denominator):
        return numerator / denominator if denominator else float("nan")

    return {"loss": share(floor["loss"] - image["loss"], floor["loss"] - ceiling["loss"]),
            "accuracy": share(image["accuracy"] - floor["accuracy"], ceiling["accuracy"] - floor["accuracy"])}
