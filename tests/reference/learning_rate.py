# Reference copy of https://github.com/pytorch/pytorch/blob/v2.11.0/torch/optim/lr_scheduler.py#L1457-L1476 (license: BSD-3-Clause), lines 1457-1476, copied 2026-10-15.
    def _get_closed_form_lr(self) -> list[float | Tensor]:
        r"""Compute learning rates for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` at :attr:`last_epoch` using
        a closed-form formula.

        Uses :attr:`base_lrs` to compute learning rates. This method is called
        when an epoch is passed to :meth:`step`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.
        """
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
            for base_lr in self.base_lrs
        ]
