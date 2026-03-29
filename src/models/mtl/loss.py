"""Multi-task loss with homoscedastic uncertainty weighting.

Implements the loss function from Kendall et al. (2018) that learns
per-task uncertainty parameters to automatically balance rate stats
(OBP ~0.3, SLG ~0.4) vs count stats (HR ~20, R ~65, RBI ~60, SB ~8).

Each task t has a learned log-variance parameter log_var_t, and the
weighted loss is:

    L = Σ_t [ 0.5 * exp(-log_var_t) * MSE_t + 0.5 * log_var_t ]

The precision term exp(-log_var_t) downweights noisy (high-variance)
tasks, while the log_var_t regulariser prevents all precisions from
going to zero.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """Homoscedastic uncertainty weighting for multi-task learning.

    Parameters
    ----------
    n_tasks:
        Number of prediction tasks (default 6 for the stat targets).
    """

    def __init__(self, n_tasks: int = 6) -> None:
        super().__init__()
        self.n_tasks = n_tasks
        # Initialise log(σ²) = 0  →  σ² = 1  →  precision = 1 (equal weighting)
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the uncertainty-weighted multi-task loss.

        Parameters
        ----------
        predictions:
            List of *n_tasks* tensors, each shaped ``(batch, 1)``.
        targets:
            Ground-truth tensor shaped ``(batch, n_tasks)``.
        sample_weights:
            Optional per-sample weights shaped ``(batch,)``.  When
            provided, per-sample losses are weighted before averaging.

        Returns
        -------
        tuple[torch.Tensor, dict[str, float]]
            ``(total_loss, details)`` where *details* contains per-task
            MSE losses, precision weights, and the total scalar.
        """
        total_loss = torch.tensor(0.0, device=targets.device)
        details: dict[str, float] = {}

        for t in range(self.n_tasks):
            pred_t = predictions[t].squeeze(-1)  # (batch,)
            target_t = targets[:, t]              # (batch,)

            if sample_weights is not None:
                mse_t = (sample_weights * (pred_t - target_t) ** 2).mean()
            else:
                mse_t = nn.functional.mse_loss(pred_t, target_t)
            precision_t = torch.exp(-self.log_vars[t])
            task_loss = 0.5 * precision_t * mse_t + 0.5 * self.log_vars[t]

            total_loss = total_loss + task_loss

            details[f"task_{t}_mse"] = mse_t.item()
            details[f"task_{t}_weight"] = precision_t.item()

        details["total"] = total_loss.item()
        return total_loss, details

    def get_task_weights(self) -> np.ndarray:
        """Return current precision weights exp(-log_var) per task.

        Higher values indicate the network treats the task as more
        predictable (lower uncertainty).
        """
        with torch.no_grad():
            return torch.exp(-self.log_vars).cpu().numpy()


class HuberMultiTaskLoss(nn.Module):
    """Homoscedastic uncertainty weighting with Huber loss per task.

    Identical to :class:`MultiTaskLoss` except that it uses Huber loss
    (smooth L1) instead of MSE.  On z-scored targets, ``delta=2.0``
    means errors beyond 2 standard deviations receive linear (not
    quadratic) penalty, reducing the influence of outlier seasons.

    Parameters
    ----------
    n_tasks:
        Number of prediction tasks (default 6).
    delta:
        Huber transition threshold on z-scored targets.
    """

    def __init__(self, n_tasks: int = 6, delta: float = 2.0) -> None:
        super().__init__()
        self.n_tasks = n_tasks
        self.delta = delta
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute uncertainty-weighted Huber loss.

        Parameters
        ----------
        predictions:
            List of *n_tasks* tensors, each shaped ``(batch, 1)``.
        targets:
            Ground-truth tensor shaped ``(batch, n_tasks)``.
        sample_weights:
            Optional per-sample weights shaped ``(batch,)``.  When
            provided, per-sample losses are weighted before averaging.

        Returns
        -------
        tuple[torch.Tensor, dict[str, float]]
            ``(total_loss, details)`` where *details* contains per-task
            losses, precision weights, and the total scalar.
        """
        total_loss = torch.tensor(0.0, device=targets.device)
        details: dict[str, float] = {}

        for t in range(self.n_tasks):
            pred_t = predictions[t].squeeze(-1)  # (batch,)
            target_t = targets[:, t]              # (batch,)

            if sample_weights is not None:
                per_sample = nn.functional.huber_loss(
                    pred_t, target_t, delta=self.delta, reduction="none",
                )
                huber_t = (sample_weights * per_sample).mean()
            else:
                huber_t = nn.functional.huber_loss(
                    pred_t, target_t, delta=self.delta, reduction="mean",
                )
            precision_t = torch.exp(-self.log_vars[t])
            task_loss = 0.5 * precision_t * huber_t + 0.5 * self.log_vars[t]

            total_loss = total_loss + task_loss

            details[f"task_{t}_loss"] = huber_t.item()
            details[f"task_{t}_weight"] = precision_t.item()

        details["total"] = total_loss.item()
        return total_loss, details

    def get_task_weights(self) -> np.ndarray:
        """Return current precision weights exp(-log_var) per task."""
        with torch.no_grad():
            return torch.exp(-self.log_vars).cpu().numpy()
