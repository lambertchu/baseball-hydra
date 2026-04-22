"""Multi-task quantile loss with Kendall uncertainty weighting.

The preseason MTL uses MSE / Huber to produce single-point estimates per
target.  The ROS variant needs a *distributional* forecast — one prediction
per quantile tau — plus an auxiliary PA-remaining regression head.

This module implements:

1. **Pinball (quantile) loss** per (task, quantile):

       pinball(tau, err) = max(tau * err, (tau - 1) * err)

   averaged over the batch (optionally weighted by per-sample weights) then
   over quantiles to produce one scalar per task.

2. **Kendall homoscedastic uncertainty weighting** over the six per-PA-rate
   tasks (identical to ``src/models/mtl/loss.py``'s approach):

       total_ros = Σ_t [ 0.5 * exp(-log_var_t) * pinball_t + 0.5 * log_var_t ]

   where ``log_var_t`` is a learnable per-task parameter.

3. **PA-remaining auxiliary loss** — either MSE (default) or Gaussian NLL
   with its own learned ``pa_log_var`` parameter.  Weighted by a config knob
   ``pa_weight`` and added to ``total_ros``.

The PA-remaining log-variance is kept separate from the per-task log_vars
so the Kendall weighting over the 6 rate/count targets remains
interpretable on its own.
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn as nn


class MultiTaskQuantileLoss(nn.Module):
    """Pinball loss + Kendall uncertainty weighting + PA-remaining head.

    Parameters
    ----------
    n_tasks:
        Number of rate targets (default 6: OBP, SLG, HR/PA, R/PA, RBI/PA, SB/PA).
    taus:
        Quantile levels the network predicts.  Default is a 5-point
        distribution ``(0.05, 0.25, 0.50, 0.75, 0.95)``.  The loss is
        averaged uniformly across these quantiles per task.
    pa_loss:
        ``"mse"`` (default) or ``"gaussian_nll"``.  ``"mse"`` is simple
        squared error; ``"gaussian_nll"`` adds a learnable ``pa_log_var``
        parameter and uses the same Kendall-style weighting as the rate
        targets (variance-aware NLL).
    pa_weight:
        Multiplicative weight on the PA-remaining loss before adding to
        the total.  Default 1.0.
    """

    def __init__(
        self,
        n_tasks: int = 6,
        taus: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
        pa_loss: Literal["mse", "gaussian_nll"] = "mse",
        pa_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_tasks = n_tasks
        self.pa_loss = pa_loss
        self.pa_weight = float(pa_weight)

        # Store taus as a buffer so it moves with .to(device) and serialises
        # with the state_dict, but is not a learnable parameter.
        self.register_buffer("taus", torch.tensor(list(taus), dtype=torch.float32))

        # Learnable per-task log-variance (Kendall weighting).
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

        # Optional PA log-variance — only created when pa_loss == "gaussian_nll",
        # so ``"mse"`` mode keeps the PA head simple and interpretable (loss
        # in PA² units).
        if pa_loss == "gaussian_nll":
            self.pa_log_var = nn.Parameter(torch.zeros(1))
        else:
            self.pa_log_var = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        quantile_preds: torch.Tensor,
        pa_pred: torch.Tensor,
        targets: torch.Tensor,
        pa_target: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the combined multi-task quantile loss.

        Parameters
        ----------
        quantile_preds:
            Tensor of shape ``(batch, n_tasks, n_quantiles)`` — the
            network's quantile output.
        pa_pred:
            Tensor of shape ``(batch, 1)`` — the PA-remaining head's
            scalar prediction (point estimate, NOT a quantile).
        targets:
            Tensor of shape ``(batch, n_tasks)`` — ROS rate targets in
            z-scored (or raw) scale; matches the scale of ``quantile_preds``.
        pa_target:
            Tensor of shape ``(batch, 1)`` — ROS PA targets in raw PA units.
        sample_weights:
            Optional per-sample weights of shape ``(batch,)``.  When
            provided, per-sample pinball losses are multiplied by the
            weights before averaging.  The simple multiplicative-mean
            convention matches ``HuberMultiTaskLoss`` in the preseason MTL.

        Returns
        -------
        (total_loss, details)
            ``total_loss`` is a scalar tensor suitable for backward.
            ``details`` contains per-task pinball, per-task precision,
            PA loss, and the total — as Python floats for logging.
        """
        batch, n_tasks, n_quantiles = quantile_preds.shape
        if n_tasks != self.n_tasks:
            raise ValueError(
                f"quantile_preds.shape[1] ({n_tasks}) != n_tasks ({self.n_tasks})"
            )
        if n_quantiles != self.taus.numel():
            raise ValueError(
                f"quantile_preds.shape[2] ({n_quantiles}) != taus.numel "
                f"({self.taus.numel()})"
            )

        details: dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=targets.device)

        # --- Per-task pinball with Kendall weighting ---------------------
        # Broadcast targets from (B, T) → (B, T, 1) for pinball computation.
        # Errors: (B, T, Q)
        errors = targets.unsqueeze(-1) - quantile_preds
        # taus broadcasting: (Q,) → (1, 1, Q)
        tau_b = self.taus.view(1, 1, -1)
        pinball = torch.maximum(tau_b * errors, (tau_b - 1.0) * errors)  # (B, T, Q)

        if sample_weights is not None:
            # Multiplicative per-sample weighting before the batch mean.
            w = sample_weights.view(-1, 1, 1)
            pinball = pinball * w

        # Per-task pinball: mean over (batch, quantiles)
        pinball_per_task = pinball.mean(dim=(0, 2))  # (n_tasks,)

        for t in range(self.n_tasks):
            precision_t = torch.exp(-self.log_vars[t])
            task_loss = 0.5 * precision_t * pinball_per_task[t] + 0.5 * self.log_vars[t]
            total_loss = total_loss + task_loss
            details[f"task_{t}_pinball"] = pinball_per_task[t].item()
            details[f"task_{t}_precision"] = precision_t.item()

        # --- PA-remaining loss ------------------------------------------
        pa_sq_err = (pa_pred - pa_target) ** 2  # (B, 1)
        if sample_weights is not None:
            pa_sq_err = pa_sq_err * sample_weights.view(-1, 1)
        pa_mse = pa_sq_err.mean()

        if self.pa_loss == "gaussian_nll":
            pa_precision = torch.exp(-self.pa_log_var)
            pa_loss_val = (
                0.5 * pa_precision.squeeze() * pa_mse + 0.5 * self.pa_log_var.squeeze()
            )
            details["pa_precision"] = pa_precision.item()
        else:
            pa_loss_val = pa_mse

        total_loss = total_loss + self.pa_weight * pa_loss_val
        details["pa_loss"] = pa_loss_val.item()
        details["total"] = total_loss.item()

        return total_loss, details

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_task_precisions(self) -> torch.Tensor:
        """Return current precision weights ``exp(-log_var)`` per task."""
        with torch.no_grad():
            return torch.exp(-self.log_vars).cpu()
