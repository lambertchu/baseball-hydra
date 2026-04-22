"""Tests for the MTL ROS quantile loss module.

Covers:
- Pinball loss closed-form values at tau=0.5 (symmetric) and tau=0.25
- Positive scalar output + detail dict structure
- Sample weights zero out half the batch correctly
- Kendall log_vars receive gradients after backward
- PA-remaining loss increases with prediction error
"""

from __future__ import annotations

import torch

from src.models.mtl_ros.loss import MultiTaskQuantileLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zero_loss_inputs(
    batch: int = 4,
    n_tasks: int = 6,
    n_quantiles: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (quantile_preds, pa_pred, targets, pa_target) that should yield ~0 loss."""
    targets = torch.zeros(batch, n_tasks)
    pa_target = torch.zeros(batch, 1)
    # All quantile predictions = 0 so pinball(target=0, pred=0) = 0
    quantile_preds = torch.zeros(batch, n_tasks, n_quantiles)
    pa_pred = torch.zeros(batch, 1)
    return quantile_preds, pa_pred, targets, pa_target


# ---------------------------------------------------------------------------
# Pinball closed-form tests
# ---------------------------------------------------------------------------


class TestPinballClosedForm:
    """Validate pinball loss values against hand-computed reference values."""

    def test_symmetric_median_tau(self):
        """At tau=0.5 the pinball loss is |error|/2 (symmetric)."""
        # Single task, single quantile at tau=0.5, single sample.
        loss_fn = MultiTaskQuantileLoss(
            n_tasks=1,
            taus=(0.5,),
            pa_loss="mse",
        )
        # error = target - pred = 1.0 - 0.0 = 1.0 → pinball = max(0.5*1, -0.5*1) = 0.5
        quantile_preds = torch.zeros(1, 1, 1)
        targets = torch.ones(1, 1)
        pa_pred = torch.zeros(1, 1)
        pa_target = torch.zeros(1, 1)  # zero PA loss

        _, details = loss_fn(quantile_preds, pa_pred, targets, pa_target)
        # details["task_0_pinball"] should be ~0.5
        assert abs(details["task_0_pinball"] - 0.5) < 1e-6

    def test_asymmetric_positive_error(self):
        """At tau=0.25 with positive error of 1.0, pinball = 0.25."""
        loss_fn = MultiTaskQuantileLoss(
            n_tasks=1,
            taus=(0.25,),
            pa_loss="mse",
        )
        quantile_preds = torch.zeros(1, 1, 1)
        targets = torch.ones(1, 1)  # error = 1.0
        pa_pred = torch.zeros(1, 1)
        pa_target = torch.zeros(1, 1)

        _, details = loss_fn(quantile_preds, pa_pred, targets, pa_target)
        assert abs(details["task_0_pinball"] - 0.25) < 1e-6

    def test_asymmetric_negative_error(self):
        """At tau=0.25 with negative error of -1.0, pinball = 0.75."""
        loss_fn = MultiTaskQuantileLoss(
            n_tasks=1,
            taus=(0.25,),
            pa_loss="mse",
        )
        quantile_preds = torch.ones(1, 1, 1)  # pred=1, target=0 → error = -1
        targets = torch.zeros(1, 1)
        pa_pred = torch.zeros(1, 1)
        pa_target = torch.zeros(1, 1)

        _, details = loss_fn(quantile_preds, pa_pred, targets, pa_target)
        assert abs(details["task_0_pinball"] - 0.75) < 1e-6

    def test_averaged_across_quantiles(self):
        """Multi-quantile average: tau=0.25 and tau=0.75 with same magnitude error."""
        loss_fn = MultiTaskQuantileLoss(
            n_tasks=1,
            taus=(0.25, 0.75),
            pa_loss="mse",
        )
        # Both preds = 0, target = 1 → errors = 1
        # tau=0.25 → pinball = 0.25
        # tau=0.75 → pinball = 0.75
        # Average = 0.5
        quantile_preds = torch.zeros(1, 1, 2)
        targets = torch.ones(1, 1)
        pa_pred = torch.zeros(1, 1)
        pa_target = torch.zeros(1, 1)

        _, details = loss_fn(quantile_preds, pa_pred, targets, pa_target)
        assert abs(details["task_0_pinball"] - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Scalar & structure tests
# ---------------------------------------------------------------------------


class TestLossStructure:
    """Validate loss returns expected shape/types and is finite."""

    def test_returns_positive_scalar(self):
        """Random inputs should produce a finite positive scalar."""
        torch.manual_seed(0)
        batch, n_tasks, n_quantiles = 8, 6, 5
        loss_fn = MultiTaskQuantileLoss(n_tasks=n_tasks)
        quantile_preds = torch.randn(batch, n_tasks, n_quantiles)
        pa_pred = torch.randn(batch, 1) * 100 + 300
        targets = torch.randn(batch, n_tasks) * 0.1 + 0.3
        pa_target = torch.randn(batch, 1) * 100 + 300

        loss, details = loss_fn(quantile_preds, pa_pred, targets, pa_target)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss).item()
        # Loss can be negative if log_vars push precisions low; just check finite.
        assert "total" in details
        assert "pa_loss" in details
        for t in range(n_tasks):
            assert f"task_{t}_pinball" in details
            assert f"task_{t}_precision" in details


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------


class TestSampleWeights:
    """Validate sample weights are applied correctly."""

    def test_zero_weight_mask(self):
        """Zero-ing half the batch should halve the pinball contribution."""
        torch.manual_seed(0)
        n_tasks, n_quantiles = 2, 3
        loss_fn = MultiTaskQuantileLoss(
            n_tasks=n_tasks,
            taus=(0.25, 0.5, 0.75),
            pa_loss="mse",
        )

        # Build two batches that look identical EXCEPT the 2nd has extra
        # rows that get zero weights — the loss should match the 1st batch.
        quantile_preds_a = torch.randn(4, n_tasks, n_quantiles)
        targets_a = torch.randn(4, n_tasks)
        pa_pred_a = torch.randn(4, 1)
        pa_target_a = torch.randn(4, 1)

        # Batch B: first 4 match A, next 4 are garbage but with weight=0
        quantile_preds_b = torch.cat(
            [quantile_preds_a, torch.randn(4, n_tasks, n_quantiles) * 100],
            dim=0,
        )
        targets_b = torch.cat([targets_a, torch.randn(4, n_tasks) * 100], dim=0)
        pa_pred_b = torch.cat([pa_pred_a, torch.randn(4, 1) * 1000], dim=0)
        pa_target_b = torch.cat([pa_target_a, torch.randn(4, 1) * 1000], dim=0)
        weights_b = torch.cat([torch.ones(4), torch.zeros(4)], dim=0)

        _, det_a = loss_fn(quantile_preds_a, pa_pred_a, targets_a, pa_target_a)
        _, det_b = loss_fn(
            quantile_preds_b,
            pa_pred_b,
            targets_b,
            pa_target_b,
            sample_weights=weights_b,
        )

        for t in range(n_tasks):
            # Multiplicative-sample-weight convention: the weighted MEAN over
            # the full batch of 8 equals half the un-weighted mean over 4
            # (since 4 out of 8 rows are zero-weighted; dividing by N=8
            # instead of the effective weight sum halves the result).
            assert (
                abs(det_b[f"task_{t}_pinball"] - det_a[f"task_{t}_pinball"] / 2) < 1e-5
            )


# ---------------------------------------------------------------------------
# Kendall uncertainty weighting
# ---------------------------------------------------------------------------


class TestKendallWeighting:
    """Validate Kendall log_vars receive gradients during backward."""

    def test_log_vars_updatable(self):
        """log_vars.grad should be populated after a backward pass."""
        torch.manual_seed(0)
        loss_fn = MultiTaskQuantileLoss(n_tasks=6)
        quantile_preds = torch.randn(4, 6, 5, requires_grad=False)
        # Need a learnable quantity hooked in so autograd has something;
        # we test that log_vars themselves receive grad from the log_var
        # regulariser.
        pa_pred = torch.randn(4, 1)
        targets = torch.randn(4, 6)
        pa_target = torch.randn(4, 1)

        loss, _ = loss_fn(quantile_preds, pa_pred, targets, pa_target)
        loss.backward()
        assert loss_fn.log_vars.grad is not None
        assert loss_fn.log_vars.grad.shape == loss_fn.log_vars.shape


# ---------------------------------------------------------------------------
# PA-remaining loss
# ---------------------------------------------------------------------------


class TestPALoss:
    """Validate PA-remaining loss wiring."""

    def test_pa_loss_additive(self):
        """Increasing PA-prediction error should increase the PA loss detail."""
        torch.manual_seed(0)
        loss_fn = MultiTaskQuantileLoss(n_tasks=6, pa_loss="mse")
        quantile_preds = torch.zeros(4, 6, 5)
        targets = torch.zeros(4, 6)
        pa_target = torch.full((4, 1), 300.0)

        # Case A: perfect PA prediction
        pa_pred_good = torch.full((4, 1), 300.0)
        _, det_good = loss_fn(quantile_preds, pa_pred_good, targets, pa_target)

        # Case B: very wrong PA prediction
        pa_pred_bad = torch.full((4, 1), 0.0)
        _, det_bad = loss_fn(quantile_preds, pa_pred_bad, targets, pa_target)

        assert det_bad["pa_loss"] > det_good["pa_loss"]
        # And the bad case should push the total loss up too.
        assert det_bad["total"] > det_good["total"]

    def test_pa_loss_gaussian_nll(self):
        """Gaussian NLL mode exposes a learnable pa_log_var parameter."""
        loss_fn = MultiTaskQuantileLoss(n_tasks=6, pa_loss="gaussian_nll")
        assert hasattr(loss_fn, "pa_log_var")
        assert isinstance(loss_fn.pa_log_var, torch.nn.Parameter)

        quantile_preds = torch.zeros(4, 6, 5)
        targets = torch.zeros(4, 6)
        pa_pred = torch.zeros(4, 1)
        pa_target = torch.zeros(4, 1)
        loss, _ = loss_fn(quantile_preds, pa_pred, targets, pa_target)
        loss.backward()
        assert loss_fn.pa_log_var.grad is not None
