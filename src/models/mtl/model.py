"""MTL Forecaster for MLB batter stat predictions.

Multi-Task Learning neural network with a shared backbone and
per-target prediction heads.  A wrapper class (``MTLForecaster``)
provides the same Forecaster interface as Ridge and XGBoost
(fit / predict / save / load).

Usage
-----
    from src.models.mtl.model import MTLForecaster

    model = MTLForecaster(config)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    predictions = model.predict(X_test)
    model.save("data/models/mtl/")
    loaded = MTLForecaster.load("data/models/mtl/")
"""

from __future__ import annotations

import json
import logging
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.eval.metrics import rmse
from src.models.mtl.dataset import BatterDataset
from src.models.mtl.loss import HuberMultiTaskLoss, MultiTaskLoss
from src.models.utils import to_float64_array as _to_float64_array

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    """Linear layer with an optional residual skip connection."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual

        if use_residual:
            self.proj: nn.Module = (
                nn.Identity()
                if in_dim == out_dim
                else nn.Linear(in_dim, out_dim, bias=False)
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self.linear(x)))
        if self.use_residual:
            out = out + self.proj(x)
        return out


class GatedTaskHead(nn.Module):
    """Task head with a sigmoid gate over shared features."""

    def __init__(self, in_dim: int, head_dim: int = 32) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(in_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x * self.gate(x))


# ---------------------------------------------------------------------------
# Pure PyTorch model
# ---------------------------------------------------------------------------


class MTLNetwork(nn.Module):
    """Shared backbone with per-target prediction heads.

    Architecture (default, all sizes configurable)::

        Input -> BatchNorm1d
              -> Linear(N, 256) -> ReLU -> Dropout(0.3)
              -> Linear(256, 128) -> ReLU -> Dropout(0.2)
              -> Linear(128, 64) -> ReLU -> Dropout(0.1)
              +- Head 0: Linear(64, 32) -> ReLU -> Linear(32, 1)
              +- ...
              +- Head 5: Linear(64, 32) -> ReLU -> Linear(32, 1)

    Parameters
    ----------
    n_features:
        Number of input feature columns.
    n_targets:
        Number of prediction targets (6 by default).
    hidden_dims:
        Widths of the shared backbone layers.
    head_dim:
        Hidden width inside each task head.
    dropouts:
        Dropout rate after each backbone layer.  Must have the same
        length as *hidden_dims*.
    use_residual:
        Add skip connections in the shared backbone.
    use_gated_heads:
        Use soft-attention gating in each task head.
    two_stage:
        When True, rate heads (indices 0, 1) run first and their
        detached outputs are concatenated with the backbone
        representation before feeding into count heads (indices 2-5).
    """

    def __init__(
        self,
        n_features: int,
        n_targets: int = 6,
        hidden_dims: list[int] | None = None,
        head_dim: int = 32,
        dropouts: list[float] | None = None,
        use_residual: bool = False,
        use_gated_heads: bool = False,
        two_stage: bool = False,
        speed_head_indices: list[int] | None = None,
        speed_heads_receive_rates: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        if dropouts is None:
            dropouts = [0.3, 0.2, 0.1]
        if len(hidden_dims) != len(dropouts):
            raise ValueError(
                f"hidden_dims ({len(hidden_dims)}) and dropouts "
                f"({len(dropouts)}) must have the same length"
            )

        self.two_stage = two_stage
        self.speed_head_indices = speed_head_indices or []
        self.speed_heads_receive_rates = speed_heads_receive_rates
        self.batch_norm = nn.BatchNorm1d(n_features)

        # Shared backbone
        backbone_layers: list[nn.Module] = []
        in_dim = n_features
        for h_dim, drop in zip(hidden_dims, dropouts):
            backbone_layers.append(
                ResidualBlock(in_dim, h_dim, dropout=drop, use_residual=use_residual)
            )
            in_dim = h_dim
        self.backbone = nn.Sequential(*backbone_layers)

        # Per-target heads
        self.heads = nn.ModuleList()
        for i in range(n_targets):
            if (
                two_stage
                and i >= 2
                and (i not in self.speed_head_indices or self.speed_heads_receive_rates)
            ):
                head_in = in_dim + 2
            else:
                head_in = in_dim
            if use_gated_heads:
                self.heads.append(GatedTaskHead(head_in, head_dim))
            else:
                self.heads.append(
                    nn.Sequential(
                        nn.Linear(head_in, head_dim),
                        nn.ReLU(),
                        nn.Linear(head_dim, 1),
                    )
                )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x:
            Input features, shape ``(batch, n_features)``.

        Returns
        -------
        list[torch.Tensor]
            One ``(batch, 1)`` tensor per target.
        """
        x = self.batch_norm(x)
        shared = self.backbone(x)

        if not self.two_stage:
            return [head(shared) for head in self.heads]

        # Stage 1: rate heads (OBP=0, SLG=1)
        outputs: list[torch.Tensor | None] = [None] * len(self.heads)
        rate_preds = []
        for i in (0, 1):
            out = self.heads[i](shared)
            outputs[i] = out
            rate_preds.append(out.detach())  # detach prevents gradient flow

        # Stage 2: count heads with concatenated rate predictions
        count_input = torch.cat([shared, *rate_preds], dim=1)  # (batch, in_dim + 2)
        for i in range(2, len(self.heads)):
            if i in self.speed_head_indices and not self.speed_heads_receive_rates:
                outputs[i] = self.heads[i](shared)
            else:
                outputs[i] = self.heads[i](count_input)

        return outputs  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Scaler serialisation helper
# ---------------------------------------------------------------------------


def _reconstruct_scaler(
    mean: torch.Tensor,
    scale: torch.Tensor,
    var: torch.Tensor,
    n_samples_seen: torch.Tensor | int,
) -> StandardScaler:
    """Rebuild a fitted StandardScaler from saved tensor parameters."""
    scaler = StandardScaler()
    scaler.mean_ = mean.numpy()
    scaler.scale_ = scale.numpy()
    scaler.var_ = var.numpy()
    if isinstance(n_samples_seen, torch.Tensor):
        n_samples_seen = n_samples_seen.numpy()
    scaler.n_samples_seen_ = n_samples_seen
    scaler.n_features_in_ = len(mean)
    return scaler


# ---------------------------------------------------------------------------
# Legacy checkpoint migration
# ---------------------------------------------------------------------------


def _migrate_legacy_backbone_keys(state_dict: dict) -> dict:
    """Remap old nn.Sequential backbone keys to ResidualBlock format.

    Old format stored Linear layers at every 3rd index in a flat
    nn.Sequential (Linear, ReLU, Dropout), producing keys like
    ``backbone.0.weight``, ``backbone.3.weight``, ``backbone.6.weight``.

    The current ResidualBlock wrapper produces keys like
    ``backbone.0.linear.weight``, ``backbone.1.linear.weight``, etc.
    """
    has_new = any(k.startswith("backbone.") and ".linear." in k for k in state_dict)
    if has_new:
        return state_dict

    # Collect old-format backbone Linear params
    old_linear: dict[int, dict[str, torch.Tensor]] = {}
    other: dict[str, torch.Tensor] = {}

    for key, val in state_dict.items():
        parts = key.split(".")
        if (
            parts[0] == "backbone"
            and len(parts) == 3
            and parts[2] in ("weight", "bias")
        ):
            old_linear.setdefault(int(parts[1]), {})[parts[2]] = val
        else:
            other[key] = val

    if not old_linear:
        return state_dict

    new_sd = dict(other)
    for new_idx, old_idx in enumerate(sorted(old_linear)):
        for param, val in old_linear[old_idx].items():
            new_sd[f"backbone.{new_idx}.linear.{param}"] = val

    return new_sd


# ---------------------------------------------------------------------------
# Forecaster wrapper (sklearn-like API)
# ---------------------------------------------------------------------------


class MTLForecaster:
    """MTL Forecaster with fit / predict / save / load interface.

    Wraps :class:`MTLNetwork` and handles feature/target scaling,
    the training loop (AdamW + LR scheduling + early stopping),
    and model persistence via ``torch.save`` / ``torch.load``.

    Parameters
    ----------
    config:
        Model config dict (typically from ``configs/mtl.yaml``).
        See the plan document for expected keys and defaults.
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = {}
        model_cfg = config.get("model", {})

        # Architecture
        self.hidden_dims: list[int] = list(model_cfg.get("hidden_dims", [256, 128, 64]))
        self.head_dim: int = model_cfg.get("head_dim", 32)
        self.dropouts: list[float] = list(model_cfg.get("dropouts", [0.3, 0.2, 0.1]))
        self.use_residual: bool = model_cfg.get("use_residual", False)
        self.use_gated_heads: bool = model_cfg.get("use_gated_heads", False)
        self.two_stage: bool = model_cfg.get("two_stage", False)
        self.speed_head_indices: list[int] = list(
            model_cfg.get("speed_head_indices", [])
        )
        self.speed_heads_receive_rates: bool = model_cfg.get(
            "speed_heads_receive_rates", False
        )

        # Training
        self.batch_size: int = model_cfg.get("batch_size", 64)
        self.max_epochs: int = model_cfg.get("epochs", 200)
        self.learning_rate: float = model_cfg.get("learning_rate", 1e-3)
        self.weight_decay: float = model_cfg.get("weight_decay", 1e-4)
        self.early_stopping_patience: int = model_cfg.get("early_stopping_patience", 20)
        self.seed: int = config.get("seed", 42)

        # Mixup augmentation
        self.mixup_alpha: float = model_cfg.get("mixup_alpha", 0.0)

        # Sample recency weighting
        self.recency_decay_lambda: float = model_cfg.get("recency_decay_lambda", 0.0)

        # LR scheduler
        lr_cfg = model_cfg.get("lr_scheduler", {})
        self.lr_scheduler_type: str = lr_cfg.get("type", "plateau")
        self.lr_scheduler_patience: int = lr_cfg.get("patience", 10)
        self.lr_scheduler_factor: float = lr_cfg.get("factor", 0.5)
        self.lr_scheduler_min_lr: float = lr_cfg.get("min_lr", 1e-6)
        self.lr_cosine_t0: int = lr_cfg.get("T_0", 20)
        self.lr_cosine_t_mult: int = lr_cfg.get("T_mult", 2)
        self.lr_cosine_eta_min: float = lr_cfg.get("eta_min", 1e-6)

        # Loss function
        loss_cfg = config.get("loss", {})
        self.loss_type: str = loss_cfg.get("type", "mse")
        self.loss_delta: float = loss_cfg.get("delta", 2.0)
        self.target_winsorize_pct: float = loss_cfg.get("target_winsorize_pct", 0.0)

        # Stochastic Weight Averaging
        swa_cfg = model_cfg.get("swa", {})
        self.swa_enabled: bool = swa_cfg.get("enabled", False)
        self.swa_lr: float = swa_cfg.get("lr", 1e-4)
        self.swa_epochs: int = swa_cfg.get("epochs", 10)
        self.swa_anneal_epochs: int = swa_cfg.get("anneal_epochs", 5)

        # State (populated during fit)
        self.network_: MTLNetwork | None = None
        self.loss_fn_: MultiTaskLoss | None = None
        self.feature_scaler_: StandardScaler | None = None
        self.target_scaler_: StandardScaler | None = None
        self.feature_names_: list[str] = []
        self.target_names_: list[str] = []
        self.is_fitted_: bool = False
        self.training_history_: list[dict] = []
        # Per-feature min/max of scaled training data, used to clamp
        # out-of-distribution inputs at prediction time (e.g. bat speed
        # features that are constant in 2016-2022 but real in 2024+).
        self.feature_min_: np.ndarray | None = None
        self.feature_max_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Seed helpers
    # ------------------------------------------------------------------

    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _scale_and_clamp(self, X_arr: np.ndarray) -> np.ndarray:
        """Scale features and clamp to training range.

        Features like bat speed are constant (zero variance) in the
        training data (2016-2022) but have real values in test (2024+).
        Without clamping, the network receives inputs far outside the
        distribution it was trained on, producing wild predictions.
        """
        X_scaled = self.feature_scaler_.transform(X_arr)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        if self.feature_min_ is not None and self.feature_max_ is not None:
            np.clip(X_scaled, self.feature_min_, self.feature_max_, out=X_scaled)
        return X_scaled

    def _make_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> ReduceLROnPlateau | CosineAnnealingWarmRestarts:
        """Create a learning rate scheduler based on config."""
        if self.lr_scheduler_type == "cosine_warm":
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.lr_cosine_t0,
                T_mult=self.lr_cosine_t_mult,
                eta_min=self.lr_cosine_eta_min,
            )
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.lr_scheduler_patience,
            factor=self.lr_scheduler_factor,
            min_lr=self.lr_scheduler_min_lr,
        )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | np.ndarray,
        eval_set: (
            tuple[
                pd.DataFrame | np.ndarray,
                pd.DataFrame | np.ndarray,
            ]
            | None
        ) = None,
        season: np.ndarray | pd.Series | None = None,
    ) -> MTLForecaster:
        """Train the MTL network.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Target matrix (n_samples, n_targets).
        eval_set:
            Optional ``(X_val, y_val)`` for early stopping on
            validation mean RMSE (computed in original target scale).
        season:
            Optional season array ``(n_samples,)`` for recency
            weighting.  When provided and ``recency_decay_lambda > 0``,
            samples from more recent seasons receive higher loss weight.

        Returns
        -------
        MTLForecaster
            ``self`` (for call chaining).
        """
        self._set_seeds()

        # Extract names from DataFrames
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"f{i}" for i in range(X.shape[1])]
        if isinstance(y, pd.DataFrame):
            self.target_names_ = list(y.columns)
        else:
            self.target_names_ = [f"target_{i}" for i in range(y.shape[1])]

        X_arr = _to_float64_array(X)
        y_arr = _to_float64_array(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        n_features = X_arr.shape[1]
        n_targets = y_arr.shape[1]

        # Feature scaling  (NaN → 0 after scaling)
        self.feature_scaler_ = StandardScaler()
        X_scaled = self.feature_scaler_.fit_transform(X_arr)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # Record per-feature bounds so we can clamp OOD inputs later
        # (e.g. bat speed is constant in training but real in test).
        self.feature_min_ = X_scaled.min(axis=0)
        self.feature_max_ = X_scaled.max(axis=0)

        # Target winsorization (optional, reduces outlier influence on scaling)
        if self.target_winsorize_pct > 0:
            lower = np.percentile(y_arr, self.target_winsorize_pct, axis=0)
            upper = np.percentile(y_arr, 100 - self.target_winsorize_pct, axis=0)
            y_arr = np.clip(y_arr, lower, upper)

        # Target scaling
        self.target_scaler_ = StandardScaler()
        y_scaled = self.target_scaler_.fit_transform(y_arr)

        # Recency sample weights
        sample_weights: np.ndarray | None = None
        if season is not None and self.recency_decay_lambda > 0:
            season_arr = np.asarray(season, dtype=np.float64)
            max_season = season_arr.max()
            raw_weights = np.exp(
                -self.recency_decay_lambda * (max_season - season_arr)
            )
            sample_weights = raw_weights / raw_weights.mean()

        # DataLoaders
        gen = torch.Generator().manual_seed(self.seed)
        train_ds = BatterDataset(X_scaled, y_scaled, sample_weights=sample_weights)
        self._use_sample_weights = train_ds.has_nontrivial_weights
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            generator=gen,
        )

        val_loader: DataLoader | None = None
        if eval_set is not None:
            X_val_arr = _to_float64_array(eval_set[0])
            y_val_arr = _to_float64_array(eval_set[1])
            if y_val_arr.ndim == 1:
                y_val_arr = y_val_arr.reshape(-1, 1)
            X_val_scaled = self._scale_and_clamp(X_val_arr)
            y_val_scaled = self.target_scaler_.transform(y_val_arr)
            val_ds = BatterDataset(X_val_scaled, y_val_scaled)
            val_loader = DataLoader(val_ds, batch_size=len(val_ds))

        # Network + loss
        self.network_ = MTLNetwork(
            n_features,
            n_targets,
            self.hidden_dims,
            self.head_dim,
            self.dropouts,
            use_residual=self.use_residual,
            use_gated_heads=self.use_gated_heads,
            two_stage=self.two_stage,
            speed_head_indices=self.speed_head_indices,
            speed_heads_receive_rates=self.speed_heads_receive_rates,
        )
        if self.loss_type == "huber":
            self.loss_fn_ = HuberMultiTaskLoss(n_targets, delta=self.loss_delta)
        else:
            self.loss_fn_ = MultiTaskLoss(n_targets)

        # Optimiser + scheduler
        params = list(self.network_.parameters()) + list(self.loss_fn_.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = self._make_scheduler(optimizer)

        # Training loop
        best_val_rmse = float("inf")
        patience_counter = 0
        best_network_state: dict | None = None
        best_loss_state: dict | None = None
        self.training_history_ = []

        for epoch in range(self.max_epochs):
            train_loss = self._train_one_epoch(train_loader, optimizer)

            epoch_record: dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }

            if val_loader is not None:
                val_rmse = self._validate(val_loader)
                epoch_record["val_rmse"] = val_rmse

                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_rmse)
                else:
                    scheduler.step()

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_network_state = deepcopy(self.network_.state_dict())
                    best_loss_state = deepcopy(self.loss_fn_.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d (best val RMSE %.4f)",
                        epoch + 1,
                        best_val_rmse,
                    )
                    break
            elif not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()

            self.training_history_.append(epoch_record)

        # Restore best weights
        if best_network_state is not None:
            self.network_.load_state_dict(best_network_state)
        if best_loss_state is not None:
            self.loss_fn_.load_state_dict(best_loss_state)

        if self.swa_enabled:
            self._run_swa(train_loader, optimizer)

        self.network_.eval()
        self.is_fitted_ = True
        return self

    def _run_swa(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Run a post-training SWA phase."""
        from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

        swa_model = AveragedModel(self.network_)
        swa_scheduler = SWALR(
            optimizer,
            swa_lr=self.swa_lr,
            anneal_epochs=self.swa_anneal_epochs,
        )
        for _ in range(self.swa_epochs):
            self._train_one_epoch(train_loader, optimizer)
            swa_model.update_parameters(self.network_)
            swa_scheduler.step()

        update_bn(train_loader, swa_model)
        self.network_.load_state_dict(
            {k: v for k, v in swa_model.module.state_dict().items()}
        )

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run one training epoch; return mean batch loss."""
        self.network_.train()
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch, w_batch in loader:
            if self.mixup_alpha > 0.0 and X_batch.size(0) > 1:
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                idx = torch.randperm(X_batch.size(0))
                X_batch = lam * X_batch + (1 - lam) * X_batch[idx]
                y_batch = lam * y_batch + (1 - lam) * y_batch[idx]
                w_batch = lam * w_batch + (1 - lam) * w_batch[idx]

            weights = w_batch if self._use_sample_weights else None
            optimizer.zero_grad()
            predictions = self.network_(X_batch)
            loss, _ = self.loss_fn_(predictions, y_batch, sample_weights=weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        """Compute validation mean RMSE in original target scale."""
        self.network_.eval()
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        for X_batch, y_batch, _ in loader:
            predictions = self.network_(X_batch)
            pred_tensor = torch.cat(predictions, dim=1)
            all_preds.append(pred_tensor.numpy())
            all_targets.append(y_batch.numpy())

        preds_scaled = np.vstack(all_preds)
        targets_scaled = np.vstack(all_targets)

        preds_orig = self.target_scaler_.inverse_transform(preds_scaled)
        targets_orig = self.target_scaler_.inverse_transform(targets_scaled)

        per_target = [
            rmse(targets_orig[:, i], preds_orig[:, i])
            for i in range(preds_orig.shape[1])
        ]
        return float(np.mean(per_target))

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Generate predictions for all targets.

        Returns a DataFrame with one column per target, inverse-
        transformed to the original scale.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        index = X.index if isinstance(X, pd.DataFrame) else None
        X_arr = _to_float64_array(X)
        X_scaled = self._scale_and_clamp(X_arr)

        self.network_.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            outputs = self.network_(X_tensor)
            pred_scaled = torch.cat(outputs, dim=1).numpy()

        pred_original = self.target_scaler_.inverse_transform(pred_scaled)
        return pd.DataFrame(pred_original, columns=self.target_names_, index=index)

    # ------------------------------------------------------------------
    # Learned task weights
    # ------------------------------------------------------------------

    def get_learned_task_weights(self) -> dict[str, float]:
        """Return precision weights per target (higher = more predictable).

        Precision is ``exp(-log_var)`` from the uncertainty-weighted loss.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        weights = self.loss_fn_.get_task_weights()
        return dict(zip(self.target_names_, weights.tolist()))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Save model state to disk.

        Uses ``torch.save`` with state dicts (not pickle of the full
        nn.Module) for robustness across PyTorch versions.
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted model.")

        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "mtl_forecaster.pt"

        state = {
            "network_state_dict": self.network_.state_dict(),
            "loss_state_dict": self.loss_fn_.state_dict(),
            # Store scaler parameters as tensors (not pickled sklearn
            # objects) so we can use weights_only=True on load.
            "feature_scaler_mean": torch.from_numpy(self.feature_scaler_.mean_),
            "feature_scaler_scale": torch.from_numpy(self.feature_scaler_.scale_),
            "feature_scaler_var": torch.from_numpy(self.feature_scaler_.var_),
            "feature_scaler_n_samples": torch.as_tensor(
                self.feature_scaler_.n_samples_seen_
            ),
            "target_scaler_mean": torch.from_numpy(self.target_scaler_.mean_),
            "target_scaler_scale": torch.from_numpy(self.target_scaler_.scale_),
            "target_scaler_var": torch.from_numpy(self.target_scaler_.var_),
            "target_scaler_n_samples": torch.as_tensor(
                self.target_scaler_.n_samples_seen_
            ),
            "target_names": self.target_names_,
            "feature_names": self.feature_names_,
            "hidden_dims": self.hidden_dims,
            "head_dim": self.head_dim,
            "dropouts": self.dropouts,
            "use_residual": self.use_residual,
            "use_gated_heads": self.use_gated_heads,
            "two_stage": self.two_stage,
            "speed_head_indices": self.speed_head_indices,
            "speed_heads_receive_rates": self.speed_heads_receive_rates,
            "seed": self.seed,
            "training_history": self.training_history_,
            "n_features": len(self.feature_names_),
            "n_targets": len(self.target_names_),
            "feature_min": torch.from_numpy(self.feature_min_),
            "feature_max": torch.from_numpy(self.feature_max_),
            "loss_type": self.loss_type,
            "loss_delta": self.loss_delta,
            "target_winsorize_pct": self.target_winsorize_pct,
        }
        torch.save(state, model_path)
        logger.info("Saved MTL model → %s", model_path)
        return model_path

    @classmethod
    def load(cls, path: str | Path) -> MTLForecaster:
        """Load a saved model from disk.

        Reconstructs the network architecture from the saved config
        values and loads the trained weights.
        """
        model_path = Path(path) / "mtl_forecaster.pt"
        state = torch.load(model_path, map_location="cpu", weights_only=True)

        instance = cls()
        instance.feature_names_ = state["feature_names"]
        instance.target_names_ = state["target_names"]

        # Reconstruct StandardScalers from saved parameters
        instance.feature_scaler_ = _reconstruct_scaler(
            state["feature_scaler_mean"],
            state["feature_scaler_scale"],
            state["feature_scaler_var"],
            state["feature_scaler_n_samples"],
        )
        instance.target_scaler_ = _reconstruct_scaler(
            state["target_scaler_mean"],
            state["target_scaler_scale"],
            state["target_scaler_var"],
            state["target_scaler_n_samples"],
        )
        instance.hidden_dims = state["hidden_dims"]
        instance.head_dim = state["head_dim"]
        instance.dropouts = state["dropouts"]
        instance.use_residual = state.get("use_residual", False)
        instance.use_gated_heads = state.get("use_gated_heads", False)
        instance.two_stage = state.get("two_stage", False)
        instance.speed_head_indices = state.get("speed_head_indices", [])
        instance.speed_heads_receive_rates = state.get(
            "speed_heads_receive_rates", False
        )
        instance.seed = state["seed"]
        instance.training_history_ = state.get("training_history", [])
        feat_min = state.get("feature_min")
        feat_max = state.get("feature_max")
        instance.feature_min_ = (
            feat_min.numpy() if isinstance(feat_min, torch.Tensor) else feat_min
        )
        instance.feature_max_ = (
            feat_max.numpy() if isinstance(feat_max, torch.Tensor) else feat_max
        )

        n_features = state["n_features"]
        n_targets = state["n_targets"]

        instance.network_ = MTLNetwork(
            n_features,
            n_targets,
            instance.hidden_dims,
            instance.head_dim,
            instance.dropouts,
            use_residual=instance.use_residual,
            use_gated_heads=instance.use_gated_heads,
            two_stage=instance.two_stage,
            speed_head_indices=instance.speed_head_indices,
            speed_heads_receive_rates=instance.speed_heads_receive_rates,
        )
        network_sd = _migrate_legacy_backbone_keys(state["network_state_dict"])
        instance.network_.load_state_dict(network_sd)
        instance.network_.eval()

        loss_type = state.get("loss_type", "mse")
        loss_delta = state.get("loss_delta", 2.0)
        instance.loss_type = loss_type
        instance.loss_delta = loss_delta
        instance.target_winsorize_pct = state.get("target_winsorize_pct", 0.0)
        if loss_type == "huber":
            instance.loss_fn_ = HuberMultiTaskLoss(n_targets, delta=loss_delta)
        else:
            instance.loss_fn_ = MultiTaskLoss(n_targets)
        instance.loss_fn_.load_state_dict(state["loss_state_dict"])

        instance.is_fitted_ = True
        logger.info("Loaded MTL model ← %s", model_path)
        return instance


# ---------------------------------------------------------------------------
# Multi-seed ensemble of MTLForecaster instances
# ---------------------------------------------------------------------------


class MTLEnsembleForecaster:
    """Multi-seed ensemble of MTLForecaster instances.

    Trains N independent MTLForecaster models with different random seeds
    and averages their predictions to reduce initialization variance.
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = {}
        ensemble_cfg = config.get("ensemble", {})
        self.n_seeds: int = ensemble_cfg.get("n_seeds", 5)
        if self.n_seeds < 1:
            raise ValueError(f"n_seeds must be >= 1, got {self.n_seeds}")
        self.base_seed: int = ensemble_cfg.get("base_seed", config.get("seed", 42))
        self.config = config
        self.models_: list[MTLForecaster] = []
        self.is_fitted_: bool = False
        self.feature_names_: list[str] = []
        self.target_names_: list[str] = []

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | np.ndarray,
        eval_set: tuple | None = None,
        season: np.ndarray | pd.Series | None = None,
    ) -> "MTLEnsembleForecaster":
        """Train N models with different seeds."""
        self.models_ = []
        for i in range(self.n_seeds):
            cfg = deepcopy(self.config)
            cfg["seed"] = self.base_seed + i
            model = MTLForecaster(cfg)
            logger.info(
                "Training ensemble member %d/%d (seed=%d) ...",
                i + 1,
                self.n_seeds,
                cfg["seed"],
            )
            model.fit(X, y, eval_set=eval_set, season=season)
            self.models_.append(model)
        self.feature_names_ = self.models_[0].feature_names_
        self.target_names_ = self.models_[0].target_names_
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Average predictions across all ensemble members."""
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted.")
        preds = [m.predict(X) for m in self.models_]
        avg = preds[0].copy()
        for p in preds[1:]:
            avg += p
        avg /= len(preds)
        return avg

    def get_learned_task_weights(self) -> dict[str, float]:
        """Return averaged task weights."""
        if not self.models_:
            return {}
        all_weights = [m.get_learned_task_weights() for m in self.models_]
        keys = all_weights[0].keys()
        return {k: sum(w[k] for w in all_weights) / len(all_weights) for k in keys}

    def save(self, path: str | Path) -> Path:
        """Save ensemble to disk."""
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {"n_seeds": self.n_seeds, "base_seed": self.base_seed}
        with open(out_dir / "ensemble_meta.json", "w") as f:
            json.dump(meta, f)
        for i, model in enumerate(self.models_):
            model.save(out_dir / f"seed_{i}")
        logger.info("Saved MTL ensemble (%d members) → %s", self.n_seeds, out_dir)
        return out_dir

    @classmethod
    def load(cls, path: str | Path) -> "MTLEnsembleForecaster":
        """Load ensemble from disk."""
        load_dir = Path(path)
        with open(load_dir / "ensemble_meta.json") as f:
            meta = json.load(f)
        instance = cls()
        instance.n_seeds = meta["n_seeds"]
        instance.base_seed = meta["base_seed"]
        instance.models_ = []
        for i in range(instance.n_seeds):
            model = MTLForecaster.load(load_dir / f"seed_{i}")
            instance.models_.append(model)
        instance.feature_names_ = instance.models_[0].feature_names_
        instance.target_names_ = instance.models_[0].target_names_
        instance.is_fitted_ = True
        logger.info("Loaded MTL ensemble (%d members) ← %s", instance.n_seeds, load_dir)
        return instance
