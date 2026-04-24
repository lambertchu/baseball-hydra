"""MTLQuantileNetwork + MTLQuantileForecaster + ensemble wrapper.

Phase 2 (ROS) parallel to ``src/models/mtl/model.py``.  The preseason MTL
produces point-estimate predictions for the six per-PA-rate targets
(OBP, SLG, HR/PA, R/PA, RBI/PA, SB/PA).  The ROS variant instead produces
a *distribution* for each of those targets — ``n_quantiles`` pinball-regressed
quantiles at fixed tau levels — plus an auxiliary PA-remaining regression
head.

Design choices:

* **Re-use ``ResidualBlock``** from ``src.models.mtl.model`` for the
  shared backbone.  This is the only dependency on preseason internals.
  The preseason network itself is frozen and untouched.
* **Two-stage rate→count** mirrors the preseason: rate heads (indices 0,1)
  run first on pure backbone output; count heads (indices 2-5) receive
  ``concat(backbone_output, rate_quantile_medians_detached)``.  Medians
  come from the tau=0.50 slice (index 2 in the default 5-quantile grid).
  Speed heads (``speed_head_indices``) optionally skip the rate input.
* **Quantile scaling**: both the network and the preseason-style target
  scaler work in z-scored target space.  At predict-time we inverse-transform
  each quantile slice independently (``StandardScaler`` is affine, so the
  quantile order is preserved).
* **PA target IS scaled** (StandardScaler, stored as ``pa_scaler_``).  The
  PA-remaining head is trained in z-scored PA space so the MSE loss sits
  near O(1) alongside the pinball-scale rate losses; without this,
  ``pa_weight=1.0`` would dominate the total loss by ~6 orders of magnitude
  and drown out the quantile heads.  At predict-time PA is inverse-scaled
  back to raw PA units.
* **OOD clamp**: identical to preseason — scale inputs to training stats,
  clip to the per-feature train min/max before feeding the network.

Usage
-----
    from src.models.mtl_ros.model import MTLQuantileForecaster

    model = MTLQuantileForecaster(config)
    model.fit(X_train, y_train, pa_target=pa_train,
              eval_set=(X_val, y_val, pa_val))
    preds = model.predict(X_test)
    # preds == {"quantiles": (n, 6, 5), "pa_remaining": (n, 1)}
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
from torch.utils.data import DataLoader, Dataset

from src.models.mtl.model import ResidualBlock
from src.models.mtl_ros.loss import MultiTaskQuantileLoss
from src.models.utils import reconstruct_scaler, scale_and_clamp
from src.models.utils import to_float64_array as _to_float64_array

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class MTLQuantileNetwork(nn.Module):
    """Shared backbone + quantile heads + PA-remaining head.

    Architecture (default)::

        Input → BatchNorm1d
              → ResidualBlock(N, 256) → ResidualBlock(256, 128) → ResidualBlock(128, 64)
              +- Head 0  (rate):  Linear(64, 32) → ReLU → Linear(32, n_quantiles)
              +- Head 1  (rate):  Linear(64, 32) → ReLU → Linear(32, n_quantiles)
              +- Head 2  (count): Linear(66, 32) → ReLU → Linear(32, n_quantiles)
                  ...
              +- Head 5  (speed): Linear(64, 32) → ReLU → Linear(32, n_quantiles)
              +- pa_head:         Linear(64, 32) → ReLU → Linear(32, 1)

    Parameters
    ----------
    n_features:
        Number of input feature columns.
    n_targets:
        Number of rate prediction targets.  Default 6.
    n_quantiles:
        Number of quantiles per target head.  Default 5.
    hidden_dims, dropouts, head_dim, use_residual:
        Backbone sizing, same semantics as the preseason MTL.
    two_stage:
        If True, count heads (indices ≥ 2 that are not in ``speed_head_indices``
        with ``speed_heads_receive_rates=False``) receive
        ``concat(backbone, rate_quantile_medians_detached)``.
    speed_head_indices:
        Target indices whose heads use pure backbone output even in two-stage
        mode.  Default ``(5,)`` — the SB/PA head.
    speed_heads_receive_rates:
        When True, speed heads also receive the rate medians (ignoring the
        ``speed_head_indices`` exclusion).  Default False.
    taus:
        Quantile levels.  Stored as a buffer (moves with ``.to(device)``
        but is not learnable).  Default ``(0.05, 0.25, 0.50, 0.75, 0.95)``.
    """

    def __init__(
        self,
        n_features: int,
        n_targets: int = 6,
        n_quantiles: int = 5,
        hidden_dims: list[int] | None = None,
        head_dim: int = 32,
        dropouts: list[float] | None = None,
        use_residual: bool = False,
        two_stage: bool = True,
        speed_head_indices: tuple[int, ...] | list[int] = (5,),
        speed_heads_receive_rates: bool = False,
        taus: tuple[float, ...] | list[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
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
        if len(taus) != n_quantiles:
            raise ValueError(
                f"taus has {len(taus)} entries but n_quantiles={n_quantiles}"
            )

        self.n_targets = n_targets
        self.n_quantiles = n_quantiles
        self.hidden_dims = list(hidden_dims)
        self.two_stage = two_stage
        self.speed_head_indices = tuple(speed_head_indices)
        self.speed_heads_receive_rates = speed_heads_receive_rates

        # Locate the tau=0.5 slice for the rate→count hand-off.  Fall back to
        # the middle quantile if 0.5 isn't present.
        taus_list = list(taus)
        try:
            self.median_index = taus_list.index(0.5)
        except ValueError:
            self.median_index = n_quantiles // 2

        # Non-learnable tau buffer — travels with state_dict / .to(device).
        self.register_buffer("taus", torch.tensor(taus_list, dtype=torch.float32))

        self.batch_norm = nn.BatchNorm1d(n_features)

        # Shared backbone — identical block structure to preseason MTL so the
        # preseason ResidualBlock is reusable directly.
        backbone_layers: list[nn.Module] = []
        in_dim = n_features
        for h_dim, drop in zip(hidden_dims, dropouts):
            backbone_layers.append(
                ResidualBlock(in_dim, h_dim, dropout=drop, use_residual=use_residual)
            )
            in_dim = h_dim
        self.backbone = nn.Sequential(*backbone_layers)

        backbone_out_dim = in_dim

        # Per-target quantile heads.
        self.heads = nn.ModuleList()
        for i in range(n_targets):
            uses_rate_input = (
                two_stage
                and i >= 2
                and (i not in self.speed_head_indices or self.speed_heads_receive_rates)
            )
            # Rate-medians add 2 scalars (OBP_median, SLG_median) per row.
            head_in = backbone_out_dim + 2 if uses_rate_input else backbone_out_dim
            self.heads.append(
                nn.Sequential(
                    nn.Linear(head_in, head_dim),
                    nn.ReLU(),
                    nn.Linear(head_dim, n_quantiles),
                )
            )

        # PA-remaining head — point estimate on raw PA units.  Takes backbone
        # only (no rate input); PA is largely an availability / playing-time
        # signal and shouldn't be confounded with rate predictions.
        self.pa_head = nn.Sequential(
            nn.Linear(backbone_out_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode scaled Phase 2 features into the shared latent state."""
        x = self.batch_norm(x)
        return self.backbone(x)

    def decode_from_latent(self, shared: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode a shared latent state with the Phase 2 quantile/PA heads.

        This is factored out for Phase 3's sequential ROS model, which builds
        its own GRU latent and then reuses these frozen decoder heads.  The
        logic is intentionally identical to the previous ``forward`` body.
        """
        pa_out = self.pa_head(shared)  # (batch, 1)

        quantile_outputs: list[torch.Tensor] = [None] * self.n_targets  # type: ignore[list-item]

        if not self.two_stage:
            for i, head in enumerate(self.heads):
                quantile_outputs[i] = head(shared)  # (batch, n_quantiles)
        else:
            # Stage 1: rate heads (OBP=0, SLG=1) on raw backbone.
            rate_medians: list[torch.Tensor] = []
            for i in (0, 1):
                out = self.heads[i](shared)  # (batch, n_quantiles)
                quantile_outputs[i] = out
                # Detach the median slice so gradients do NOT flow from
                # count heads back into the rate heads' quantile predictions.
                median = out[:, self.median_index : self.median_index + 1]
                rate_medians.append(median.detach())

            count_input = torch.cat([shared, *rate_medians], dim=1)

            # Stage 2: count / speed heads.
            for i in range(2, self.n_targets):
                if i in self.speed_head_indices and not self.speed_heads_receive_rates:
                    quantile_outputs[i] = self.heads[i](shared)
                else:
                    quantile_outputs[i] = self.heads[i](count_input)

        # Stack per-target (batch, n_quantiles) → (batch, n_targets, n_quantiles).
        quantiles = torch.stack(quantile_outputs, dim=1)
        return {"quantiles": quantiles, "pa_remaining": pa_out}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the network.

        Returns a dict::

            {
                "quantiles":    (batch, n_targets, n_quantiles),
                "pa_remaining": (batch, 1),
            }
        """
        return self.decode_from_latent(self.encode(x))


# ---------------------------------------------------------------------------
# Small internal dataset used during training.
#
# Task 3 will replace this with a cutoff-sampling dataset that randomises the
# within-season observation point for each player-season; for Task 2 we only
# need a tensor dataset that yields (X, y, pa_target, sample_weight) tuples.
# ---------------------------------------------------------------------------


class _QuantileDataset(Dataset):
    """(features, rate_targets, pa_target, sample_weight) rows."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        pa_target: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        X_arr = np.asarray(X, dtype=np.float32)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        y_arr = np.asarray(y, dtype=np.float32)
        pa_arr = np.asarray(pa_target, dtype=np.float32).reshape(-1, 1)
        if sample_weights is not None:
            w_arr = np.asarray(sample_weights, dtype=np.float32)
            self.has_nontrivial_weights = True
        else:
            w_arr = np.ones(len(X_arr), dtype=np.float32)
            self.has_nontrivial_weights = False

        self.X = torch.from_numpy(X_arr)
        self.y = torch.from_numpy(y_arr)
        self.pa = torch.from_numpy(pa_arr)
        self.w = torch.from_numpy(w_arr)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.pa[idx], self.w[idx]


# ---------------------------------------------------------------------------
# Forecaster (sklearn-style wrapper)
# ---------------------------------------------------------------------------


class MTLQuantileForecaster:
    """Sklearn-style wrapper around ``MTLQuantileNetwork``.

    Mirrors the preseason ``MTLForecaster`` fit/predict/save/load interface.
    The key differences:

    * ``fit(X, y, pa_target=...)`` requires a PA-remaining target.
    * ``predict(X)`` returns a ``dict`` with ``"quantiles"`` (n, T, Q) and
      ``"pa_remaining"`` (n, 1), inverse-transformed back to raw target scale.
    * The target scaler is applied to the 6 rate targets only; PA target is
      passed through in raw PA units (no scaler).
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = {}
        model_cfg = config.get("model", {})
        train_cfg = config.get("training", {})
        loss_cfg = config.get("loss", {})

        # Architecture
        self.n_targets: int = model_cfg.get("n_targets", 6)
        self.n_quantiles: int = model_cfg.get("n_quantiles", 5)
        self.taus: list[float] = list(
            model_cfg.get("taus", [0.05, 0.25, 0.50, 0.75, 0.95])
        )
        self.hidden_dims: list[int] = list(model_cfg.get("hidden_dims", [256, 128, 64]))
        self.head_dim: int = model_cfg.get("head_dim", 32)
        self.dropouts: list[float] = list(model_cfg.get("dropouts", [0.3, 0.2, 0.1]))
        self.use_residual: bool = model_cfg.get("use_residual", False)
        self.two_stage: bool = model_cfg.get("two_stage", True)
        self.speed_head_indices: tuple[int, ...] = tuple(
            model_cfg.get("speed_head_indices", (5,))
        )
        self.speed_heads_receive_rates: bool = model_cfg.get(
            "speed_heads_receive_rates", False
        )

        # Loss
        self.pa_loss_type: str = loss_cfg.get("pa_loss", "mse")
        self.pa_weight: float = float(loss_cfg.get("pa_weight", 1.0))

        # Training
        self.batch_size: int = train_cfg.get("batch_size", 64)
        self.max_epochs: int = train_cfg.get("epochs", 200)
        self.learning_rate: float = train_cfg.get("learning_rate", 1e-3)
        self.weight_decay: float = train_cfg.get("weight_decay", 1e-4)
        self.early_stopping_patience: int = train_cfg.get("early_stopping_patience", 20)
        self.recency_decay_lambda: float = train_cfg.get("recency_decay_lambda", 0.0)
        self.device_: str = str(train_cfg.get("device", "cpu"))
        self.seed: int = config.get("seed", 42)

        # LR scheduler
        lr_cfg = train_cfg.get("lr_scheduler", {})
        self.lr_scheduler_type: str = lr_cfg.get("type", "plateau")
        self.lr_scheduler_patience: int = lr_cfg.get("patience", 10)
        self.lr_scheduler_factor: float = lr_cfg.get("factor", 0.5)
        self.lr_scheduler_min_lr: float = lr_cfg.get("min_lr", 1e-6)
        self.lr_cosine_t0: int = lr_cfg.get("T_0", 20)
        self.lr_cosine_t_mult: int = lr_cfg.get("T_mult", 2)
        self.lr_cosine_eta_min: float = lr_cfg.get("eta_min", 1e-6)

        # Keep the full config around so save/load can round-trip exotic keys.
        self.config = deepcopy(config)

        # State populated during fit
        self.network_: MTLQuantileNetwork | None = None
        self.loss_fn_: MultiTaskQuantileLoss | None = None
        self.feature_scaler_: StandardScaler | None = None
        self.target_scaler_: StandardScaler | None = None
        # PA target scaler — kept separate from ``target_scaler_`` because PA
        # is scalar and needs its own affine transform for inverse-scaling
        # inside ``predict()``.
        self.pa_scaler_: StandardScaler | None = None
        self.feature_names_: list[str] = []
        self.target_names_: list[str] = []
        self.is_fitted_: bool = False
        self.training_history_: list[dict] = []
        # Per-feature scaled-space min/max for OOD clamping at predict time.
        self.feature_min_: np.ndarray | None = None
        self.feature_max_: np.ndarray | None = None
        # Whether any sample had non-unit weight — toggles DataLoader plumbing.
        self._use_sample_weights: bool = False

    # ------------------------------------------------------------------
    # Seed / scaling helpers
    # ------------------------------------------------------------------

    def _set_seeds(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _scale_and_clamp(self, X_arr: np.ndarray) -> np.ndarray:
        """Scale inputs and clip to the training per-feature range.

        Thin wrapper over :func:`src.models.utils.scale_and_clamp` that
        reads the fitted scaler + train-time per-feature min/max from
        ``self``. Kept as a method so call sites don't change.
        """
        return scale_and_clamp(
            X_arr, self.feature_scaler_, self.feature_min_, self.feature_max_
        )

    def _make_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> ReduceLROnPlateau | CosineAnnealingWarmRestarts:
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
        pa_target: np.ndarray | pd.Series,
        sample_weights: np.ndarray | None = None,
        eval_set: (
            tuple[
                pd.DataFrame | np.ndarray,
                pd.DataFrame | np.ndarray,
                np.ndarray | pd.Series,
            ]
            | None
        ) = None,
        season: np.ndarray | pd.Series | None = None,
    ) -> "MTLQuantileForecaster":
        """Train the quantile MTL network.

        Parameters
        ----------
        X:
            Feature matrix ``(n_samples, n_features)``.
        y:
            Rate-target matrix ``(n_samples, n_targets)``.
        pa_target:
            PA-remaining regression target ``(n_samples,)``.  Raw PA units;
            not scaled.
        sample_weights:
            Optional explicit per-sample weights ``(n_samples,)``.  If both
            ``sample_weights`` and ``season`` are provided, the explicit
            weights win; recency weights are only derived from ``season``
            when ``sample_weights is None``.
        eval_set:
            Optional ``(X_val, y_val, pa_val)`` for early stopping on the
            mean pinball loss in original target scale.
        season:
            Optional season array ``(n_samples,)`` for recency weighting.
            When provided and ``recency_decay_lambda > 0``, more recent
            seasons receive higher loss weight.
        """
        self._set_seeds()

        # Capture feature / target names.
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
        pa_arr = np.asarray(pa_target, dtype=np.float64).reshape(-1, 1)

        n_features = X_arr.shape[1]
        n_targets = y_arr.shape[1]
        if n_targets != self.n_targets:
            # Allow y to dictate n_targets if the user passed a differently
            # shaped frame, to match preseason's permissive behaviour.
            self.n_targets = n_targets

        # Feature scaling (NaN → 0 after scaling)
        self.feature_scaler_ = StandardScaler()
        X_scaled = self.feature_scaler_.fit_transform(X_arr)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        self.feature_min_ = X_scaled.min(axis=0)
        self.feature_max_ = X_scaled.max(axis=0)

        # Target scaling — rate targets AND PA are both scaled into z-space
        # so the losses sit on comparable magnitudes. Raw PA targets range
        # ~0–700; raw MSE on those is O(10_000) while rate-target pinball loss
        # is O(0.01), so an unscaled PA head with pa_weight=1.0 would drown
        # out the quantile objective.
        self.target_scaler_ = StandardScaler()
        y_scaled = self.target_scaler_.fit_transform(y_arr)

        self.pa_scaler_ = StandardScaler()
        pa_scaled = self.pa_scaler_.fit_transform(pa_arr)

        # Sample weights: explicit > recency-from-season > none.
        sw: np.ndarray | None = None
        if sample_weights is not None:
            sw = np.asarray(sample_weights, dtype=np.float64)
        elif season is not None and self.recency_decay_lambda > 0:
            season_arr = np.asarray(season, dtype=np.float64)
            max_season = season_arr.max()
            raw_weights = np.exp(-self.recency_decay_lambda * (max_season - season_arr))
            sw = raw_weights / raw_weights.mean()

        # Training DataLoader. ``drop_last=True`` avoids the "BatchNorm1d
        # expects >1 value per channel" crash when the final batch has size 1
        # (``len(dataset) % batch_size == 1``). Since we shuffle, losing at
        # most one sample per epoch is negligible.
        gen = torch.Generator().manual_seed(self.seed)
        train_ds = _QuantileDataset(X_scaled, y_scaled, pa_scaled, sample_weights=sw)
        self._use_sample_weights = train_ds.has_nontrivial_weights
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            generator=gen,
            drop_last=True,
        )

        # Optional eval loader (uses a full-batch pass for speed).
        val_loader: DataLoader | None = None
        if eval_set is not None:
            X_val_arr = _to_float64_array(eval_set[0])
            y_val_arr = _to_float64_array(eval_set[1])
            pa_val_arr = np.asarray(eval_set[2], dtype=np.float64).reshape(-1, 1)
            if y_val_arr.ndim == 1:
                y_val_arr = y_val_arr.reshape(-1, 1)
            X_val_scaled = self._scale_and_clamp(X_val_arr)
            y_val_scaled = self.target_scaler_.transform(y_val_arr)
            pa_val_scaled = self.pa_scaler_.transform(pa_val_arr)
            val_ds = _QuantileDataset(X_val_scaled, y_val_scaled, pa_val_scaled)
            val_loader = DataLoader(val_ds, batch_size=len(val_ds))

        # Network + loss — moved to the configured device so batches can flow
        # through without a per-step ``.cpu()`` detour.
        device = torch.device(self.device_)
        self.network_ = MTLQuantileNetwork(
            n_features=n_features,
            n_targets=self.n_targets,
            n_quantiles=self.n_quantiles,
            hidden_dims=self.hidden_dims,
            head_dim=self.head_dim,
            dropouts=self.dropouts,
            use_residual=self.use_residual,
            two_stage=self.two_stage,
            speed_head_indices=self.speed_head_indices,
            speed_heads_receive_rates=self.speed_heads_receive_rates,
            taus=tuple(self.taus),
        ).to(device)
        self.loss_fn_ = MultiTaskQuantileLoss(
            n_tasks=self.n_targets,
            taus=tuple(self.taus),
            pa_loss=self.pa_loss_type,
            pa_weight=self.pa_weight,
        ).to(device)

        # Optimiser
        params = list(self.network_.parameters()) + list(self.loss_fn_.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = self._make_scheduler(optimizer)

        # Training loop with optional early stopping.
        best_val: float = float("inf")
        patience_counter = 0
        best_network_state: dict | None = None
        best_loss_state: dict | None = None
        self.training_history_ = []

        for epoch in range(self.max_epochs):
            train_loss = self._train_one_epoch(train_loader, optimizer)
            record: dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }

            if val_loader is not None:
                val_metric = self._validate(val_loader)
                record["val_metric"] = val_metric
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metric)
                else:
                    scheduler.step()

                if val_metric < best_val:
                    best_val = val_metric
                    best_network_state = deepcopy(self.network_.state_dict())
                    best_loss_state = deepcopy(self.loss_fn_.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d (best val %.4f)",
                        epoch + 1,
                        best_val,
                    )
                    break
            elif not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()

            self.training_history_.append(record)

        if best_network_state is not None:
            self.network_.load_state_dict(best_network_state)
        if best_loss_state is not None:
            self.loss_fn_.load_state_dict(best_loss_state)

        self.network_.eval()
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # Training loop helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.network_.train()
        device = next(self.network_.parameters()).device
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch, pa_batch, w_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pa_batch = pa_batch.to(device)
            weights = w_batch.to(device) if self._use_sample_weights else None
            optimizer.zero_grad()
            outputs = self.network_(X_batch)
            loss, _ = self.loss_fn_(
                outputs["quantiles"],
                outputs["pa_remaining"],
                y_batch,
                pa_batch,
                sample_weights=weights,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        """Return mean pinball loss across tasks in scaled space.

        The scale choice matters less than the relative ordering for early
        stopping; we keep it in scaled space so different seeds are
        directly comparable.
        """
        self.network_.eval()
        device = next(self.network_.parameters()).device
        losses: list[float] = []
        for X_batch, y_batch, pa_batch, _ in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pa_batch = pa_batch.to(device)
            outputs = self.network_(X_batch)
            loss, _ = self.loss_fn_(
                outputs["quantiles"],
                outputs["pa_remaining"],
                y_batch,
                pa_batch,
            )
            losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("inf")

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame | np.ndarray) -> dict[str, np.ndarray]:
        """Return predictions in ORIGINAL target scale.

        ``quantiles`` are inverse-scaled (mean + scale per target applied to
        each quantile slice).  ``pa_remaining`` is inverse-scaled back into
        raw PA units via ``pa_scaler_``.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        X_arr = _to_float64_array(X)
        X_scaled = self._scale_and_clamp(X_arr)

        self.network_.eval()
        device = next(self.network_.parameters()).device
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            outputs = self.network_(X_tensor)
            q_scaled = outputs["quantiles"].cpu().numpy()  # (n, T, Q)
            pa_scaled_pred = outputs["pa_remaining"].cpu().numpy()  # (n, 1)

        # Inverse-transform quantiles.  StandardScaler is affine:
        #   y_orig = y_scaled * scale + mean.
        # Broadcasting (T,) scale/mean across the quantile axis is correct
        # because the same scaler applies regardless of tau.
        scale = self.target_scaler_.scale_.reshape(1, -1, 1)
        mean = self.target_scaler_.mean_.reshape(1, -1, 1)
        q_original = q_scaled * scale + mean

        # Enforce monotonicity across the tau axis. Pinball-loss training does
        # not guarantee tau-ordered outputs on unseen rows, so sort the last
        # axis before handing predictions to callers. StandardScaler is affine
        # with positive scale so sorting post-inverse-transform is equivalent
        # to sorting pre-inverse-transform.
        q_original = np.sort(q_original, axis=-1)

        # Inverse-transform PA back to raw PA units.
        pa_pred = self.pa_scaler_.inverse_transform(pa_scaled_pred)

        return {"quantiles": q_original, "pa_remaining": pa_pred}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted model.")

        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "mtl_ros_forecaster.pt"

        state = {
            "network_state_dict": self.network_.state_dict(),
            "loss_state_dict": self.loss_fn_.state_dict(),
            # Scalers as tensors for weights_only=True load.
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
            "pa_scaler_mean": torch.from_numpy(self.pa_scaler_.mean_),
            "pa_scaler_scale": torch.from_numpy(self.pa_scaler_.scale_),
            "pa_scaler_var": torch.from_numpy(self.pa_scaler_.var_),
            "pa_scaler_n_samples": torch.as_tensor(self.pa_scaler_.n_samples_seen_),
            "feature_names": self.feature_names_,
            "target_names": self.target_names_,
            "n_features": len(self.feature_names_),
            "n_targets": self.n_targets,
            "n_quantiles": self.n_quantiles,
            "taus": self.taus,
            "hidden_dims": self.hidden_dims,
            "head_dim": self.head_dim,
            "dropouts": self.dropouts,
            "use_residual": self.use_residual,
            "two_stage": self.two_stage,
            "speed_head_indices": list(self.speed_head_indices),
            "speed_heads_receive_rates": self.speed_heads_receive_rates,
            "pa_loss_type": self.pa_loss_type,
            "pa_weight": self.pa_weight,
            "seed": self.seed,
            "feature_min": torch.from_numpy(self.feature_min_),
            "feature_max": torch.from_numpy(self.feature_max_),
            "training_history": self.training_history_,
        }
        torch.save(state, model_path)

        # Config JSON for easy inspection.
        config_path = out_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2, default=str)

        logger.info("Saved MTL ROS model → %s", model_path)
        return model_path

    @classmethod
    def load(cls, path: str | Path) -> "MTLQuantileForecaster":
        model_path = Path(path) / "mtl_ros_forecaster.pt"
        state = torch.load(model_path, map_location="cpu", weights_only=True)

        instance = cls()
        instance.feature_names_ = state["feature_names"]
        instance.target_names_ = state["target_names"]

        instance.feature_scaler_ = reconstruct_scaler(
            state["feature_scaler_mean"],
            state["feature_scaler_scale"],
            state["feature_scaler_var"],
            state["feature_scaler_n_samples"],
        )
        instance.target_scaler_ = reconstruct_scaler(
            state["target_scaler_mean"],
            state["target_scaler_scale"],
            state["target_scaler_var"],
            state["target_scaler_n_samples"],
        )
        # PA scaler was added for the loss-magnitude balance fix. Older
        # checkpoints predate it; fall back to identity (mean=0, scale=1)
        # which is mathematically a no-op on raw PA predictions.
        if "pa_scaler_mean" in state:
            instance.pa_scaler_ = reconstruct_scaler(
                state["pa_scaler_mean"],
                state["pa_scaler_scale"],
                state["pa_scaler_var"],
                state["pa_scaler_n_samples"],
            )
        else:
            identity_pa = StandardScaler()
            identity_pa.mean_ = np.zeros(1)
            identity_pa.scale_ = np.ones(1)
            identity_pa.var_ = np.ones(1)
            identity_pa.n_samples_seen_ = 1
            identity_pa.n_features_in_ = 1
            instance.pa_scaler_ = identity_pa

        instance.n_targets = state["n_targets"]
        instance.n_quantiles = state["n_quantiles"]
        instance.taus = list(state["taus"])
        instance.hidden_dims = list(state["hidden_dims"])
        instance.head_dim = state["head_dim"]
        instance.dropouts = list(state["dropouts"])
        instance.use_residual = state.get("use_residual", False)
        instance.two_stage = state.get("two_stage", True)
        instance.speed_head_indices = tuple(state.get("speed_head_indices", (5,)))
        instance.speed_heads_receive_rates = state.get(
            "speed_heads_receive_rates", False
        )
        instance.pa_loss_type = state.get("pa_loss_type", "mse")
        instance.pa_weight = float(state.get("pa_weight", 1.0))
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
        instance.network_ = MTLQuantileNetwork(
            n_features=n_features,
            n_targets=instance.n_targets,
            n_quantiles=instance.n_quantiles,
            hidden_dims=instance.hidden_dims,
            head_dim=instance.head_dim,
            dropouts=instance.dropouts,
            use_residual=instance.use_residual,
            two_stage=instance.two_stage,
            speed_head_indices=instance.speed_head_indices,
            speed_heads_receive_rates=instance.speed_heads_receive_rates,
            taus=tuple(instance.taus),
        )
        instance.network_.load_state_dict(state["network_state_dict"])
        instance.network_.eval()

        instance.loss_fn_ = MultiTaskQuantileLoss(
            n_tasks=instance.n_targets,
            taus=tuple(instance.taus),
            pa_loss=instance.pa_loss_type,
            pa_weight=instance.pa_weight,
        )
        instance.loss_fn_.load_state_dict(state["loss_state_dict"])

        instance.is_fitted_ = True
        logger.info("Loaded MTL ROS model ← %s", model_path)
        return instance


# ---------------------------------------------------------------------------
# Ensemble wrapper
# ---------------------------------------------------------------------------


class MTLQuantileEnsembleForecaster:
    """Multi-seed ensemble of ``MTLQuantileForecaster``.

    Each member trains with a different seed; at predict time we take the
    **per-quantile** element-wise mean across members (NOT the mean of
    medians — each tau level is averaged independently).  The PA-remaining
    prediction is likewise averaged across members.
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = {}
        ensemble_cfg = config.get("ensemble", {})
        self.n_seeds: int = ensemble_cfg.get("n_seeds", 5)
        if self.n_seeds < 1:
            raise ValueError(f"n_seeds must be >= 1, got {self.n_seeds}")
        self.base_seed: int = ensemble_cfg.get("base_seed", config.get("seed", 42))

        self.config = deepcopy(config)
        self.forecasters_: list[MTLQuantileForecaster] = []
        self.is_fitted_: bool = False
        self.feature_names_: list[str] = []
        self.target_names_: list[str] = []

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | np.ndarray,
        pa_target: np.ndarray | pd.Series,
        sample_weights: np.ndarray | None = None,
        eval_set: tuple | None = None,
        season: np.ndarray | pd.Series | None = None,
    ) -> "MTLQuantileEnsembleForecaster":
        self.forecasters_ = []
        for i in range(self.n_seeds):
            member_cfg = deepcopy(self.config)
            member_cfg["seed"] = self.base_seed + i
            forecaster = MTLQuantileForecaster(member_cfg)
            logger.info(
                "Training MTL ROS ensemble member %d/%d (seed=%d)...",
                i + 1,
                self.n_seeds,
                member_cfg["seed"],
            )
            forecaster.fit(
                X,
                y,
                pa_target=pa_target,
                sample_weights=sample_weights,
                eval_set=eval_set,
                season=season,
            )
            self.forecasters_.append(forecaster)
        self.feature_names_ = self.forecasters_[0].feature_names_
        self.target_names_ = self.forecasters_[0].target_names_
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> dict[str, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("Ensemble is not fitted.")
        preds = [m.predict(X) for m in self.forecasters_]
        # Per-quantile mean: (n_members, n_rows, n_targets, n_quantiles) → mean over 0.
        quantiles_stack = np.stack([p["quantiles"] for p in preds], axis=0)
        pa_stack = np.stack([p["pa_remaining"] for p in preds], axis=0)
        quantiles_mean = quantiles_stack.mean(axis=0)
        # Each member already returns tau-sorted quantiles, but averaging can
        # reintroduce tiny crossings when members disagree on ordering. Sort
        # again post-aggregation so downstream callers always see monotonic
        # quantiles.
        quantiles_mean = np.sort(quantiles_mean, axis=-1)
        return {
            "quantiles": quantiles_mean,
            "pa_remaining": pa_stack.mean(axis=0),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "n_seeds": self.n_seeds,
            "base_seed": self.base_seed,
            "feature_names": self.feature_names_,
            "target_names": self.target_names_,
        }
        with open(out_dir / "ensemble_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        for i, forecaster in enumerate(self.forecasters_):
            forecaster.save(out_dir / f"seed_{i}")
        logger.info(
            "Saved MTL ROS ensemble (%d members) → %s",
            self.n_seeds,
            out_dir,
        )
        return out_dir

    @classmethod
    def load(cls, path: str | Path) -> "MTLQuantileEnsembleForecaster":
        load_dir = Path(path)
        with open(load_dir / "ensemble_meta.json") as f:
            meta = json.load(f)

        instance = cls()
        instance.n_seeds = meta["n_seeds"]
        instance.base_seed = meta["base_seed"]
        instance.forecasters_ = []
        for i in range(instance.n_seeds):
            forecaster = MTLQuantileForecaster.load(load_dir / f"seed_{i}")
            instance.forecasters_.append(forecaster)
        instance.feature_names_ = meta.get(
            "feature_names", instance.forecasters_[0].feature_names_
        )
        instance.target_names_ = meta.get(
            "target_names", instance.forecasters_[0].target_names_
        )
        instance.is_fitted_ = True
        logger.info(
            "Loaded MTL ROS ensemble (%d members) ← %s",
            instance.n_seeds,
            load_dir,
        )
        return instance
