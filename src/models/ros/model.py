"""Phase 3 sequential GRU ROS forecaster."""

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
from torch.utils.data import DataLoader

from src.models.mtl_ros.loss import MultiTaskQuantileLoss
from src.models.mtl_ros.model import MTLQuantileForecaster, MTLQuantileNetwork
from src.models.ros.dataset import (
    ROSCutoffSequenceDataset,
    default_sequence_sample_weights,
)
from src.models.ros.features import (
    SEQUENCE_FEATURE_GROUPS,
    SEQUENCE_FEATURE_NAMES,
    compute_weekly_sequence_features,
)
from src.models.utils import reconstruct_scaler, scale_and_clamp
from src.models.utils import to_float64_array as _to_float64_array

logger = logging.getLogger(__name__)


def _mlp(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim, out_dim),
        nn.ReLU(),
    )


class ROSSequenceNetwork(nn.Module):
    """GRU latent updater with a frozen Phase 2 encoder/decoder."""

    def __init__(
        self,
        phase2_network: MTLQuantileNetwork,
        seq_group_dims: dict[str, int],
        encoder_dim: int = 32,
        gru_hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.phase2_network = phase2_network
        for param in self.phase2_network.parameters():
            param.requires_grad = False
        self.phase2_network.eval()

        self.seq_group_dims = dict(seq_group_dims)
        self.encoder_dim = int(encoder_dim)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.latent_dim = int(self.phase2_network.hidden_dims[-1])

        self.mechanics_encoder = _mlp(
            self.seq_group_dims["mechanics"], self.encoder_dim, dropout
        )
        self.plate_encoder = _mlp(
            self.seq_group_dims["plate"], self.encoder_dim, dropout
        )
        self.outcome_encoder = _mlp(
            self.seq_group_dims["outcome"], self.encoder_dim, dropout
        )
        self.gru = nn.GRU(
            input_size=self.encoder_dim * 3,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.latent_proj = nn.Linear(self.gru_hidden_dim, self.latent_dim)
        self.log_prior_precision = nn.Parameter(torch.zeros(self.latent_dim))
        self.seq_precision = nn.Sequential(
            nn.Linear(2, max(8, self.latent_dim // 4)),
            nn.ReLU(),
            nn.Linear(max(8, self.latent_dim // 4), self.latent_dim),
        )
        self.softplus = nn.Softplus()

    def _split_groups(
        self, seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mech_dim = self.seq_group_dims["mechanics"]
        plate_dim = self.seq_group_dims["plate"]
        mech = seq[..., :mech_dim]
        plate = seq[..., mech_dim : mech_dim + plate_dim]
        outcome = seq[..., mech_dim + plate_dim :]
        return mech, plate, outcome

    def forward(
        self,
        seq: torch.Tensor,
        phase2_x: torch.Tensor,
        seq_mask: torch.Tensor,
        blend_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, time, _ = seq.shape
        mech, plate, outcome = self._split_groups(seq)

        mech_enc = self.mechanics_encoder(mech.reshape(batch * time, -1))
        plate_enc = self.plate_encoder(plate.reshape(batch * time, -1))
        outcome_enc = self.outcome_encoder(outcome.reshape(batch * time, -1))
        encoded = torch.cat([mech_enc, plate_enc, outcome_enc], dim=-1).reshape(
            batch, time, -1
        )

        gru_out, _ = self.gru(encoded)
        seq_latent = self.latent_proj(gru_out)
        with torch.no_grad():
            z0 = self.phase2_network.encode(phase2_x)
        z0 = z0.unsqueeze(1).expand_as(seq_latent)

        prior_precision = self.softplus(self.log_prior_precision).view(1, 1, -1) + 1e-6
        seq_precision = self.softplus(self.seq_precision(blend_features)) + 1e-6
        latent = (prior_precision * z0 + seq_precision * seq_latent) / (
            prior_precision + seq_precision
        )

        flat_latent = latent.reshape(batch * time, self.latent_dim)
        decoded = self.phase2_network.decode_from_latent(flat_latent)
        quantiles = decoded["quantiles"].reshape(
            batch,
            time,
            self.phase2_network.n_targets,
            self.phase2_network.n_quantiles,
        )
        pa_remaining = decoded["pa_remaining"].reshape(batch, time, 1)
        return {
            "quantiles": quantiles,
            "pa_remaining": pa_remaining,
            "latent": latent,
            "seq_mask": seq_mask,
        }


class ROSSequenceForecaster:
    """Sklearn-style wrapper for ``ROSSequenceNetwork``."""

    def __init__(
        self, base_forecaster: MTLQuantileForecaster, config: dict | None = None
    ):
        self.base_forecaster = base_forecaster
        self.config = deepcopy(config or {})
        self.seed = int(self.config.get("seed", 42))

        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("training", {})
        loss_cfg = self.config.get("loss", {})
        self.encoder_dim = int(model_cfg.get("encoder_dim", 32))
        self.gru_hidden_dim = int(model_cfg.get("gru_hidden_dim", 128))
        self.dropout = float(model_cfg.get("dropout", 0.1))
        self.max_seq_len = int(model_cfg.get("max_seq_len", 32))
        self.batch_size = int(train_cfg.get("batch_size", 64))
        self.max_epochs = int(train_cfg.get("epochs", 50))
        self.learning_rate = float(train_cfg.get("learning_rate", 1e-3))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.early_stopping_patience = int(train_cfg.get("early_stopping_patience", 10))
        self.device_ = str(train_cfg.get("device", "cpu"))
        self.pa_weight = float(loss_cfg.get("pa_weight", 1.0))

        self.sequence_feature_names_: list[str] = list(SEQUENCE_FEATURE_NAMES)
        self.seq_group_dims_: dict[str, int] = {
            name: len(cols) for name, cols in SEQUENCE_FEATURE_GROUPS.items()
        }
        self.sequence_scaler_: StandardScaler | None = None
        self.sequence_min_: np.ndarray | None = None
        self.sequence_max_: np.ndarray | None = None
        self.network_: ROSSequenceNetwork | None = None
        self.loss_fn_: MultiTaskQuantileLoss | None = None
        self.training_history_: list[dict] = []
        self.is_fitted_: bool = False

    def _set_seeds(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _build_network(self) -> ROSSequenceNetwork:
        net = ROSSequenceNetwork(
            phase2_network=deepcopy(self.base_forecaster.network_),
            seq_group_dims=self.seq_group_dims_,
            encoder_dim=self.encoder_dim,
            gru_hidden_dim=self.gru_hidden_dim,
            dropout=self.dropout,
        )
        return net.to(torch.device(self.device_))

    def _scale_phase2(self, phase2_features: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_arr = _to_float64_array(phase2_features)
        return scale_and_clamp(
            X_arr,
            self.base_forecaster.feature_scaler_,
            self.base_forecaster.feature_min_,
            self.base_forecaster.feature_max_,
        ).astype(np.float32)

    def _sequence_features_scaled(
        self, snapshots: pd.DataFrame, fit: bool
    ) -> pd.DataFrame:
        raw = compute_weekly_sequence_features(snapshots).reindex(
            columns=self.sequence_feature_names_
        )
        raw = raw.astype(float).copy()
        all_nan = raw.isna().all(axis=0)
        if all_nan.any():
            raw.loc[:, all_nan] = 0.0
        arr = _to_float64_array(raw)
        if fit:
            self.sequence_scaler_ = StandardScaler()
            scaled = self.sequence_scaler_.fit_transform(arr)
            scaled = np.nan_to_num(scaled, nan=0.0)
            self.sequence_min_ = scaled.min(axis=0)
            self.sequence_max_ = scaled.max(axis=0)
        else:
            scaled = self.sequence_scaler_.transform(arr)
            scaled = np.nan_to_num(scaled, nan=0.0)
            scaled = np.clip(scaled, self.sequence_min_, self.sequence_max_)
        return pd.DataFrame(
            scaled, columns=self.sequence_feature_names_, index=snapshots.index
        )

    def _make_dataset(
        self,
        snapshots: pd.DataFrame,
        phase2_features: pd.DataFrame | np.ndarray,
        targets: pd.DataFrame | np.ndarray | None,
        pa_target: pd.Series | np.ndarray | None,
        sample_weights: np.ndarray | None,
        fit_sequence_scaler: bool,
    ) -> ROSCutoffSequenceDataset:
        seq = self._sequence_features_scaled(snapshots, fit=fit_sequence_scaler)
        phase2_scaled = pd.DataFrame(
            self._scale_phase2(phase2_features),
            columns=list(self.base_forecaster.feature_names_),
            index=snapshots.index,
        )
        return ROSCutoffSequenceDataset(
            snapshots=snapshots,
            sequence_features=seq,
            phase2_features=phase2_scaled,
            targets=targets,
            pa_target=pa_target,
            sample_weights=sample_weights,
            max_seq_len=self.max_seq_len,
            sequence_feature_cols=self.sequence_feature_names_,
        )

    @staticmethod
    def _last_valid(
        outputs: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = mask.sum(dim=1).clamp(min=1).long()
        idx = lengths - 1
        batch_idx = torch.arange(mask.shape[0], device=mask.device)
        return (
            outputs["quantiles"][batch_idx, idx],
            outputs["pa_remaining"][batch_idx, idx],
        )

    def fit(
        self,
        snapshots: pd.DataFrame,
        phase2_features: pd.DataFrame | np.ndarray,
        targets: pd.DataFrame | np.ndarray,
        pa_target: pd.Series | np.ndarray,
        sample_weights: np.ndarray | None = None,
        eval_set: tuple[
            pd.DataFrame,
            pd.DataFrame | np.ndarray,
            pd.DataFrame | np.ndarray,
            pd.Series | np.ndarray,
        ]
        | None = None,
    ) -> "ROSSequenceForecaster":
        self._set_seeds()
        if sample_weights is None:
            sample_weights = default_sequence_sample_weights(snapshots)

        y_arr = _to_float64_array(targets)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        y_scaled = self.base_forecaster.target_scaler_.transform(y_arr)
        pa_arr = np.asarray(pa_target, dtype=np.float64).reshape(-1, 1)
        pa_scaled = self.base_forecaster.pa_scaler_.transform(pa_arr)

        train_ds = self._make_dataset(
            snapshots.reset_index(drop=True),
            phase2_features,
            y_scaled,
            pa_scaled.reshape(-1),
            sample_weights,
            fit_sequence_scaler=True,
        )
        gen = torch.Generator().manual_seed(self.seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            generator=gen,
        )

        val_loader: DataLoader | None = None
        if eval_set is not None:
            val_rows, val_phase2, val_targets, val_pa = eval_set
            val_y = _to_float64_array(val_targets)
            if val_y.ndim == 1:
                val_y = val_y.reshape(-1, 1)
            val_y_scaled = self.base_forecaster.target_scaler_.transform(val_y)
            val_pa_scaled = self.base_forecaster.pa_scaler_.transform(
                np.asarray(val_pa, dtype=np.float64).reshape(-1, 1)
            )
            val_ds = self._make_dataset(
                val_rows.reset_index(drop=True),
                val_phase2,
                val_y_scaled,
                val_pa_scaled.reshape(-1),
                None,
                fit_sequence_scaler=False,
            )
            val_loader = DataLoader(val_ds, batch_size=max(1, len(val_ds)))

        self.network_ = self._build_network()
        self.loss_fn_ = MultiTaskQuantileLoss(
            n_tasks=self.base_forecaster.n_targets,
            taus=tuple(self.base_forecaster.taus),
            pa_loss=self.base_forecaster.pa_loss_type,
            pa_weight=self.pa_weight,
        ).to(torch.device(self.device_))

        params = [
            p
            for p in list(self.network_.parameters()) + list(self.loss_fn_.parameters())
            if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )

        best_val = float("inf")
        best_state: dict | None = None
        best_loss_state: dict | None = None
        patience = 0
        self.training_history_ = []
        for epoch in range(self.max_epochs):
            train_loss = self._run_epoch(train_loader, optimizer=optimizer)
            record = {"epoch": epoch + 1, "train_loss": train_loss}
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                record["val_loss"] = val_loss
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = deepcopy(self.network_.state_dict())
                    best_loss_state = deepcopy(self.loss_fn_.state_dict())
                    patience = 0
                else:
                    patience += 1
                if patience >= self.early_stopping_patience:
                    break
            self.training_history_.append(record)

        if best_state is not None:
            self.network_.load_state_dict(best_state)
        if best_loss_state is not None:
            self.loss_fn_.load_state_dict(best_loss_state)
        self.network_.eval()
        self.is_fitted_ = True
        return self

    def _run_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.network_.train()
        self.network_.phase2_network.eval()
        device = next(self.network_.parameters()).device
        total = 0.0
        batches = 0
        for batch in loader:
            optimizer.zero_grad()
            seq = batch["seq"].to(device)
            phase2_x = batch["phase2_x"].to(device)
            mask = batch["seq_mask"].to(device)
            blend = batch["blend_features"].to(device)
            target = batch["target"].to(device)
            pa_target = batch["pa_target"].to(device)
            weights = batch["sample_weight"].to(device)
            outputs = self.network_(seq, phase2_x, mask, blend)
            q_last, pa_last = self._last_valid(outputs, mask)
            loss, _ = self.loss_fn_(
                q_last,
                pa_last,
                target,
                pa_target,
                sample_weights=weights,
            )
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1
        return total / max(batches, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        self.network_.eval()
        device = next(self.network_.parameters()).device
        losses: list[float] = []
        for batch in loader:
            seq = batch["seq"].to(device)
            phase2_x = batch["phase2_x"].to(device)
            mask = batch["seq_mask"].to(device)
            blend = batch["blend_features"].to(device)
            target = batch["target"].to(device)
            pa_target = batch["pa_target"].to(device)
            outputs = self.network_(seq, phase2_x, mask, blend)
            q_last, pa_last = self._last_valid(outputs, mask)
            loss, _ = self.loss_fn_(q_last, pa_last, target, pa_target)
            losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else float("inf")

    @torch.no_grad()
    def predict(
        self,
        snapshots: pd.DataFrame,
        phase2_features: pd.DataFrame | np.ndarray,
    ) -> dict[str, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        pred_ds = self._make_dataset(
            snapshots.reset_index(drop=True),
            phase2_features,
            targets=None,
            pa_target=None,
            sample_weights=None,
            fit_sequence_scaler=False,
        )
        loader = DataLoader(pred_ds, batch_size=self.batch_size, shuffle=False)
        device = next(self.network_.parameters()).device
        q_batches: list[np.ndarray] = []
        pa_batches: list[np.ndarray] = []
        self.network_.eval()
        for batch in loader:
            outputs = self.network_(
                batch["seq"].to(device),
                batch["phase2_x"].to(device),
                batch["seq_mask"].to(device),
                batch["blend_features"].to(device),
            )
            q_last, pa_last = self._last_valid(outputs, batch["seq_mask"].to(device))
            q_batches.append(q_last.cpu().numpy())
            pa_batches.append(pa_last.cpu().numpy())

        q_scaled = np.concatenate(q_batches, axis=0)
        pa_scaled = np.concatenate(pa_batches, axis=0)
        scale = self.base_forecaster.target_scaler_.scale_.reshape(1, -1, 1)
        mean = self.base_forecaster.target_scaler_.mean_.reshape(1, -1, 1)
        q_original = np.sort(q_scaled * scale + mean, axis=-1)
        pa_original = self.base_forecaster.pa_scaler_.inverse_transform(pa_scaled)
        return {"quantiles": q_original, "pa_remaining": pa_original}

    def save(self, path: str | Path) -> Path:
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted ROSSequenceForecaster.")
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.base_forecaster.save(out_dir / "base_phase2")
        state = {
            "network_state_dict": self.network_.state_dict(),
            "loss_state_dict": self.loss_fn_.state_dict(),
            "sequence_scaler_mean": torch.from_numpy(self.sequence_scaler_.mean_),
            "sequence_scaler_scale": torch.from_numpy(self.sequence_scaler_.scale_),
            "sequence_scaler_var": torch.from_numpy(self.sequence_scaler_.var_),
            "sequence_scaler_n_samples": torch.as_tensor(
                self.sequence_scaler_.n_samples_seen_
            ),
            "sequence_min": torch.from_numpy(self.sequence_min_),
            "sequence_max": torch.from_numpy(self.sequence_max_),
            "sequence_feature_names": self.sequence_feature_names_,
            "seq_group_dims": self.seq_group_dims_,
            "config": self.config,
            "training_history": self.training_history_,
        }
        torch.save(state, out_dir / "ros_sequence_forecaster.pt")
        with open(out_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2, default=str)
        return out_dir

    @classmethod
    def load(cls, path: str | Path) -> "ROSSequenceForecaster":
        load_dir = Path(path)
        base = MTLQuantileForecaster.load(load_dir / "base_phase2")
        state = torch.load(
            load_dir / "ros_sequence_forecaster.pt",
            map_location="cpu",
            weights_only=False,
        )
        instance = cls(base, state.get("config", {}))
        instance.sequence_feature_names_ = list(state["sequence_feature_names"])
        instance.seq_group_dims_ = dict(state["seq_group_dims"])
        instance.sequence_scaler_ = reconstruct_scaler(
            state["sequence_scaler_mean"],
            state["sequence_scaler_scale"],
            state["sequence_scaler_var"],
            state["sequence_scaler_n_samples"],
        )
        instance.sequence_min_ = state["sequence_min"].numpy()
        instance.sequence_max_ = state["sequence_max"].numpy()
        instance.network_ = instance._build_network()
        instance.network_.load_state_dict(state["network_state_dict"])
        instance.network_.eval()
        instance.loss_fn_ = MultiTaskQuantileLoss(
            n_tasks=instance.base_forecaster.n_targets,
            taus=tuple(instance.base_forecaster.taus),
            pa_loss=instance.base_forecaster.pa_loss_type,
            pa_weight=instance.pa_weight,
        )
        instance.loss_fn_.load_state_dict(state["loss_state_dict"])
        instance.training_history_ = state.get("training_history", [])
        instance.is_fitted_ = True
        return instance


class ROSSequenceEnsembleForecaster:
    """Multi-seed ensemble wrapper for Phase 3."""

    def __init__(
        self, base_forecaster: MTLQuantileForecaster, config: dict | None = None
    ):
        self.base_forecaster = base_forecaster
        self.config = deepcopy(config or {})
        ens_cfg = self.config.get("ensemble", {})
        self.n_seeds = int(ens_cfg.get("n_seeds", 1))
        self.base_seed = int(ens_cfg.get("base_seed", self.config.get("seed", 42)))
        self.forecasters_: list[ROSSequenceForecaster] = []
        self.is_fitted_: bool = False

    def fit(
        self,
        snapshots: pd.DataFrame,
        phase2_features: pd.DataFrame | np.ndarray,
        targets: pd.DataFrame | np.ndarray,
        pa_target: pd.Series | np.ndarray,
        sample_weights: np.ndarray | None = None,
        eval_set: tuple[
            pd.DataFrame,
            pd.DataFrame | np.ndarray,
            pd.DataFrame | np.ndarray,
            pd.Series | np.ndarray,
        ]
        | None = None,
    ) -> "ROSSequenceEnsembleForecaster":
        self.forecasters_ = []
        for i in range(self.n_seeds):
            cfg = deepcopy(self.config)
            cfg["seed"] = self.base_seed + i
            forecaster = ROSSequenceForecaster(deepcopy(self.base_forecaster), cfg)
            forecaster.fit(
                snapshots,
                phase2_features,
                targets,
                pa_target,
                sample_weights=sample_weights,
                eval_set=eval_set,
            )
            self.forecasters_.append(forecaster)
        self.is_fitted_ = True
        return self

    def predict(
        self,
        snapshots: pd.DataFrame,
        phase2_features: pd.DataFrame | np.ndarray,
    ) -> dict[str, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("Ensemble is not fitted.")
        preds = [m.predict(snapshots, phase2_features) for m in self.forecasters_]
        q = np.sort(
            np.stack([p["quantiles"] for p in preds], axis=0).mean(axis=0), axis=-1
        )
        pa = np.stack([p["pa_remaining"] for p in preds], axis=0).mean(axis=0)
        return {"quantiles": q, "pa_remaining": pa}

    def save(self, path: str | Path) -> Path:
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "ensemble_meta.json", "w") as f:
            json.dump(
                {"n_seeds": self.n_seeds, "base_seed": self.base_seed},
                f,
                indent=2,
            )
        for i, forecaster in enumerate(self.forecasters_):
            forecaster.save(out_dir / f"seed_{i}")
        return out_dir

    @classmethod
    def load(cls, path: str | Path) -> "ROSSequenceEnsembleForecaster":
        load_dir = Path(path)
        with open(load_dir / "ensemble_meta.json") as f:
            meta = json.load(f)
        first = ROSSequenceForecaster.load(load_dir / "seed_0")
        instance = cls(first.base_forecaster, first.config)
        instance.n_seeds = int(meta["n_seeds"])
        instance.base_seed = int(meta["base_seed"])
        instance.forecasters_ = [first]
        for i in range(1, instance.n_seeds):
            instance.forecasters_.append(
                ROSSequenceForecaster.load(load_dir / f"seed_{i}")
            )
        instance.is_fitted_ = True
        return instance
