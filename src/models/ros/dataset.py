"""Cutoff sequence dataset for the Phase 3 GRU ROS model."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.eval.ros_metrics import ROS_RATE_TARGETS
from src.models.mtl_ros.dataset import compute_sample_weights
from src.models.ros.features import BLEND_FEATURE_NAMES, SEQUENCE_FEATURE_NAMES

_KEY_COLS: tuple[str, ...] = ("mlbam_id", "season")
_SORT_COLS: tuple[str, ...] = ("mlbam_id", "season", "iso_year", "iso_week")
_ROW_KEY_COLS: tuple[str, ...] = ("mlbam_id", "season", "iso_year", "iso_week")


class ROSCutoffSequenceDataset(Dataset):
    """One sample per cutoff row, containing the trajectory up to that cutoff."""

    def __init__(
        self,
        snapshots: pd.DataFrame,
        sequence_features: pd.DataFrame,
        phase2_features: pd.DataFrame,
        targets: pd.DataFrame | np.ndarray | None = None,
        pa_target: pd.Series | np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
        max_seq_len: int = 32,
        sequence_feature_cols: Sequence[str] = SEQUENCE_FEATURE_NAMES,
        blend_feature_cols: Sequence[str] = BLEND_FEATURE_NAMES,
    ) -> None:
        if len(snapshots) != len(sequence_features) or len(snapshots) != len(
            phase2_features
        ):
            raise ValueError("snapshots, sequence_features, and phase2_features align")
        missing = [
            c for c in (*_ROW_KEY_COLS, *_KEY_COLS) if c not in snapshots.columns
        ]
        if missing:
            raise KeyError(f"snapshots missing key columns: {missing}")

        self.snapshots = snapshots.reset_index(drop=True).copy()
        self.sequence_features = (
            sequence_features.reset_index(drop=True)
            .reindex(columns=list(sequence_feature_cols))
            .astype("float32")
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=True)
        )
        self.phase2_features = (
            phase2_features.reset_index(drop=True)
            .astype("float32")
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=True)
        )
        self.max_seq_len = int(max_seq_len)
        self.sequence_feature_cols = list(sequence_feature_cols)
        self.blend_feature_cols = list(blend_feature_cols)
        self.blend_indices = [
            self.sequence_feature_cols.index(c)
            for c in self.blend_feature_cols
            if c in self.sequence_feature_cols
        ]

        if targets is None:
            y_arr = np.zeros(
                (len(self.snapshots), len(ROS_RATE_TARGETS)), dtype="float32"
            )
        else:
            y_arr = np.asarray(targets, dtype=np.float32)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        self.targets = y_arr

        if pa_target is None:
            pa_arr = np.zeros((len(self.snapshots), 1), dtype="float32")
        else:
            pa_arr = np.asarray(pa_target, dtype=np.float32).reshape(-1, 1)
        self.pa_target = pa_arr

        if sample_weights is None:
            weights = np.ones(len(self.snapshots), dtype=np.float32)
        else:
            weights = np.asarray(sample_weights, dtype=np.float32)
        self.sample_weights = weights

        self.row_keys: list[tuple[int, int, int, int]] = [
            tuple(int(v) for v in row)
            for row in self.snapshots[list(_ROW_KEY_COLS)].to_numpy()
        ]

        self._positions_by_group, self._position_rank = self._build_positions_by_group()

    def _build_positions_by_group(
        self,
    ) -> tuple[dict[tuple[int, int], np.ndarray], np.ndarray]:
        ordered = self.snapshots.reset_index(names="__pos").sort_values(
            list(_SORT_COLS)
        )
        out: dict[tuple[int, int], np.ndarray] = {}
        rank = np.zeros(len(self.snapshots), dtype=np.int32)
        for key, frame in ordered.groupby(list(_KEY_COLS), sort=False):
            positions = frame["__pos"].to_numpy(dtype=np.int64, copy=True)
            out[tuple(int(v) for v in key)] = positions
            rank[positions] = np.arange(len(positions), dtype=np.int32)
        return out, rank

    def _history_positions(self, idx: int) -> np.ndarray:
        mlbam_id, season, _, _ = self.row_keys[idx]
        key = (mlbam_id, season)
        positions = self._positions_by_group[key]
        rank = int(self._position_rank[idx])
        start = max(0, rank + 1 - self.max_seq_len)
        return positions[start : rank + 1]

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        hist = self._history_positions(idx)
        seq = np.zeros(
            (self.max_seq_len, len(self.sequence_feature_cols)),
            dtype=np.float32,
        )
        mask = np.zeros(self.max_seq_len, dtype=bool)
        hist_arr = self.sequence_features[hist]
        n = len(hist_arr)
        seq[:n] = hist_arr
        mask[:n] = True
        if self.blend_indices:
            blend = seq[:, self.blend_indices]
        else:
            blend = np.zeros((self.max_seq_len, 2), dtype=np.float32)
        if blend.shape[1] == 1:
            blend = np.concatenate(
                [blend, np.zeros((self.max_seq_len, 1), dtype=np.float32)],
                axis=1,
            )

        return {
            "seq": torch.from_numpy(seq),
            "seq_mask": torch.from_numpy(mask),
            "blend_features": torch.from_numpy(blend.astype(np.float32)),
            "phase2_x": torch.from_numpy(
                self.phase2_features[idx].copy()
            ),
            "target": torch.from_numpy(self.targets[idx]),
            "pa_target": torch.from_numpy(self.pa_target[idx]),
            "sample_weight": torch.tensor(
                self.sample_weights[idx], dtype=torch.float32
            ),
        }


def default_sequence_sample_weights(
    snapshots: pd.DataFrame,
    recency_lambda: float = 0.30,
) -> np.ndarray:
    """Use the Phase 2 recency × sqrt(ros_pa+1) weighting for cutoffs."""
    return compute_sample_weights(
        snapshots,
        recency_lambda=recency_lambda,
        ros_pa_col="ros_pa",
        season_col="season",
    )
