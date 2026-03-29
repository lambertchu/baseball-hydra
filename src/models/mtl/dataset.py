"""PyTorch Dataset for batter stat prediction.

Wraps feature and target matrices as float32 tensors.  NaN values in
features are replaced with 0.0 (matching the Ridge regression pipeline's
convention of filling NaN with 0 after StandardScaler normalisation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BatterDataset(Dataset):
    """Dataset of (features, targets[, weights]) for one player-season per row.

    Parameters
    ----------
    X:
        Feature matrix ``(n_samples, n_features)``.  NaN values are
        replaced with 0.0.
    y:
        Target matrix ``(n_samples, n_targets)``.  Pass ``None`` for
        inference-only mode (predict without targets).
    sample_weights:
        Optional per-sample weights ``(n_samples,)`` for loss weighting.
        When provided, ``__getitem__`` returns a 3-tuple
        ``(features, targets, weight)``.
    """

    def __init__(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.DataFrame | None = None,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        X_arr = np.asarray(X, dtype=np.float32)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        self.X = torch.from_numpy(X_arr)

        if y is not None:
            y_arr = np.asarray(y, dtype=np.float32)
            self.y: torch.Tensor | None = torch.from_numpy(y_arr)
        else:
            self.y = None

        if sample_weights is not None:
            w_arr = np.asarray(sample_weights, dtype=np.float32)
            self.w = torch.from_numpy(w_arr)
            self.has_nontrivial_weights = True
        else:
            self.w = torch.ones(len(self.X))
            self.has_nontrivial_weights = False

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        """Return ``(features, targets, weight)`` or just ``features``.

        During training (y is not None), always returns a 3-tuple.
        The weight defaults to 1.0 if no sample_weights were provided.
        """
        if self.y is not None:
            return self.X[idx], self.y[idx], self.w[idx]
        return self.X[idx]
