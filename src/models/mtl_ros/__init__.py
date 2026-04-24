"""MTL ROS: quantile-head MTL with a PA-remaining auxiliary head.

Phase 2 of the rest-of-season projection effort.  The preseason MTL in
``src.models.mtl`` is a point-estimate regressor.  This module parallels
it for in-season use, replacing each per-target point head with ``n_quantiles``
pinball-loss quantile heads and adding a 7th point-estimate head for
remaining-PA regression.

Nothing in this package modifies or depends on ``src.models.mtl`` internals
beyond importing the reusable ``ResidualBlock`` building block.
"""

from __future__ import annotations

from src.models.mtl_ros.loss import MultiTaskQuantileLoss
from src.models.mtl_ros.model import (
    MTLQuantileEnsembleForecaster,
    MTLQuantileForecaster,
    MTLQuantileNetwork,
)
from src.models.mtl_ros.train import train_ros

__all__ = [
    "MultiTaskQuantileLoss",
    "MTLQuantileNetwork",
    "MTLQuantileForecaster",
    "MTLQuantileEnsembleForecaster",
    "train_ros",
]
