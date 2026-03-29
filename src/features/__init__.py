"""Feature engineering for MLB batter stat predictions.

Modules
-------
batting
    Derived batting features (walk rate, K rate, ISO, SB rate).
statcast
    Statcast quality feature validation and imputation.
context
    Context features (age, park factors, team stats).
temporal
    Multi-year temporal features (prev year, weighted avg, trend).
pipeline
    End-to-end feature pipeline orchestration.
registry
    Feature name registry and metadata.
"""
from src.features.batting import compute_batting_features
from src.features.context import compute_context_features
from src.features.pipeline import build_features, extract_xy
from src.features.registry import (
    ALL_FEATURES,
    TARGET_COLUMNS,
    FeatureGroup,
    FeatureMeta,
    FeatureType,
    get_feature_names,
)
from src.features.statcast import compute_statcast_features
from src.features.temporal import compute_temporal_features

__all__ = [
    "compute_batting_features",
    "compute_statcast_features",
    "compute_context_features",
    "compute_temporal_features",
    "build_features",
    "extract_xy",
    "get_feature_names",
    "ALL_FEATURES",
    "TARGET_COLUMNS",
    "FeatureGroup",
    "FeatureMeta",
    "FeatureType",
]
