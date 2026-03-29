"""Tests for non-contact feature integration.

Covers:
- Non-contact regressed rate computation
- Feature registry and pipeline integration
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.non_contact import (
    LEAGUE_AVG,
    STABILISATION_PA,
    compute_non_contact_features,
    regress_to_mean,
)
from src.features.batting import compute_batting_features
from src.features.pipeline import build_features, extract_xy
from src.features.registry import (
    ALL_FEATURES,
    BATTING_FEATURES,
    NON_CONTACT_FEATURES,
    TEMPORAL_FEATURES,
    FeatureGroup,
    get_feature_names,
)
from src.features.temporal import compute_temporal_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_merged_row(
    mlbam_id: int = 1000,
    season: int = 2023,
    pa: int = 500,
    **kwargs,
) -> dict:
    """Build a single row of synthetic merged data."""
    row = {
        "mlbam_id": mlbam_id,
        "idfg": mlbam_id // 10,
        "name": f"Player_{mlbam_id}",
        "team": kwargs.get("team", "NYY"),
        "season": season,
        "age": kwargs.get("age", 28),
        "g": 150,
        "pa": pa,
        "ab": int(pa * 0.88),
        "h": int(pa * 0.22),
        "bb": kwargs.get("bb", int(pa * 0.08)),
        "ibb": 4,
        "so": kwargs.get("so", int(pa * 0.22)),
        "hbp": kwargs.get("hbp", int(pa * 0.01)),
        "sf": 5,
        "sh": 0,
        "hr": kwargs.get("hr", 20),
        "r": kwargs.get("r", 80),
        "rbi": kwargs.get("rbi", 75),
        "sb": kwargs.get("sb", 10),
        "cs": kwargs.get("cs", 3),
        "avg": kwargs.get("avg", 0.260),
        "obp": kwargs.get("obp", 0.330),
        "slg": kwargs.get("slg", 0.420),
        "ops": 0.750,
        "babip": 0.300,
        "woba": 0.320,
        "wrc_plus": 110,
        "war": 3.0,
        # Statcast
        "avg_exit_velocity": 89.5,
        "ev_p95": 106.0,
        "max_exit_velocity": 112.0,
        "avg_launch_angle": 12.5,
        "barrel_rate": 0.08,
        "hard_hit_rate": 0.38,
        "sweet_spot_rate": 0.34,
        # Speed
        "sprint_speed": 27.5,
        "avg_bat_speed": 72.0,
        "avg_swing_speed": 70.0,
        "has_avg_bat_speed": 1,
        "has_avg_swing_speed": 1,
        # Context
        "park_factor_runs": 1.0,
        "park_factor_hr": 1.0,
        "team_runs_per_game": 4.5,
        "team_ops": 0.720,
        "team_sb": 80,
        # Targets
        "target_obp": np.nan,
        "target_slg": np.nan,
        "target_hr": np.nan,
        "target_r": np.nan,
        "target_rbi": np.nan,
        "target_sb": np.nan,
    }
    return row


def _make_multi_season_df(
    player_id: int = 1000,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Build multi-season data."""
    if seasons is None:
        seasons = list(range(2020, 2026))
    rows = []
    for i, year in enumerate(seasons):
        rows.append(_make_merged_row(
            mlbam_id=player_id,
            season=year,
            hr=20 + i * 2,
            r=80 + i * 3,
            rbi=75 + i * 2,
            sb=10 - i,
            obp=0.330 + i * 0.005,
            slg=0.420 + i * 0.010,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: Non-contact regressed rate computation
# ---------------------------------------------------------------------------


class TestNonContactFeatures:
    """Verify stabilisation-point regression to the mean."""

    def test_regress_to_mean_known_values(self):
        """regress_to_mean matches hand-computed example."""
        # observed=0.30, pa=500, k_rate stab=60
        # regressed = (500*0.30 + 60*0.224) / (500+60)
        expected = (500 * 0.30 + 60 * 0.224) / 560
        result = regress_to_mean(0.30, 500, "k_rate")
        assert abs(result - expected) < 1e-6

    def test_regress_to_mean_zero_pa(self):
        """With PA=0, regressed rate equals league average."""
        result = regress_to_mean(0.50, 0, "k_rate")
        assert abs(result - LEAGUE_AVG["k_rate"]) < 1e-6

    def test_regress_to_mean_large_pa(self):
        """With very large PA, regressed rate is close to observed."""
        result = regress_to_mean(0.30, 10000, "bb_rate")
        assert abs(result - 0.30) < 0.005

    def test_compute_non_contact_features_columns(self):
        """compute_non_contact_features adds 3 regressed rate columns."""
        df = pd.DataFrame([_make_merged_row()])
        df = compute_batting_features(df)
        result = compute_non_contact_features(df)
        assert "regressed_k_rate" in result.columns
        assert "regressed_bb_rate" in result.columns
        assert "regressed_hbp_rate" in result.columns

    def test_regressed_rates_between_observed_and_league(self):
        """Regressed rates fall between observed and league average."""
        df = pd.DataFrame([_make_merged_row(pa=500, so=150, bb=40, hbp=5)])
        df = compute_batting_features(df)
        result = compute_non_contact_features(df)
        row = result.iloc[0]

        # k_rate = 150/500 = 0.30, which is > league avg 0.224
        # regressed should be between 0.224 and 0.30
        assert LEAGUE_AVG["k_rate"] <= row["regressed_k_rate"] <= row["k_rate"]

    def test_does_not_modify_original(self):
        """compute_non_contact_features returns a copy."""
        df = pd.DataFrame([_make_merged_row()])
        df = compute_batting_features(df)
        original_cols = set(df.columns)
        _ = compute_non_contact_features(df)
        assert set(df.columns) == original_cols


# ---------------------------------------------------------------------------
# Tests: Registry and pipeline integration
# ---------------------------------------------------------------------------


class TestRegistryAndPipeline:
    """Verify feature registry counts and pipeline integration."""

    def test_batting_features_count(self):
        """28 batting features (21 original + 7 new hit-type/walk decomposition rates)."""
        assert len(BATTING_FEATURES) == 28

    def test_non_contact_features_count(self):
        """6 non-contact features (3 original + regressed_babip + regressed_iso + regressed_hr_per_bbe)."""
        assert len(NON_CONTACT_FEATURES) == 6

    def test_temporal_features_count(self):
        """45 temporal features: 18 target + 9 xstat + 18 per-PA rate (incl ISO)."""
        assert len(TEMPORAL_FEATURES) == 45

    def test_total_feature_count(self):
        """Registry expands over time; should include at least original footprint."""
        assert len(ALL_FEATURES) >= 79

    def test_all_features_have_unique_names(self):
        """No duplicate feature names in the registry."""
        names = [f.name for f in ALL_FEATURES]
        assert len(names) == len(set(names)), f"Duplicates: {[n for n in names if names.count(n) > 1]}"

    def test_non_contact_group_toggle(self):
        """Disabling non_contact excludes its features."""
        enabled = {
            "batting": True, "statcast": True,
            "non_contact": False, "sprint_speed": True, "bat_speed": True,
            "park_factors": True, "team_stats": True, "age": True, "temporal": True,
        }
        names = get_feature_names(enabled)
        for f in NON_CONTACT_FEATURES:
            assert f.name not in names

    def test_pipeline_builds_non_contact_features(self):
        """build_features produces non-contact regressed rate columns."""
        df = _make_multi_season_df()
        result = build_features(df)
        assert "regressed_k_rate" in result.columns
        assert "regressed_bb_rate" in result.columns
        assert "regressed_hbp_rate" in result.columns
