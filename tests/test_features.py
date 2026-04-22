"""Tests for the baseball-hydra feature engineering pipeline.

Covers:
- Batting feature computation (bb_rate, k_rate, iso, sb_rate)
- Statcast feature validation and imputation
- Context feature validation (age, park factors, team stats)
- Temporal feature computation (prev_year, weighted_avg, trend)
- No temporal leakage in temporal features
- Missing data handling (rookies, gap years, absent columns)
- End-to-end pipeline integration
- Feature registry completeness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.batting import compute_batting_features
from src.features.context import compute_context_features
from src.features.pipeline import build_features, extract_xy
from src.features.registry import (
    ALL_FEATURES,
    BATTING_FEATURES,
    STATCAST_FEATURES,
    TARGET_COLUMNS,
    TEMPORAL_FEATURES,
    FeatureGroup,
    FeatureType,
    get_feature_names,
    get_feature_metadata,
)
from src.features.statcast import compute_statcast_features
from src.features.temporal import compute_temporal_features


# ---------------------------------------------------------------------------
# Helpers: build synthetic merged data
# ---------------------------------------------------------------------------


def _make_merged_row(
    mlbam_id: int = 1000,
    season: int = 2023,
    pa: int = 500,
    bb: int = 40,
    so: int = 110,
    avg: float = 0.260,
    obp: float = 0.330,
    slg: float = 0.420,
    hr: int = 20,
    r: int = 80,
    rbi: int = 75,
    sb: int = 10,
    cs: int = 3,
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
        "g": kwargs.get("g", 150),
        "pa": pa,
        "ab": int(pa * 0.88),
        "h": int(pa * 0.22),
        "bb": bb,
        "ibb": 4,
        "so": so,
        "hbp": 5,
        "sf": 5,
        "sh": 0,
        "doubles": kwargs.get("doubles", 25),
        "triples": kwargs.get("triples", 3),
        "hr": hr,
        "r": r,
        "rbi": rbi,
        "sb": sb,
        "cs": cs,
        "avg": avg,
        "obp": obp,
        "slg": slg,
        "ops": obp + slg - avg,
        "babip": kwargs.get("babip", 0.300),
        "woba": kwargs.get("woba", 0.320),
        "wrc_plus": kwargs.get("wrc_plus", 110),
        "war": kwargs.get("war", 3.0),
        # Statcast
        "avg_exit_velocity": kwargs.get("avg_exit_velocity", 89.5),
        "ev_p95": kwargs.get("ev_p95", 106.0),
        "max_exit_velocity": kwargs.get("max_exit_velocity", 112.0),
        "avg_launch_angle": kwargs.get("avg_launch_angle", 12.5),
        "barrel_rate": kwargs.get("barrel_rate", 0.08),
        "hard_hit_rate": kwargs.get("hard_hit_rate", 0.38),
        "sweet_spot_rate": kwargs.get("sweet_spot_rate", 0.34),
        # Speed
        "sprint_speed": kwargs.get("sprint_speed", 27.5),
        "avg_bat_speed": kwargs.get("avg_bat_speed", 72.0),
        "avg_swing_speed": kwargs.get("avg_swing_speed", 70.0),
        "has_avg_bat_speed": kwargs.get("has_avg_bat_speed", 1),
        "has_avg_swing_speed": kwargs.get("has_avg_swing_speed", 1),
        # Context
        "park_factor_runs": kwargs.get("park_factor_runs", 1.0),
        "park_factor_hr": kwargs.get("park_factor_hr", 1.0),
        "team_runs_per_game": kwargs.get("team_runs_per_game", 4.5),
        "team_ops": kwargs.get("team_ops", 0.720),
        "team_sb": kwargs.get("team_sb", 80),
        "team_sb_per_game": kwargs.get("team_sb_per_game", 0.494),
    }
    # Targets (NaN for latest season)
    row["target_obp"] = kwargs.get("target_obp", np.nan)
    row["target_slg"] = kwargs.get("target_slg", np.nan)
    row["target_hr"] = kwargs.get("target_hr", np.nan)
    row["target_r"] = kwargs.get("target_r", np.nan)
    row["target_rbi"] = kwargs.get("target_rbi", np.nan)
    row["target_sb"] = kwargs.get("target_sb", np.nan)
    return row


def _make_multi_season_df(
    player_id: int = 1000,
    seasons: list[int] | None = None,
    base_hr: int = 20,
    base_obp: float = 0.330,
    base_slg: float = 0.420,
) -> pd.DataFrame:
    """Build multi-season data for a single player with trends."""
    if seasons is None:
        seasons = list(range(2020, 2026))

    rows = []
    for i, year in enumerate(seasons):
        rows.append(
            _make_merged_row(
                mlbam_id=player_id,
                season=year,
                hr=base_hr + i * 2,
                r=80 + i * 3,
                rbi=75 + i * 2,
                sb=10 - i,
                obp=base_obp + i * 0.005,
                slg=base_slg + i * 0.010,
            )
        )
    return pd.DataFrame(rows)


@pytest.fixture
def single_season_df() -> pd.DataFrame:
    """Single player, single season."""
    return pd.DataFrame([_make_merged_row()])


@pytest.fixture
def multi_season_df() -> pd.DataFrame:
    """Single player across 2020-2025 (6 seasons)."""
    return _make_multi_season_df()


@pytest.fixture
def multi_player_df() -> pd.DataFrame:
    """Three players across 2020-2025, each with different stats."""
    frames = []
    for pid, base_hr, base_obp in [
        (1000, 20, 0.330),
        (2000, 30, 0.370),
        (3000, 10, 0.290),
    ]:
        frames.append(
            _make_multi_season_df(
                player_id=pid,
                base_hr=base_hr,
                base_obp=base_obp,
            )
        )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Tests: Batting features
# ---------------------------------------------------------------------------


class TestBattingFeatures:
    """Verify derived batting feature computation."""

    def test_walk_rate_computed_correctly(self, single_season_df):
        """bb_rate = BB / PA."""
        result = compute_batting_features(single_season_df)
        row = result.iloc[0]
        expected = row["bb"] / row["pa"]
        assert abs(row["bb_rate"] - expected) < 1e-6

    def test_strikeout_rate_computed_correctly(self, single_season_df):
        """k_rate = SO / PA."""
        result = compute_batting_features(single_season_df)
        row = result.iloc[0]
        expected = row["so"] / row["pa"]
        assert abs(row["k_rate"] - expected) < 1e-6

    def test_iso_computed_correctly(self, single_season_df):
        """iso = SLG - AVG."""
        result = compute_batting_features(single_season_df)
        row = result.iloc[0]
        expected = row["slg"] - row["avg"]
        assert abs(row["iso"] - expected) < 1e-6

    def test_sb_rate_computed_correctly(self):
        """sb_rate = SB / (SB + CS)."""
        df = pd.DataFrame([_make_merged_row(sb=20, cs=5)])
        result = compute_batting_features(df)
        assert abs(result.iloc[0]["sb_rate"] - 0.8) < 1e-6  # 20 / 25

    def test_sb_rate_zero_when_no_attempts(self):
        """sb_rate is 0 when SB + CS = 0 (no stolen base attempts)."""
        df = pd.DataFrame([_make_merged_row(sb=0, cs=0)])
        result = compute_batting_features(df)
        assert result.iloc[0]["sb_rate"] == 0.0

    def test_bb_rate_zero_when_pa_is_zero(self):
        """bb_rate is 0 when PA = 0 (edge case)."""
        df = pd.DataFrame([_make_merged_row(pa=0, bb=0, so=0)])
        result = compute_batting_features(df)
        assert result.iloc[0]["bb_rate"] == 0.0
        assert result.iloc[0]["k_rate"] == 0.0

    def test_sb_attempt_rate_computed_correctly(self):
        """sb_attempt_rate = (SB + CS) / PA."""
        df = pd.DataFrame([_make_merged_row(sb=15, cs=5, pa=500)])
        result = compute_batting_features(df)
        expected = (15 + 5) / 500
        assert abs(result.iloc[0]["sb_attempt_rate"] - expected) < 1e-6

    def test_sb_attempt_rate_zero_when_pa_zero(self):
        """sb_attempt_rate is 0 when PA = 0 (division by zero guard)."""
        df = pd.DataFrame([_make_merged_row(pa=0, bb=0, so=0, sb=0, cs=0)])
        result = compute_batting_features(df)
        assert result.iloc[0]["sb_attempt_rate"] == 0.0

    @pytest.mark.parametrize("feature,numerator_cols,expected_fn", [
        ("ibb_rate", ["ibb"], lambda r: r["ibb"] / r["pa"]),
        ("ubb_rate", ["bb", "ibb"], lambda r: (r["bb"] - r["ibb"]) / r["pa"]),
        ("doubles_rate", ["doubles"], lambda r: r["doubles"] / r["pa"]),
        ("triples_rate", ["triples"], lambda r: r["triples"] / r["pa"]),
        ("cs_rate", ["cs"], lambda r: r["cs"] / r["pa"]),
    ])
    def test_new_rate_features_computed(self, feature, numerator_cols, expected_fn):
        """New per-PA rate features are computed correctly."""
        df = pd.DataFrame([_make_merged_row()])
        result = compute_batting_features(df)
        row = result.iloc[0]
        assert abs(row[feature] - expected_fn(row)) < 1e-6

    def test_singles_rate_computed(self):
        """singles_rate = (H - 2B - 3B - HR) / PA."""
        df = pd.DataFrame([_make_merged_row(pa=500, hr=20, doubles=25, triples=3)])
        result = compute_batting_features(df)
        row = result.iloc[0]
        singles = row["h"] - 25 - 3 - 20
        expected = max(0, singles) / 500
        assert abs(row["singles_rate"] - expected) < 1e-6

    def test_extra_base_rate_computed(self):
        """extra_base_rate = (2B + 3B + HR) / PA."""
        df = pd.DataFrame([_make_merged_row(pa=500, hr=20, doubles=25, triples=3)])
        result = compute_batting_features(df)
        expected = (25 + 3 + 20) / 500
        assert abs(result.iloc[0]["extra_base_rate"] - expected) < 1e-6

    def test_new_rate_features_zero_when_pa_zero(self):
        """All new rate features are 0 when PA = 0."""
        df = pd.DataFrame([_make_merged_row(pa=0, bb=0, so=0, sb=0, cs=0, hr=0)])
        result = compute_batting_features(df)
        for feat in ("ibb_rate", "ubb_rate", "singles_rate", "doubles_rate",
                     "triples_rate", "extra_base_rate", "cs_rate"):
            assert result.iloc[0][feat] == 0.0, f"{feat} should be 0 when PA=0"

    def test_ubb_rate_clips_negative(self):
        """ubb_rate clips to 0 if IBB > BB (data error)."""
        df = pd.DataFrame([_make_merged_row(bb=2, pa=500)])
        df["ibb"] = 5  # More IBB than total BB — should clip
        result = compute_batting_features(df)
        assert result.iloc[0]["ubb_rate"] == 0.0

    def test_does_not_modify_original(self, single_season_df):
        """compute_batting_features returns a copy, not a mutation."""
        original_cols = set(single_season_df.columns)
        _ = compute_batting_features(single_season_df)
        assert set(single_season_df.columns) == original_cols


# ---------------------------------------------------------------------------
# Tests: Statcast features
# ---------------------------------------------------------------------------


class TestStatcastFeatures:
    """Verify Statcast feature validation and imputation."""

    def test_all_statcast_columns_present(self, single_season_df):
        """All 7 Statcast features are present after computation."""
        result = compute_statcast_features(single_season_df)
        for col in [
            "avg_exit_velocity",
            "ev_p95",
            "max_exit_velocity",
            "avg_launch_angle",
            "barrel_rate",
            "hard_hit_rate",
            "sweet_spot_rate",
        ]:
            assert col in result.columns

    def test_missing_statcast_filled_with_defaults(self):
        """If a Statcast column is entirely missing, fill with default."""
        df = pd.DataFrame([_make_merged_row()])
        df = df.drop(columns=["avg_exit_velocity"])
        result = compute_statcast_features(df)
        assert "avg_exit_velocity" in result.columns
        assert result.iloc[0]["avg_exit_velocity"] == 88.0  # default

    def test_nan_statcast_imputed_with_median(self):
        """NaN values in existing Statcast columns are filled with median."""
        rows = [
            _make_merged_row(mlbam_id=1, season=2023, avg_exit_velocity=90.0),
            _make_merged_row(mlbam_id=2, season=2023, avg_exit_velocity=88.0),
            _make_merged_row(mlbam_id=3, season=2023),
        ]
        df = pd.DataFrame(rows)
        df.loc[2, "avg_exit_velocity"] = np.nan

        result = compute_statcast_features(df)
        # Median of [90.0, 88.0] = 89.0
        assert abs(result.iloc[2]["avg_exit_velocity"] - 89.0) < 1e-6

    def test_no_nans_after_statcast_imputation(self, single_season_df):
        """No NaN values remain in Statcast columns after processing."""
        result = compute_statcast_features(single_season_df)
        from src.features.statcast import STATCAST_COLS

        for col in STATCAST_COLS:
            assert result[col].isna().sum() == 0


# ---------------------------------------------------------------------------
# Tests: Context features
# ---------------------------------------------------------------------------


class TestContextFeatures:
    """Verify context feature validation and defaults."""

    def test_age_squared_computed(self, single_season_df):
        """age_squared = age^2."""
        result = compute_context_features(single_season_df)
        row = result.iloc[0]
        assert abs(row["age_squared"] - row["age"] ** 2) < 1e-6

    def test_missing_age_column_filled(self):
        """If age column is absent, default to 28."""
        df = pd.DataFrame([_make_merged_row()])
        df = df.drop(columns=["age"])
        if "age_squared" in df.columns:
            df = df.drop(columns=["age_squared"])
        result = compute_context_features(df)
        assert result.iloc[0]["age"] == 28.0
        assert abs(result.iloc[0]["age_squared"] - 784.0) < 1e-6

    def test_missing_park_factors_neutral(self):
        """Missing park factors default to 1.0 (neutral)."""
        df = pd.DataFrame([_make_merged_row()])
        df = df.drop(columns=["park_factor_runs", "park_factor_hr"])
        result = compute_context_features(df)
        assert result.iloc[0]["park_factor_runs"] == 1.0
        assert result.iloc[0]["park_factor_hr"] == 1.0

    def test_missing_team_stats_defaults(self):
        """Missing team stats get league-average defaults."""
        df = pd.DataFrame([_make_merged_row()])
        df = df.drop(columns=["team_runs_per_game", "team_ops", "team_sb"])
        result = compute_context_features(df)
        assert result.iloc[0]["team_runs_per_game"] == 4.5
        assert result.iloc[0]["team_ops"] == 0.720
        assert result.iloc[0]["team_sb"] == 80.0

    def test_nan_park_factors_filled(self):
        """NaN park factors filled with neutral 1.0."""
        df = pd.DataFrame([_make_merged_row()])
        df.loc[0, "park_factor_runs"] = np.nan
        result = compute_context_features(df)
        assert result.iloc[0]["park_factor_runs"] == 1.0

    def test_aging_curve_features_present(self, single_season_df):
        """All 3 aging curve features exist after compute_context_features()."""
        result = compute_context_features(single_season_df)
        assert "age_delta_speed" in result.columns
        assert "age_delta_power" in result.columns
        assert "age_delta_patience" in result.columns

    def test_aging_speed_peak_at_23(self):
        """age_delta_speed = 0.0 at age 23 (peak age)."""
        df = pd.DataFrame([_make_merged_row(age=23)])
        result = compute_context_features(df)
        assert result.iloc[0]["age_delta_speed"] == 0.0

    def test_aging_power_peak_near_27_30(self):
        """age_delta_power is at maximum (plateau) between ages 27-30."""
        plateau_value = 0.025  # 0.005 * (27 - 22) = 0.025
        for age in [27, 28, 29, 30]:
            df = pd.DataFrame([_make_merged_row(age=age)])
            result = compute_context_features(df)
            assert (
                abs(result.iloc[0]["age_delta_power"] - plateau_value) < 1e-6
            ), f"age_delta_power at age {age} should be {plateau_value}"

    def test_aging_patience_peak_near_30_33(self):
        """age_delta_patience is at maximum (plateau) between ages 30-33."""
        plateau_value = 0.018  # 0.003 * (30 - 24) = 0.018
        for age in [30, 31, 32, 33]:
            df = pd.DataFrame([_make_merged_row(age=age)])
            result = compute_context_features(df)
            assert (
                abs(result.iloc[0]["age_delta_patience"] - plateau_value) < 1e-6
            ), f"age_delta_patience at age {age} should be {plateau_value}"

    def test_aging_features_decline_after_peak(self):
        """All aging curve values decrease after their respective peak ages."""
        # Speed: should be strictly decreasing after age 23
        ages_speed = [24, 27, 30, 35]
        speed_values = []
        for age in ages_speed:
            df = pd.DataFrame([_make_merged_row(age=age)])
            result = compute_context_features(df)
            speed_values.append(result.iloc[0]["age_delta_speed"])
        for i in range(len(speed_values) - 1):
            assert speed_values[i] > speed_values[i + 1], (
                f"age_delta_speed should decline: age {ages_speed[i]} ({speed_values[i]}) "
                f"not > age {ages_speed[i+1]} ({speed_values[i+1]})"
            )

        # Power: should decline after age 30
        df_30 = pd.DataFrame([_make_merged_row(age=30)])
        df_35 = pd.DataFrame([_make_merged_row(age=35)])
        power_30 = compute_context_features(df_30).iloc[0]["age_delta_power"]
        power_35 = compute_context_features(df_35).iloc[0]["age_delta_power"]
        assert (
            power_30 > power_35
        ), f"age_delta_power should decline after 30: {power_30} > {power_35}"

        # Patience: should decline after age 33
        df_33 = pd.DataFrame([_make_merged_row(age=33)])
        df_38 = pd.DataFrame([_make_merged_row(age=38)])
        patience_33 = compute_context_features(df_33).iloc[0]["age_delta_patience"]
        patience_38 = compute_context_features(df_38).iloc[0]["age_delta_patience"]
        assert (
            patience_33 > patience_38
        ), f"age_delta_patience should decline after 33: {patience_33} > {patience_38}"

    def test_aging_features_no_nan(self):
        """No NaN values for any reasonable age range (18-45)."""
        ages = list(range(18, 46))
        rows = [_make_merged_row(age=a) for a in ages]
        df = pd.DataFrame(rows)
        result = compute_context_features(df)
        for col in ["age_delta_speed", "age_delta_power", "age_delta_patience"]:
            assert result[col].isna().sum() == 0, f"NaN found in {col}"

    def test_sb_era_x_attempt_rate_post_2023(self):
        """sb_era_x_attempt_rate = sb_attempt_rate for post-2023 seasons."""
        df = pd.DataFrame([_make_merged_row(season=2024)])
        df = compute_batting_features(df)  # compute sb_attempt_rate first
        result = compute_context_features(df)
        assert abs(result.iloc[0]["sb_era_x_attempt_rate"] - result.iloc[0]["sb_attempt_rate"]) < 1e-6

    def test_sb_era_x_attempt_rate_pre_2023(self):
        """sb_era_x_attempt_rate = 0 for pre-2023 seasons."""
        df = pd.DataFrame([_make_merged_row(season=2022)])
        df = compute_batting_features(df)
        result = compute_context_features(df)
        assert result.iloc[0]["sb_era_x_attempt_rate"] == 0.0

    def test_sb_era_x_attempt_rate_missing_column(self):
        """sb_era_x_attempt_rate = 0 when sb_attempt_rate column missing."""
        df = pd.DataFrame([_make_merged_row(season=2024)])
        # Don't compute batting features — sb_attempt_rate won't exist
        result = compute_context_features(df)
        assert result.iloc[0]["sb_era_x_attempt_rate"] == 0.0


# ---------------------------------------------------------------------------
# Tests: Temporal features
# ---------------------------------------------------------------------------


class TestTemporalFeatures:
    """Verify temporal feature computation and no-leakage guarantees."""

    def test_prev_year_is_lag1(self, multi_season_df):
        """prev_year_hr for season Y equals the player's HR from season Y-1."""
        result = compute_temporal_features(multi_season_df)

        # 2021's prev_year_hr should be 2020's HR
        row_2020 = multi_season_df[multi_season_df["season"] == 2020].iloc[0]
        row_2021 = result[result["season"] == 2021].iloc[0]
        assert row_2021["prev_year_hr"] == row_2020["hr"]

    def test_prev_year_nan_for_first_season(self, multi_season_df):
        """First season has no prior year data → prev_year is NaN."""
        result = compute_temporal_features(multi_season_df)
        row_2020 = result[result["season"] == 2020].iloc[0]
        assert pd.isna(row_2020["prev_year_hr"])
        assert pd.isna(row_2020["prev_year_obp"])

    def test_weighted_avg_with_three_years(self, multi_season_df):
        """weighted_avg uses weights [5, 3, 2] for Y-1, Y-2, Y-3."""
        result = compute_temporal_features(multi_season_df, weights=[5, 3, 2])

        # For season 2023 (index 3), lags are 2022, 2021, 2020
        hr_2022 = multi_season_df[multi_season_df["season"] == 2022].iloc[0]["hr"]
        hr_2021 = multi_season_df[multi_season_df["season"] == 2021].iloc[0]["hr"]
        hr_2020 = multi_season_df[multi_season_df["season"] == 2020].iloc[0]["hr"]

        expected = (5 * hr_2022 + 3 * hr_2021 + 2 * hr_2020) / (5 + 3 + 2)
        row_2023 = result[result["season"] == 2023].iloc[0]
        assert abs(row_2023["weighted_avg_hr"] - expected) < 1e-6

    def test_weighted_avg_with_one_year_only(self, multi_season_df):
        """weighted_avg with only Y-1 available equals just Y-1 value."""
        result = compute_temporal_features(multi_season_df, weights=[5, 3, 2])

        # Season 2021 has only lag=1 (2020). Lags 2 and 3 are missing.
        hr_2020 = multi_season_df[multi_season_df["season"] == 2020].iloc[0]["hr"]
        row_2021 = result[result["season"] == 2021].iloc[0]
        # weighted_avg = (5 * hr_2020) / 5 = hr_2020
        assert abs(row_2021["weighted_avg_hr"] - hr_2020) < 1e-6

    def test_weighted_avg_with_two_years(self, multi_season_df):
        """weighted_avg with Y-1 and Y-2 uses adjusted weights."""
        result = compute_temporal_features(multi_season_df, weights=[5, 3, 2])

        # Season 2022 has lag1=2021, lag2=2020, lag3 missing
        hr_2021 = multi_season_df[multi_season_df["season"] == 2021].iloc[0]["hr"]
        hr_2020 = multi_season_df[multi_season_df["season"] == 2020].iloc[0]["hr"]
        expected = (5 * hr_2021 + 3 * hr_2020) / (5 + 3)
        row_2022 = result[result["season"] == 2022].iloc[0]
        assert abs(row_2022["weighted_avg_hr"] - expected) < 1e-6

    def test_trend_is_lag1_minus_lag2(self, multi_season_df):
        """trend_hr = HR(Y-1) - HR(Y-2)."""
        result = compute_temporal_features(multi_season_df)

        hr_2022 = multi_season_df[multi_season_df["season"] == 2022].iloc[0]["hr"]
        hr_2021 = multi_season_df[multi_season_df["season"] == 2021].iloc[0]["hr"]
        row_2023 = result[result["season"] == 2023].iloc[0]
        assert abs(row_2023["trend_hr"] - (hr_2022 - hr_2021)) < 1e-6

    def test_trend_nan_when_insufficient_history(self, multi_season_df):
        """trend is NaN when either Y-1 or Y-2 is unavailable."""
        result = compute_temporal_features(multi_season_df)

        # 2020: no Y-1 at all → trend is NaN
        row_2020 = result[result["season"] == 2020].iloc[0]
        assert pd.isna(row_2020["trend_hr"])

        # 2021: has Y-1 but not Y-2 → trend is NaN
        row_2021 = result[result["season"] == 2021].iloc[0]
        assert pd.isna(row_2021["trend_hr"])

    def test_gap_year_nulls_prev_year(self):
        """If a player skips a season, prev_year is NaN for the return year."""
        rows = [
            _make_merged_row(mlbam_id=1000, season=2020, hr=20),
            # Player skips 2021
            _make_merged_row(mlbam_id=1000, season=2022, hr=25),
            _make_merged_row(mlbam_id=1000, season=2023, hr=28),
        ]
        df = pd.DataFrame(rows)
        result = compute_temporal_features(df)

        # 2022: prev year should be NaN (2021 is missing, not consecutive)
        row_2022 = result[result["season"] == 2022].iloc[0]
        assert pd.isna(row_2022["prev_year_hr"])

        # 2023: prev year should be 2022's HR (consecutive)
        row_2023 = result[result["season"] == 2023].iloc[0]
        assert row_2023["prev_year_hr"] == 25

    def test_gap_year_nulls_weighted_avg(self):
        """Gap years break the lookback chain for weighted averages."""
        rows = [
            _make_merged_row(mlbam_id=1000, season=2020, hr=20),
            # Player skips 2021
            _make_merged_row(mlbam_id=1000, season=2022, hr=25),
            _make_merged_row(mlbam_id=1000, season=2023, hr=28),
        ]
        df = pd.DataFrame(rows)
        result = compute_temporal_features(df)

        # 2022: no valid lags (gap) → weighted_avg is NaN
        row_2022 = result[result["season"] == 2022].iloc[0]
        assert pd.isna(row_2022["weighted_avg_hr"])

        # 2023: only lag1 (2022) is valid → weighted_avg = 25
        row_2023 = result[result["season"] == 2023].iloc[0]
        assert abs(row_2023["weighted_avg_hr"] - 25.0) < 1e-6

    def test_different_players_isolated(self, multi_player_df):
        """Temporal features don't leak between different players."""
        result = compute_temporal_features(multi_player_df)

        # Player 1's prev_year should come from Player 1 only
        p1_2021 = result[
            (result["mlbam_id"] == 1000) & (result["season"] == 2021)
        ].iloc[0]
        p1_2020_hr = multi_player_df[
            (multi_player_df["mlbam_id"] == 1000) & (multi_player_df["season"] == 2020)
        ].iloc[0]["hr"]
        assert p1_2021["prev_year_hr"] == p1_2020_hr

        # Player 2's prev_year should come from Player 2 only
        p2_2021 = result[
            (result["mlbam_id"] == 2000) & (result["season"] == 2021)
        ].iloc[0]
        p2_2020_hr = multi_player_df[
            (multi_player_df["mlbam_id"] == 2000) & (multi_player_df["season"] == 2020)
        ].iloc[0]["hr"]
        assert p2_2021["prev_year_hr"] == p2_2020_hr

        # Cross-check: player 1 and 2 have different HR values
        assert p1_2021["prev_year_hr"] != p2_2021["prev_year_hr"]

    def test_no_future_data_in_temporal_features(self, multi_season_df):
        """Temporal features NEVER use data from the current season or later.

        For a row at season Y, prev_year, weighted_avg, and trend must only
        use data from Y-1 and earlier. The stat values from the current row
        (season Y) should NOT appear in the temporal features for that row.
        """
        result = compute_temporal_features(multi_season_df)

        for _, row in result.iterrows():
            season = row["season"]
            current_hr = row["hr"]

            # prev_year_hr must NOT equal current season's HR
            # (unless by coincidence the values happen to be equal)
            if pd.notna(row["prev_year_hr"]):
                # The prev_year should correspond to a different season
                prev_row = multi_season_df[
                    (multi_season_df["mlbam_id"] == row["mlbam_id"])
                    & (multi_season_df["season"] == season - 1)
                ]
                if not prev_row.empty:
                    assert row["prev_year_hr"] == prev_row.iloc[0]["hr"]

    def test_temporal_features_for_rate_stats(self, multi_season_df):
        """Rate stats (OBP, SLG) also get temporal features."""
        result = compute_temporal_features(multi_season_df)

        # Check that OBP temporal features exist
        assert "prev_year_obp" in result.columns
        assert "weighted_avg_obp" in result.columns
        assert "trend_obp" in result.columns

        # Verify prev_year_obp for season 2021
        obp_2020 = multi_season_df[multi_season_df["season"] == 2020].iloc[0]["obp"]
        row_2021 = result[result["season"] == 2021].iloc[0]
        assert abs(row_2021["prev_year_obp"] - obp_2020) < 1e-6

    def test_all_six_target_stats_get_temporal_features(self, multi_season_df):
        """All 6 target stats get prev_year, weighted_avg, and trend columns."""
        result = compute_temporal_features(multi_season_df)
        for stat in ["obp", "slg", "hr", "r", "rbi", "sb"]:
            assert f"prev_year_{stat}" in result.columns
            assert f"weighted_avg_{stat}" in result.columns
            assert f"trend_{stat}" in result.columns


# ---------------------------------------------------------------------------
# Tests: Feature registry
# ---------------------------------------------------------------------------


class TestFeatureRegistry:
    """Verify the feature registry is complete and consistent."""

    def test_all_features_have_unique_names(self):
        """No duplicate feature names in the registry."""
        names = [f.name for f in ALL_FEATURES]
        assert len(names) == len(
            set(names)
        ), f"Duplicates: {[n for n in names if names.count(n) > 1]}"

    def test_temporal_features_cover_all_targets(self):
        """Temporal features exist for all 6 target stats."""
        temporal_names = {f.name for f in TEMPORAL_FEATURES}
        for stat in ["obp", "slg", "hr", "r", "rbi", "sb"]:
            assert f"prev_year_{stat}" in temporal_names
            assert f"weighted_avg_{stat}" in temporal_names
            assert f"trend_{stat}" in temporal_names

    def test_get_feature_names_returns_all_default_on_groups_when_no_filter(self):
        """get_feature_names with no filter returns every default-on feature.

        Opt-in groups (``in_season`` as of Phase 2) are excluded until
        explicitly toggled on to keep the preseason MTL's feature matrix
        unchanged when its config omits the key.
        """
        all_names = get_feature_names()
        from src.features.registry import FeatureGroup, IN_SEASON_FEATURES
        non_optin = [f for f in ALL_FEATURES if f.group != FeatureGroup.IN_SEASON]
        assert len(all_names) == len(non_optin)
        for f in IN_SEASON_FEATURES:
            assert f.name not in all_names

    def test_get_feature_names_respects_disabled_groups(self):
        """Disabling a group excludes its features."""
        enabled = {
            "batting": False,
            "statcast": True,
            "temporal": False,
            "sprint_speed": True,
            "bat_speed": True,
            "park_factors": True,
            "team_stats": True,
            "age": True,
        }
        names = get_feature_names(enabled)
        for f in BATTING_FEATURES:
            assert f.name not in names
        for f in STATCAST_FEATURES:
            assert f.name in names

    def test_target_columns_fixed_order(self):
        """Target columns are in the fixed order from CLAUDE.md."""
        assert TARGET_COLUMNS == [
            "target_obp",
            "target_slg",
            "target_hr",
            "target_r",
            "target_rbi",
            "target_sb",
        ]

    def test_get_feature_metadata_returns_correct_info(self):
        """get_feature_metadata looks up feature by name."""
        meta = get_feature_metadata("bb_rate")
        assert meta is not None
        assert meta.group == FeatureGroup.BATTING
        assert meta.ftype == FeatureType.RATE

    def test_get_feature_metadata_returns_none_for_unknown(self):
        """Unknown feature name returns None."""
        assert get_feature_metadata("nonexistent_feature") is None

    def test_new_batting_features_registered(self):
        """All 7 new batting features are in the registry."""
        names = {f.name for f in ALL_FEATURES}
        for feat in ("ibb_rate", "ubb_rate", "singles_rate", "doubles_rate",
                     "triples_rate", "extra_base_rate", "cs_rate"):
            assert feat in names, f"{feat} not in ALL_FEATURES"

    def test_new_context_features_registered(self):
        """team_sb_per_game and sb_era_x_attempt_rate are in the registry."""
        names = {f.name for f in ALL_FEATURES}
        assert "team_sb_per_game" in names
        assert "sb_era_x_attempt_rate" in names

    def test_iso_temporal_features_registered(self):
        """ISO gets temporal features via RATE_DECOMP_STATS."""
        temporal_names = {f.name for f in TEMPORAL_FEATURES}
        assert "prev_year_iso" in temporal_names
        assert "weighted_avg_iso" in temporal_names
        assert "trend_iso" in temporal_names

    def test_batting_features_count(self):
        """28 batting features (21 original + 7 new: ibb_rate, ubb_rate, singles_rate, doubles_rate, triples_rate, extra_base_rate, cs_rate)."""
        assert len(BATTING_FEATURES) == 28

    def test_statcast_features_count(self):
        """Statcast registry includes base + expected-contact + indicators."""
        assert len(STATCAST_FEATURES) >= 7
        names = {f.name for f in STATCAST_FEATURES}
        assert "avg_exit_velocity" in names
        assert "estimated_woba_using_speedangle" in names
        assert "has_avg_exit_velocity" in names

    def test_temporal_features_count(self):
        """45 temporal features: 18 target + 9 xstat + 18 per-PA rate (incl ISO)."""
        assert len(TEMPORAL_FEATURES) == 45


# ---------------------------------------------------------------------------
# Tests: End-to-end pipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    """Verify the end-to-end feature pipeline integration."""

    def test_build_features_adds_all_columns(self, multi_season_df):
        """build_features produces all expected derived columns."""
        result = build_features(multi_season_df)

        # Batting derived
        assert "bb_rate" in result.columns
        assert "k_rate" in result.columns
        assert "iso" in result.columns
        assert "sb_rate" in result.columns

        # Temporal
        assert "prev_year_hr" in result.columns
        assert "weighted_avg_hr" in result.columns
        assert "trend_hr" in result.columns

        # Context
        assert "age_squared" in result.columns

    def test_extract_xy_correct_shapes(self, multi_player_df):
        """extract_xy produces X and y with correct dimensions."""
        featured = build_features(multi_player_df)
        X, y = extract_xy(featured)

        assert X.shape[0] == len(multi_player_df)
        assert y.shape[0] == len(multi_player_df)
        assert y.shape[1] == 6  # 6 target stats

    def test_extract_xy_no_target_columns_in_X(self, multi_player_df):
        """Feature matrix X must not contain target columns."""
        featured = build_features(multi_player_df)
        X, _ = extract_xy(featured)

        for col in TARGET_COLUMNS:
            assert col not in X.columns

    def test_extract_xy_no_id_columns_in_X(self, multi_player_df):
        """Feature matrix X must not contain ID/metadata columns."""
        featured = build_features(multi_player_df)
        X, _ = extract_xy(featured)

        for col in ["mlbam_id", "idfg", "name", "team", "season"]:
            assert col not in X.columns

    def test_pipeline_config_driven_group_selection(self, multi_season_df):
        """Config can disable feature groups."""
        config = {
            "feature_groups": {
                "batting": True,
                "statcast": True,
                "temporal": False,
                "sprint_speed": True,
                "bat_speed": True,
                "park_factors": True,
                "team_stats": True,
                "age": True,
            },
            "temporal_weights": [5, 3, 2],
        }
        result = build_features(multi_season_df, config)

        # Temporal features should NOT be in the result because the
        # temporal computation was skipped
        assert "prev_year_hr" not in result.columns

    def test_extract_xy_with_disabled_groups(self, multi_season_df):
        """extract_xy respects disabled groups in config."""
        config = {
            "feature_groups": {
                "batting": True,
                "statcast": False,
                "temporal": True,
                "sprint_speed": True,
                "bat_speed": True,
                "park_factors": True,
                "team_stats": True,
                "age": True,
            },
            "temporal_weights": [5, 3, 2],
        }
        featured = build_features(multi_season_df, config)
        X, y = extract_xy(featured, config)

        # Statcast features should not be in X
        assert "avg_exit_velocity" not in X.columns
        assert "barrel_rate" not in X.columns

    def test_pipeline_preserves_row_count(self, multi_player_df):
        """Pipeline does not add or remove rows."""
        result = build_features(multi_player_df)
        assert len(result) == len(multi_player_df)

    def test_pipeline_with_default_config(self, multi_season_df):
        """Pipeline works with no config (all defaults)."""
        result = build_features(multi_season_df)
        X, y = extract_xy(result)
        assert X.shape[0] > 0
        assert y.shape[0] > 0

    def test_pipeline_no_temporal_leakage_in_features(self, multi_season_df):
        """After full pipeline, temporal features only look backward.

        For season Y, weighted_avg_hr should be computable from Y-1, Y-2, Y-3
        only — never from Y or later.
        """
        result = build_features(multi_season_df)

        # For the latest season (2025), there should be valid temporal
        # features because it has 5 prior seasons
        row_2025 = result[result["season"] == 2025].iloc[0]
        assert pd.notna(row_2025["prev_year_hr"])
        assert pd.notna(row_2025["weighted_avg_hr"])

        # prev_year for 2025 should be 2024's HR
        hr_2024 = multi_season_df[multi_season_df["season"] == 2024].iloc[0]["hr"]
        assert row_2025["prev_year_hr"] == hr_2024

    def test_extract_xy_converts_nullable_dtypes(self, multi_player_df):
        """extract_xy converts nullable pandas dtypes to standard float64."""
        featured = build_features(multi_player_df)
        # Inject a nullable Float64 column to simulate parquet behavior
        featured["avg_exit_velocity"] = featured["avg_exit_velocity"].astype("Float64")
        X, y = extract_xy(featured)

        for col in X.columns:
            assert X[col].dtype == np.float64, f"{col} has dtype {X[col].dtype}"
        for col in y.columns:
            assert y[col].dtype == np.float64, f"{col} has dtype {y[col].dtype}"

        # Must be convertible to numpy without error
        X_arr = np.asarray(X, dtype=np.float64)
        assert X_arr.shape == X.shape
