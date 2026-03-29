"""Tests for the baseball-hydra data pipeline.

Covers:
- Data loading and concatenation
- FanGraphs batting stat normalization
- Statcast aggregation correctness
- ID mapping between FanGraphs and MLBAM
- Merge logic across data sources
- Target alignment (Y features → Y+1 targets, no temporal leakage)
- Chronological split correctness
- Missing data handling (bat speed imputation, sprint speed fallback)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from unittest.mock import patch

from src.data.merge import (
    align_targets,
    merge_batting_with_statcast,
    merge_context_data,
    merge_speed_data,
)
from src.data.splits import SplitConfig, split_data, get_production_data
from src.data.fetch_statcast import _aggregate_batter_statcast, fetch_statcast


# ---------------------------------------------------------------------------
# Fixtures: synthetic data builders
# ---------------------------------------------------------------------------


def make_batting(
    players: list[dict],
) -> pd.DataFrame:
    """Build synthetic FanGraphs batting data.

    Each dict in ``players`` must have at minimum: idfg, season, pa.
    Note: mlbam_id is NOT included here — real FanGraphs data uses idfg only.
    The mlbam_id is added during the merge step via the id_map table.
    """
    rows = []
    for p in players:
        row = {
            "idfg": p["idfg"],
            "name": p.get("name", f"Player_{p['idfg']}"),
            "age": p.get("age", 28),
            "team": p.get("team", "NYY"),
            "season": p["season"],
            "g": p.get("g", 150),
            "pa": p["pa"],
            "ab": int(p["pa"] * 0.88),
            "h": int(p["pa"] * 0.22),
            "singles": int(p["pa"] * 0.14),
            "doubles": int(p["pa"] * 0.04),
            "triples": int(p["pa"] * 0.005),
            "hr": p.get("hr", int(p["pa"] * 0.04)),
            "r": p.get("r", int(p["pa"] * 0.12)),
            "rbi": p.get("rbi", int(p["pa"] * 0.12)),
            "bb": int(p["pa"] * 0.08),
            "ibb": int(p["pa"] * 0.005),
            "so": int(p["pa"] * 0.22),
            "hbp": int(p["pa"] * 0.01),
            "sf": int(p["pa"] * 0.01),
            "sh": 0,
            "sb": p.get("sb", int(p["pa"] * 0.02)),
            "cs": int(p["pa"] * 0.005),
            "avg": p.get("avg", 0.260),
            "obp": p.get("obp", 0.330),
            "slg": p.get("slg", 0.420),
            "ops": p.get("ops", 0.750),
            "babip": p.get("babip", 0.300),
            "woba": p.get("woba", 0.320),
            "wrc_plus": p.get("wrc_plus", 110),
            "war": p.get("war", 3.0),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def make_statcast_agg(
    players: list[dict],
) -> pd.DataFrame:
    """Build synthetic Statcast aggregated data."""
    rows = []
    for p in players:
        rows.append(
            {
                "mlbam_id": p["mlbam_id"],
                "season": p["season"],
                "bbe_count": p.get("bbe_count", 200),
                "avg_exit_velocity": p.get("avg_exit_velocity", 89.5),
                "ev_p95": p.get("ev_p95", 106.0),
                "max_exit_velocity": p.get("max_exit_velocity", 112.0),
                "avg_launch_angle": p.get("avg_launch_angle", 12.5),
                "barrel_rate": p.get("barrel_rate", 0.08),
                "hard_hit_rate": p.get("hard_hit_rate", 0.38),
                "sweet_spot_rate": p.get("sweet_spot_rate", 0.34),
            }
        )
    return pd.DataFrame(rows)


def make_sprint_speed(players: list[dict]) -> pd.DataFrame:
    """Build synthetic sprint speed data."""
    rows = []
    for p in players:
        rows.append(
            {
                "mlbam_id": p["mlbam_id"],
                "season": p["season"],
                "sprint_speed": p.get("sprint_speed", 27.0),
            }
        )
    return pd.DataFrame(rows)


def make_bat_speed(players: list[dict]) -> pd.DataFrame:
    """Build synthetic bat speed data."""
    rows = []
    for p in players:
        rows.append(
            {
                "mlbam_id": p["mlbam_id"],
                "season": p["season"],
                "avg_bat_speed": p.get("avg_bat_speed", 72.0),
                "avg_swing_speed": p.get("avg_swing_speed", 70.0),
            }
        )
    return pd.DataFrame(rows)


def make_id_map(players: list[dict]) -> pd.DataFrame:
    """Build synthetic ID mapping table."""
    seen = set()
    rows = []
    for p in players:
        if p["idfg"] not in seen:
            rows.append({"idfg": p["idfg"], "mlbam_id": p["mlbam_id"]})
            seen.add(p["idfg"])
    return pd.DataFrame(rows)


def make_park_factors(teams: list[str], season: int) -> pd.DataFrame:
    """Build synthetic park factors."""
    rows = []
    for team in teams:
        rows.append(
            {
                "team": team,
                "season": season,
                "park_factor_runs": 1.0,
                "park_factor_hr": 1.0,
            }
        )
    return pd.DataFrame(rows)


def make_team_batting(teams: list[str], season: int) -> pd.DataFrame:
    """Build synthetic team batting stats."""
    rows = []
    for team in teams:
        rows.append(
            {
                "team": team,
                "season": season,
                "team_runs_per_game": 4.5,
                "team_ops": 0.720,
                "team_sb": 80,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multi-season fixture for target alignment and split tests
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_season_data():
    """Three players across 2020-2025 seasons (6 years, enough for all splits)."""
    players = []
    for pid in [1, 2, 3]:
        for year in range(2020, 2026):
            players.append(
                {
                    "idfg": pid * 100,
                    "mlbam_id": pid * 1000,
                    "season": year,
                    "pa": 500,
                    "hr": 20 + pid + year - 2020,
                    "r": 70 + pid * 5,
                    "rbi": 65 + pid * 5,
                    "sb": 5 + pid * 2,
                    "obp": 0.320 + pid * 0.01,
                    "slg": 0.400 + pid * 0.02,
                    "team": "NYY",
                }
            )
    batting = make_batting(players)
    statcast = make_statcast_agg(players)
    id_map = make_id_map(players)
    return batting, statcast, id_map, players


# ---------------------------------------------------------------------------
# Tests: Statcast aggregation
# ---------------------------------------------------------------------------


class TestStatcastAggregation:
    """Verify per-batter Statcast metric aggregation from raw pitch data."""

    def test_aggregation_computes_correct_avg_ev(self):
        """Average exit velocity is the mean of launch_speed for a batter."""
        raw = pd.DataFrame(
            {
                "batter": [1, 1, 1, 2, 2],
                "launch_speed": [90.0, 100.0, 110.0, 85.0, 95.0],
                "launch_angle": [10.0, 15.0, 20.0, 5.0, 25.0],
                "bb_type": [
                    "line_drive",
                    "fly_ball",
                    "fly_ball",
                    "ground_ball",
                    "line_drive",
                ],
                "game_type": ["R", "R", "R", "R", "R"],
                "launch_speed_angle": [3, 4, 6, 1, 4],
            }
        )
        agg = _aggregate_batter_statcast(raw, min_bbe=2)
        row1 = agg[agg["mlbam_id"] == 1].iloc[0]
        assert abs(row1["avg_exit_velocity"] - 100.0) < 0.01

    def test_ev_p95_and_max_exit_velocity(self):
        """ev_p95 is the 95th percentile; max_exit_velocity is the absolute max."""
        # 20 BBE so 95th percentile = value at rank 19 (interpolated)
        speeds = [float(v) for v in range(81, 101)]  # 81, 82, ..., 100
        raw = pd.DataFrame(
            {
                "batter": [1] * 20,
                "launch_speed": speeds,
                "launch_angle": [12.0] * 20,
                "bb_type": ["line_drive"] * 20,
                "game_type": ["R"] * 20,
            }
        )
        agg = _aggregate_batter_statcast(raw, min_bbe=1)
        row = agg.iloc[0]

        expected_p95 = pd.Series(speeds).quantile(0.95)
        assert abs(row["ev_p95"] - expected_p95) < 0.01
        assert row["max_exit_velocity"] == 100.0  # absolute max

    def test_aggregation_respects_min_bbe(self):
        """Batters with fewer than min_bbe are excluded."""
        raw = pd.DataFrame(
            {
                "batter": [1, 2, 2, 2],
                "launch_speed": [90.0, 85.0, 95.0, 100.0],
                "launch_angle": [10.0, 5.0, 25.0, 15.0],
                "bb_type": ["line_drive", "ground_ball", "line_drive", "fly_ball"],
                "game_type": ["R", "R", "R", "R"],
            }
        )
        agg = _aggregate_batter_statcast(raw, min_bbe=3)
        assert len(agg) == 1
        assert agg.iloc[0]["mlbam_id"] == 2

    def test_barrel_rate_computation(self):
        """Barrel rate = count of launch_speed_angle==6 / total BBE."""
        raw = pd.DataFrame(
            {
                "batter": [1, 1, 1, 1, 1],
                "launch_speed": [90.0, 100.0, 110.0, 105.0, 95.0],
                "launch_angle": [10.0, 15.0, 25.0, 20.0, 12.0],
                "bb_type": ["line_drive"] * 5,
                "game_type": ["R"] * 5,
                "launch_speed_angle": [3, 4, 6, 6, 3],
            }
        )
        agg = _aggregate_batter_statcast(raw, min_bbe=1)
        assert abs(agg.iloc[0]["barrel_rate"] - 0.4) < 0.01  # 2/5

    def test_hard_hit_rate_computation(self):
        """Hard hit rate = count of launch_speed >= 95 / total BBE."""
        raw = pd.DataFrame(
            {
                "batter": [1, 1, 1, 1],
                "launch_speed": [80.0, 95.0, 100.0, 110.0],
                "launch_angle": [10.0, 15.0, 20.0, 25.0],
                "bb_type": ["ground_ball", "line_drive", "fly_ball", "fly_ball"],
                "game_type": ["R", "R", "R", "R"],
            }
        )
        agg = _aggregate_batter_statcast(raw, min_bbe=1)
        assert abs(agg.iloc[0]["hard_hit_rate"] - 0.75) < 0.01  # 3/4

    def test_sweet_spot_rate_computation(self):
        """Sweet spot rate = count of LA in [8, 32] / total BBE."""
        raw = pd.DataFrame(
            {
                "batter": [1, 1, 1, 1, 1],
                "launch_speed": [90.0] * 5,
                "launch_angle": [5.0, 10.0, 20.0, 32.0, 40.0],
                "bb_type": [
                    "ground_ball",
                    "line_drive",
                    "line_drive",
                    "fly_ball",
                    "popup",
                ],
                "game_type": ["R", "R", "R", "R", "R"],
            }
        )
        agg = _aggregate_batter_statcast(raw, min_bbe=1)
        assert abs(agg.iloc[0]["sweet_spot_rate"] - 0.6) < 0.01  # 3/5

    def test_filters_to_regular_season_only(self):
        """Only game_type == 'R' events are included."""
        raw = pd.DataFrame(
            {
                "batter": [1, 1, 1],
                "launch_speed": [90.0, 100.0, 110.0],
                "launch_angle": [10.0, 15.0, 20.0],
                "bb_type": ["line_drive", "fly_ball", "fly_ball"],
                "game_type": ["R", "R", "P"],  # One postseason event
            }
        )
        agg = _aggregate_batter_statcast(raw, min_bbe=1)
        assert agg.iloc[0]["bbe_count"] == 2

    def test_expected_contact_metrics_aggregated_when_available(self):
        """Expected speed-angle columns are aggregated to seasonal means."""
        raw = pd.DataFrame(
            {
                "batter": [1, 1, 1],
                "launch_speed": [90.0, 95.0, 100.0],
                "launch_angle": [10.0, 15.0, 20.0],
                "bb_type": ["line_drive", "fly_ball", "line_drive"],
                "game_type": ["R", "R", "R"],
                "estimated_woba_using_speedangle": [0.30, 0.35, 0.40],
                "estimated_ba_using_speedangle": [0.22, 0.26, 0.30],
                "estimated_slg_using_speedangle": [0.38, 0.45, 0.52],
            }
        )
        agg = _aggregate_batter_statcast(raw, min_bbe=1)
        row = agg.iloc[0]
        assert abs(row["estimated_woba_using_speedangle"] - 0.35) < 1e-6
        assert abs(row["estimated_ba_using_speedangle"] - 0.26) < 1e-6
        assert abs(row["estimated_slg_using_speedangle"] - 0.45) < 1e-6


# ---------------------------------------------------------------------------
# Tests: Merge logic
# ---------------------------------------------------------------------------


class TestMergeBattingStatcast:
    """Verify merging of FanGraphs and Statcast data."""

    def test_merge_on_mlbam_id_and_season(self):
        """Batting and Statcast data merge correctly on (mlbam_id, season)."""
        players = [
            {"idfg": 100, "mlbam_id": 1000, "season": 2023, "pa": 500},
            {"idfg": 200, "mlbam_id": 2000, "season": 2023, "pa": 450},
        ]
        batting = make_batting(players)
        statcast = make_statcast_agg(players)
        id_map = make_id_map(players)

        merged = merge_batting_with_statcast(batting, statcast, id_map)
        assert len(merged) == 2
        assert "avg_exit_velocity" in merged.columns
        assert "pa" in merged.columns

    def test_left_join_preserves_batters_without_statcast(self):
        """Batters with batting stats but no Statcast data are kept (with NaN)."""
        players_batting = [
            {"idfg": 100, "mlbam_id": 1000, "season": 2023, "pa": 500},
            {"idfg": 200, "mlbam_id": 2000, "season": 2023, "pa": 450},
        ]
        players_statcast = [
            {"mlbam_id": 1000, "season": 2023},
        ]
        batting = make_batting(players_batting)
        statcast = make_statcast_agg(players_statcast)
        id_map = make_id_map(players_batting)

        merged = merge_batting_with_statcast(batting, statcast, id_map)
        assert len(merged) == 2
        # Player 2 has NaN Statcast data
        row2 = merged[merged["mlbam_id"] == 2000].iloc[0]
        assert pd.isna(row2["avg_exit_velocity"])


class TestMergeSpeedData:
    """Verify sprint speed and bat speed merging."""

    def test_sprint_speed_merged(self):
        """Sprint speed is attached via left join."""
        base = pd.DataFrame(
            {
                "mlbam_id": [1000, 2000],
                "season": [2023, 2023],
            }
        )
        sprint = make_sprint_speed(
            [
                {"mlbam_id": 1000, "season": 2023, "sprint_speed": 28.5},
            ]
        )
        bat_speed = pd.DataFrame()

        result = merge_speed_data(base, sprint, bat_speed)
        assert result.loc[result["mlbam_id"] == 1000, "sprint_speed"].iloc[0] == 28.5
        # Player 2 gets imputed value (median of available data)
        assert pd.notna(result.loc[result["mlbam_id"] == 2000, "sprint_speed"].iloc[0])

    def test_bat_speed_imputed_when_missing(self):
        """Bat speed is imputed for years where data is unavailable."""
        base = pd.DataFrame(
            {
                "mlbam_id": [1000, 2000],
                "season": [2023, 2024],
            }
        )
        sprint = pd.DataFrame()
        bat_speed = make_bat_speed(
            [
                {"mlbam_id": 2000, "season": 2024, "avg_bat_speed": 73.5},
            ]
        )

        result = merge_speed_data(
            base, sprint, bat_speed, bat_speed_impute="league_median"
        )
        # Player 1 (2023) should have imputed bat speed
        assert "avg_bat_speed" in result.columns

    def test_bat_speed_indicator_reflects_observed_data(self):
        """has_avg_bat_speed is 1 for observed rows and 0 for imputed rows."""
        base = pd.DataFrame(
            {
                "mlbam_id": [1000, 2000, 3000],
                "season": [2024, 2024, 2023],
            }
        )
        sprint = pd.DataFrame()
        bat_speed = make_bat_speed(
            [
                {"mlbam_id": 1000, "season": 2024, "avg_bat_speed": 73.5},
                {"mlbam_id": 2000, "season": 2024, "avg_bat_speed": 71.0},
            ]
        )

        result = merge_speed_data(
            base, sprint, bat_speed, bat_speed_impute="league_median"
        )

        # Players 1 and 2 had real bat speed data → indicator = 1
        row1 = result[result["mlbam_id"] == 1000].iloc[0]
        row2 = result[result["mlbam_id"] == 2000].iloc[0]
        assert row1["has_avg_bat_speed"] == 1
        assert row2["has_avg_bat_speed"] == 1

        # Player 3 (2023) had no bat speed data → indicator = 0, value imputed
        row3 = result[result["mlbam_id"] == 3000].iloc[0]
        assert row3["has_avg_bat_speed"] == 0
        assert pd.notna(row3["avg_bat_speed"])  # imputed, not NaN

    def test_extended_bat_tracking_columns_are_supported(self):
        """Extended bat-tracking columns are merged, imputed, and flagged."""
        base = pd.DataFrame(
            {
                "mlbam_id": [1000, 2000],
                "season": [2024, 2024],
            }
        )
        sprint = pd.DataFrame()
        bat_speed = pd.DataFrame(
            {
                "mlbam_id": [1000],
                "season": [2024],
                "avg_bat_speed": [73.2],
                "avg_swing_speed": [71.1],
                "squared_up_rate": [0.31],
                "blast_rate": [0.09],
                "fast_swing_rate": [0.42],
                "bat_tracking_swings": [510],
            }
        )

        result = merge_speed_data(
            base, sprint, bat_speed, bat_speed_impute="league_median"
        )

        for col in [
            "squared_up_rate",
            "blast_rate",
            "fast_swing_rate",
            "bat_tracking_swings",
        ]:
            assert col in result.columns
            assert f"has_{col}" in result.columns

        row_obs = result[result["mlbam_id"] == 1000].iloc[0]
        row_imp = result[result["mlbam_id"] == 2000].iloc[0]
        assert row_obs["has_squared_up_rate"] == 1
        assert row_imp["has_squared_up_rate"] == 0
        assert pd.notna(row_imp["squared_up_rate"])


class TestMergeContextData:
    """Verify park factor and team stat merging."""

    def test_park_factors_merged(self):
        """Park factors are attached via team + season."""
        base = pd.DataFrame(
            {
                "team": ["NYY", "BOS"],
                "season": [2023, 2023],
            }
        )
        park = make_park_factors(["NYY", "BOS"], 2023)
        team = pd.DataFrame()

        result = merge_context_data(base, park, team)
        assert "park_factor_runs" in result.columns
        assert all(result["park_factor_runs"] == 1.0)

    def test_missing_park_factor_defaults_to_neutral(self):
        """Teams without park factor data get 1.0 (neutral)."""
        base = pd.DataFrame(
            {
                "team": ["NYY", "UNKNOWN"],
                "season": [2023, 2023],
            }
        )
        park = make_park_factors(["NYY"], 2023)
        team = pd.DataFrame()

        result = merge_context_data(base, park, team)
        unknown_row = result[result["team"] == "UNKNOWN"].iloc[0]
        assert unknown_row["park_factor_runs"] == 1.0


# ---------------------------------------------------------------------------
# Tests: Target alignment (critical — no temporal leakage)
# ---------------------------------------------------------------------------


class TestTargetAlignment:
    """Verify that targets are correctly aligned to the NEXT season.

    This is the most critical invariant: features from year Y must have
    target values from year Y+1, never from year Y or earlier.
    """

    def test_target_is_next_season_value(self):
        """For (player, 2022), target_hr should be player's 2023 HR."""
        players = [
            {
                "idfg": 100,
                "mlbam_id": 1000,
                "season": 2022,
                "pa": 500,
                "hr": 30,
                "obp": 0.350,
                "slg": 0.500,
                "r": 80,
                "rbi": 90,
                "sb": 10,
            },
            {
                "idfg": 100,
                "mlbam_id": 1000,
                "season": 2023,
                "pa": 550,
                "hr": 35,
                "obp": 0.360,
                "slg": 0.520,
                "r": 90,
                "rbi": 100,
                "sb": 12,
            },
        ]
        df = make_batting(players)
        df["mlbam_id"] = [1000, 1000]

        result = align_targets(df, ["obp", "slg", "hr", "r", "rbi", "sb"])

        row_2022 = result[result["season"] == 2022].iloc[0]
        assert row_2022["target_hr"] == 35
        assert row_2022["target_obp"] == 0.360
        assert row_2022["target_slg"] == 0.520

    def test_last_season_has_nan_targets(self):
        """The most recent season has no Y+1 data, so targets must be NaN."""
        players = [
            {
                "idfg": 100,
                "mlbam_id": 1000,
                "season": 2023,
                "pa": 500,
                "hr": 30,
                "obp": 0.350,
                "slg": 0.500,
                "r": 80,
                "rbi": 90,
                "sb": 10,
            },
            {
                "idfg": 100,
                "mlbam_id": 1000,
                "season": 2024,
                "pa": 520,
                "hr": 32,
                "obp": 0.340,
                "slg": 0.480,
                "r": 85,
                "rbi": 85,
                "sb": 8,
            },
        ]
        df = make_batting(players)
        df["mlbam_id"] = [1000, 1000]

        result = align_targets(df, ["hr", "obp", "slg"])

        row_2024 = result[result["season"] == 2024].iloc[0]
        assert pd.isna(row_2024["target_hr"])
        assert pd.isna(row_2024["target_obp"])

    def test_player_who_skips_year_has_nan(self):
        """If a player skips a year, the gap year's target is NaN."""
        players = [
            {
                "idfg": 100,
                "mlbam_id": 1000,
                "season": 2021,
                "pa": 500,
                "hr": 25,
                "obp": 0.330,
                "slg": 0.450,
                "r": 70,
                "rbi": 75,
                "sb": 5,
            },
            # Player skips 2022
            {
                "idfg": 100,
                "mlbam_id": 1000,
                "season": 2023,
                "pa": 400,
                "hr": 20,
                "obp": 0.310,
                "slg": 0.400,
                "r": 60,
                "rbi": 65,
                "sb": 3,
            },
        ]
        df = make_batting(players)
        df["mlbam_id"] = [1000, 1000]

        result = align_targets(df, ["hr"])

        row_2021 = result[result["season"] == 2021].iloc[0]
        assert pd.isna(row_2021["target_hr"])  # No 2022 data

    def test_different_players_get_own_targets(self):
        """Each player's target comes from THEIR OWN next season, not another player's."""
        players = [
            {
                "idfg": 100,
                "mlbam_id": 1000,
                "season": 2023,
                "pa": 500,
                "hr": 40,
                "obp": 0.350,
                "slg": 0.550,
                "r": 90,
                "rbi": 100,
                "sb": 5,
            },
            {
                "idfg": 100,
                "mlbam_id": 1000,
                "season": 2024,
                "pa": 500,
                "hr": 35,
                "obp": 0.340,
                "slg": 0.530,
                "r": 85,
                "rbi": 95,
                "sb": 4,
            },
            {
                "idfg": 200,
                "mlbam_id": 2000,
                "season": 2023,
                "pa": 500,
                "hr": 10,
                "obp": 0.300,
                "slg": 0.380,
                "r": 60,
                "rbi": 50,
                "sb": 25,
            },
            {
                "idfg": 200,
                "mlbam_id": 2000,
                "season": 2024,
                "pa": 500,
                "hr": 12,
                "obp": 0.310,
                "slg": 0.390,
                "r": 65,
                "rbi": 55,
                "sb": 30,
            },
        ]
        df = make_batting(players)
        df["mlbam_id"] = [1000, 1000, 2000, 2000]

        result = align_targets(df, ["hr"])

        # Player 1's 2023 target should be 35 (their own 2024 HR)
        p1_2023 = result[
            (result["mlbam_id"] == 1000) & (result["season"] == 2023)
        ].iloc[0]
        assert p1_2023["target_hr"] == 35

        # Player 2's 2023 target should be 12 (their own 2024 HR)
        p2_2023 = result[
            (result["mlbam_id"] == 2000) & (result["season"] == 2023)
        ].iloc[0]
        assert p2_2023["target_hr"] == 12

    def test_no_leakage_same_year_feature_not_in_target(self):
        """The target_hr for season Y must NOT equal the hr from season Y."""
        players = [
            {
                "idfg": 100,
                "mlbam_id": 1000,
                "season": 2022,
                "pa": 500,
                "hr": 30,
                "obp": 0.350,
                "slg": 0.500,
                "r": 80,
                "rbi": 90,
                "sb": 10,
            },
            {
                "idfg": 100,
                "mlbam_id": 1000,
                "season": 2023,
                "pa": 500,
                "hr": 35,
                "obp": 0.360,
                "slg": 0.520,
                "r": 90,
                "rbi": 100,
                "sb": 12,
            },
        ]
        df = make_batting(players)
        df["mlbam_id"] = [1000, 1000]

        result = align_targets(df, ["hr"])

        row_2022 = result[result["season"] == 2022].iloc[0]
        # Feature hr is 30 (from 2022), target_hr is 35 (from 2023) — different
        assert row_2022["hr"] == 30
        assert row_2022["target_hr"] == 35
        assert row_2022["hr"] != row_2022["target_hr"]

    def test_target_pa_alignment(self):
        """PA can be aligned as an auxiliary next-season target."""
        players = [
            {
                "idfg": 100,
                "mlbam_id": 1000,
                "season": 2022,
                "pa": 520,
                "hr": 30,
                "obp": 0.350,
                "slg": 0.500,
                "r": 80,
                "rbi": 90,
                "sb": 10,
            },
            {
                "idfg": 100,
                "mlbam_id": 1000,
                "season": 2023,
                "pa": 610,
                "hr": 35,
                "obp": 0.360,
                "slg": 0.520,
                "r": 90,
                "rbi": 100,
                "sb": 12,
            },
        ]
        df = make_batting(players)
        df["mlbam_id"] = [1000, 1000]

        result = align_targets(df, ["obp", "slg", "hr", "r", "rbi", "sb", "pa"])
        row_2022 = result[result["season"] == 2022].iloc[0]
        assert row_2022["target_pa"] == 610


# ---------------------------------------------------------------------------
# Tests: Rate-based targets
# ---------------------------------------------------------------------------


class TestRateTargets:
    """Verify per-PA rate target conversion for count stats."""

    def _make_two_season_df(self) -> pd.DataFrame:
        """Two consecutive seasons for one player."""
        players = [
            {
                "idfg": 100,
                "mlbam_id": 1000,
                "season": 2022,
                "pa": 500,
                "hr": 30,
                "obp": 0.350,
                "slg": 0.500,
                "r": 80,
                "rbi": 90,
                "sb": 10,
            },
            {
                "idfg": 100,
                "mlbam_id": 1000,
                "season": 2023,
                "pa": 600,
                "hr": 36,
                "obp": 0.360,
                "slg": 0.520,
                "r": 96,
                "rbi": 108,
                "sb": 12,
            },
        ]
        df = make_batting(players)
        df["mlbam_id"] = [1000, 1000]
        return df

    def test_count_targets_become_per_pa_rates(self):
        """HR target = next_season_HR / next_season_PA."""
        df = self._make_two_season_df()
        result = align_targets(
            df, ["obp", "slg", "hr", "r", "rbi", "sb"], rate_targets=True,
        )
        row_2022 = result[result["season"] == 2022].iloc[0]

        # HR rate = 36 / 600 = 0.06
        assert abs(row_2022["target_hr"] - 36 / 600) < 1e-9
        # R rate = 96 / 600 = 0.16
        assert abs(row_2022["target_r"] - 96 / 600) < 1e-9
        # RBI rate = 108 / 600 = 0.18
        assert abs(row_2022["target_rbi"] - 108 / 600) < 1e-9
        # SB rate = 12 / 600 = 0.02
        assert abs(row_2022["target_sb"] - 12 / 600) < 1e-9

    def test_rate_targets_unchanged_for_obp_slg(self):
        """OBP and SLG targets are already rates — unchanged by rate_targets."""
        df = self._make_two_season_df()
        result = align_targets(
            df, ["obp", "slg", "hr"], rate_targets=True,
        )
        row_2022 = result[result["season"] == 2022].iloc[0]
        assert row_2022["target_obp"] == 0.360
        assert row_2022["target_slg"] == 0.520

    def test_target_pa_created(self):
        """target_pa is always created when rate_targets=True."""
        df = self._make_two_season_df()
        result = align_targets(
            df, ["hr", "r"], rate_targets=True,
        )
        assert "target_pa" in result.columns
        row_2022 = result[result["season"] == 2022].iloc[0]
        assert row_2022["target_pa"] == 600

    def test_2020_rate_targets_comparable_to_normal_season(self):
        """HR/PA for a 60-game 2020 season is in same range as 162-game seasons."""
        players = [
            {
                "idfg": 100,
                "mlbam_id": 1000,
                "season": 2019,
                "pa": 600,
                "hr": 30,
                "obp": 0.350,
                "slg": 0.500,
                "r": 80,
                "rbi": 90,
                "sb": 10,
            },
            {
                "idfg": 100,
                "mlbam_id": 1000,
                "season": 2020,
                "pa": 222,  # 60-game season
                "hr": 11,   # ~same rate: 11/222 ≈ 0.0495
                "obp": 0.350,
                "slg": 0.500,
                "r": 30,
                "rbi": 32,
                "sb": 4,
            },
            {
                "idfg": 100,
                "mlbam_id": 1000,
                "season": 2021,
                "pa": 580,
                "hr": 28,  # 28/580 ≈ 0.0483
                "obp": 0.340,
                "slg": 0.480,
                "r": 78,
                "rbi": 85,
                "sb": 9,
            },
        ]
        df = make_batting(players)
        df["mlbam_id"] = [1000, 1000, 1000]

        result = align_targets(
            df, ["hr", "r", "rbi", "sb"], rate_targets=True,
        )

        # 2019 target = 2020 values: HR/PA = 11/222 ≈ 0.0495
        row_2019 = result[result["season"] == 2019].iloc[0]
        assert abs(row_2019["target_hr"] - 11 / 222) < 1e-9

        # 2020 target = 2021 values: HR/PA = 28/580 ≈ 0.0483
        row_2020 = result[result["season"] == 2020].iloc[0]
        assert abs(row_2020["target_hr"] - 28 / 580) < 1e-9

        # Both rates are in the same range (~0.048-0.050), unlike raw counts
        # which would be 11 vs 28 — a misleading 2.5x difference
        assert abs(row_2019["target_hr"] - row_2020["target_hr"]) < 0.01

    def test_rate_targets_false_preserves_raw_counts(self):
        """rate_targets=False keeps the original count target behavior."""
        df = self._make_two_season_df()
        result = align_targets(
            df, ["obp", "slg", "hr"], rate_targets=False,
        )
        row_2022 = result[result["season"] == 2022].iloc[0]
        assert row_2022["target_hr"] == 36  # raw count, not rate

    def test_last_season_nan_with_rate_targets(self):
        """Last season still has NaN targets with rate_targets=True."""
        df = self._make_two_season_df()
        result = align_targets(
            df, ["hr", "obp"], rate_targets=True,
        )
        row_2023 = result[result["season"] == 2023].iloc[0]
        assert pd.isna(row_2023["target_hr"])
        assert pd.isna(row_2023["target_obp"])


# ---------------------------------------------------------------------------
# Tests: Chronological splits
# ---------------------------------------------------------------------------


class TestChronologicalSplits:
    """Verify train/val/test splits are strictly chronological."""

    def test_split_boundaries(self):
        """Train/val/test contain only the correct seasons."""
        df = pd.DataFrame(
            {
                "season": [2020, 2021, 2022, 2023, 2024],
                "value": [1, 2, 3, 4, 5],
                "target_x": [10, 20, 30, 40, 50],
            }
        )
        config = SplitConfig(train_end=2022, val_year=2023, test_year=2024)
        train, val, test = split_data(df, config, target_cols=["target_x"])

        assert set(train["season"]) == {2020, 2021, 2022}
        assert set(val["season"]) == {2023}
        assert set(test["season"]) == {2024}

    def test_no_overlap_between_splits(self):
        """No season appears in more than one split."""
        df = pd.DataFrame(
            {
                "season": list(range(2016, 2025)),
                "value": range(9),
                "target_x": range(9),
            }
        )
        config = SplitConfig(train_end=2022, val_year=2023, test_year=2024)
        train, val, test = split_data(df, config, target_cols=["target_x"])

        train_seasons = set(train["season"])
        val_seasons = set(val["season"])
        test_seasons = set(test["season"])

        assert train_seasons & val_seasons == set()
        assert train_seasons & test_seasons == set()
        assert val_seasons & test_seasons == set()

    def test_splits_drop_na_targets(self):
        """Rows with NaN targets are dropped when drop_na_targets=True."""
        df = pd.DataFrame(
            {
                "season": [2022, 2022, 2023],
                "target_x": [10.0, np.nan, 30.0],
            }
        )
        config = SplitConfig(train_end=2022, val_year=2023, test_year=2024)
        train, val, test = split_data(df, config, target_cols=["target_x"])

        assert len(train) == 1  # One row dropped due to NaN
        assert train.iloc[0]["target_x"] == 10.0

    def test_splits_keep_na_targets_when_disabled(self):
        """Rows with NaN targets are kept when drop_na_targets=False."""
        df = pd.DataFrame(
            {
                "season": [2022, 2022],
                "target_x": [10.0, np.nan],
            }
        )
        config = SplitConfig(train_end=2022, val_year=2023, test_year=2024)
        train, _, _ = split_data(
            df, config, target_cols=["target_x"], drop_na_targets=False
        )

        assert len(train) == 2

    def test_split_config_rejects_bad_ordering(self):
        """SplitConfig raises ValueError if years aren't strictly increasing."""
        with pytest.raises(ValueError, match="strictly increasing"):
            SplitConfig(train_end=2024, val_year=2023, test_year=2022)

        with pytest.raises(ValueError, match="strictly increasing"):
            SplitConfig(train_end=2023, val_year=2023, test_year=2024)

    def test_split_config_from_dict(self):
        """SplitConfig.from_dict parses a config dictionary."""
        d = {"train_end": 2022, "val_year": 2023, "test_year": 2024}
        config = SplitConfig.from_dict(d)
        assert config.train_end == 2022
        assert config.val_year == 2023
        assert config.test_year == 2024

    def test_split_config_from_dict_target_year(self):
        """SplitConfig.from_dict derives splits from test_target_year."""
        config = SplitConfig.from_dict({"test_target_year": 2025})
        assert config.train_end == 2022
        assert config.val_year == 2023
        assert config.test_year == 2024

    def test_split_config_from_dict_target_year_shifted(self):
        """test_target_year shifts all split boundaries together."""
        config = SplitConfig.from_dict({"test_target_year": 2024})
        assert config.train_end == 2021
        assert config.val_year == 2022
        assert config.test_year == 2023

    def test_no_future_data_leaks_into_training(self, multi_season_data):
        """Training set never contains seasons beyond train_end."""
        batting, statcast, id_map, _ = multi_season_data
        merged = merge_batting_with_statcast(batting, statcast, id_map)
        merged = align_targets(merged, ["hr", "obp", "slg", "r", "rbi", "sb"])

        config = SplitConfig(train_end=2022, val_year=2023, test_year=2024)
        train, val, test = split_data(
            merged,
            config,
            target_cols=[
                "target_hr",
                "target_obp",
                "target_slg",
                "target_r",
                "target_rbi",
                "target_sb",
            ],
        )

        assert train["season"].max() <= 2022
        assert val["season"].min() == 2023
        assert val["season"].max() == 2023
        assert test["season"].min() == 2024
        assert test["season"].max() == 2024


class TestProductionSplit:
    """Verify production split for 2026 predictions."""

    def test_retrain_has_valid_targets(self):
        """Retrain data excludes the latest season (which has no targets)."""
        df = pd.DataFrame(
            {
                "season": [2022, 2023, 2024, 2025],
                "mlbam_id": [1, 1, 1, 1],
                "target_hr": [25.0, 30.0, 28.0, np.nan],
            }
        )
        retrain, predict = get_production_data(
            df,
            end_year=2025,
            target_cols=["target_hr"],
        )

        assert len(retrain) == 3
        assert all(retrain["target_hr"].notna())
        assert len(predict) == 1
        assert predict.iloc[0]["season"] == 2025

    def test_predict_set_is_latest_season(self):
        """Predict set contains only the end_year season."""
        df = pd.DataFrame(
            {
                "season": [2023, 2024, 2025],
                "mlbam_id": [1, 1, 1],
                "target_hr": [30.0, 28.0, np.nan],
            }
        )
        _, predict = get_production_data(df, end_year=2025)
        assert len(predict) == 1
        assert predict.iloc[0]["season"] == 2025


# ---------------------------------------------------------------------------
# Tests: fetch_statcast local-raw-first behavior
# ---------------------------------------------------------------------------


def _make_synthetic_raw_parquet(path, n_batters=3, bbe_per_batter=60):
    """Create a minimal raw Statcast parquet file for testing."""
    rows = []
    for batter_id in range(1, n_batters + 1):
        for _ in range(bbe_per_batter):
            rows.append(
                {
                    "batter": batter_id,
                    "launch_speed": float(np.random.uniform(70, 115)),
                    "launch_angle": float(np.random.uniform(-20, 50)),
                    "bb_type": "line_drive",
                    "events": "single",
                    "hc_x": 130.0,
                    "hc_y": 170.0,
                    "game_type": "R",
                    "woba_value": 0.9,
                    "estimated_woba_using_speedangle": 0.35,
                }
            )
    df = pd.DataFrame(rows)
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return df


class TestFetchStatcastLocalRaw:
    """Verify fetch_statcast reads local raw files and falls back to API."""

    def test_aggregates_from_local_raw_file(self, tmp_path):
        """When a raw parquet exists, aggregate from it without calling API."""
        raw_path = tmp_path / "statcast_raw_2024.parquet"
        _make_synthetic_raw_parquet(raw_path)

        with patch("pybaseball.statcast") as mock_api:
            result = fetch_statcast(
                2024,
                out_dir=str(tmp_path),
                force=True,
                min_bbe=1,
            )
            mock_api.assert_not_called()

        assert result.exists()
        assert result.name == "statcast_agg_2024.parquet"
        agg = pd.read_parquet(result)
        assert len(agg) > 0
        assert "mlbam_id" in agg.columns

    def test_falls_back_to_api_when_raw_missing(self, tmp_path):
        """When no raw file exists, falls back to API and saves both files."""
        # Build mock API return value
        n = 100
        mock_raw = pd.DataFrame(
            {
                "batter": [1] * n,
                "launch_speed": [95.0] * n,
                "launch_angle": [15.0] * n,
                "bb_type": ["line_drive"] * n,
                "events": ["single"] * n,
                "hc_x": [130.0] * n,
                "hc_y": [170.0] * n,
                "game_type": ["R"] * n,
                "woba_value": [0.9] * n,
                "estimated_woba_using_speedangle": [0.35] * n,
            }
        )

        with patch(
            "pybaseball.statcast",
            return_value=mock_raw,
        ) as mock_api:
            result = fetch_statcast(
                2024,
                out_dir=str(tmp_path),
                force=True,
                delay=0,
                min_bbe=1,
            )
            mock_api.assert_called_once()

        # Both raw and agg files should exist
        assert result.exists()
        assert (tmp_path / "statcast_raw_2024.parquet").exists()
        assert (tmp_path / "statcast_agg_2024.parquet").exists()

    def test_from_api_forces_api_call(self, tmp_path):
        """from_api=True calls API even when local raw file exists."""
        raw_path = tmp_path / "statcast_raw_2024.parquet"
        _make_synthetic_raw_parquet(raw_path)

        n = 100
        mock_raw = pd.DataFrame(
            {
                "batter": [1] * n,
                "launch_speed": [95.0] * n,
                "launch_angle": [15.0] * n,
                "bb_type": ["line_drive"] * n,
                "events": ["single"] * n,
                "hc_x": [130.0] * n,
                "hc_y": [170.0] * n,
                "game_type": ["R"] * n,
                "woba_value": [0.9] * n,
                "estimated_woba_using_speedangle": [0.35] * n,
            }
        )

        with patch(
            "pybaseball.statcast",
            return_value=mock_raw,
        ) as mock_api:
            result = fetch_statcast(
                2024,
                out_dir=str(tmp_path),
                force=True,
                delay=0,
                min_bbe=1,
                from_api=True,
            )
            mock_api.assert_called_once()

        assert result.exists()
