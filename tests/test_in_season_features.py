"""Tests for the in-season feature group (Phase 2 Task 1).

Covers:
- Registry: IN_SEASON group present, 24 features, all unique, toggle works
- compute_in_season_features: schema, dtypes, row count, row order preserved
- week_index: 0-indexed per (mlbam_id, season)
- pa_fraction: pa_ytd / 650 constant
- Trail4w rates: derived from counts when rate columns missing
- Trail4w rates: passthrough when already present
- Missing data: NaN pass-through, no crashes
- IL stubs: days_on_il_ytd == 0, has_il_data == 0 as constants
- Pure function: input DataFrame unchanged
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.in_season import (
    EXPECTED_SEASON_PA,
    IN_SEASON_FEATURE_NAMES,
    compute_in_season_features,
)
from src.features.registry import (
    ALL_FEATURES,
    FeatureGroup,
    FeatureType,
    get_feature_metadata,
    get_feature_names,
)


# ---------------------------------------------------------------------------
# Synthetic weekly snapshot builder
# ---------------------------------------------------------------------------


def _make_snapshot_row(
    mlbam_id: int = 100,
    season: int = 2024,
    iso_week: int = 15,
    pa_ytd: float = 100.0,
    obp_ytd: float = 0.330,
    slg_ytd: float = 0.420,
    hr_per_pa_ytd: float = 0.05,
    r_per_pa_ytd: float = 0.16,
    rbi_per_pa_ytd: float = 0.15,
    sb_per_pa_ytd: float = 0.02,
    iso_ytd: float = 0.150,
    bb_rate_ytd: float = 0.08,
    k_rate_ytd: float = 0.22,
    # trail4w count columns (as produced by build_snapshots)
    trail4w_pa: float = 80.0,
    trail4w_ab: float = 68.0,
    trail4w_h: float = 19.0,
    trail4w_singles: float = 12.0,
    trail4w_doubles: float = 3.0,
    trail4w_triples: float = 0.0,
    trail4w_hr: float = 4.0,
    trail4w_r: float = 12.0,
    trail4w_rbi: float = 14.0,
    trail4w_sb: float = 2.0,
    trail4w_bb: float = 8.0,
    trail4w_so: float = 18.0,
    trail4w_hbp: float = 1.0,
    trail4w_sf: float = 1.0,
    trail4w_sh: float = 0.0,
    trail4w_ibb: float = 0.0,
    trail4w_cs: float = 0.0,
    **kwargs,
) -> dict:
    """Build a synthetic weekly snapshot row with YTD + trail4w counts."""
    row = {
        "mlbam_id": mlbam_id,
        "season": season,
        "iso_year": season,
        "iso_week": iso_week,
        # YTD passthroughs
        "pa_ytd": pa_ytd,
        "obp_ytd": obp_ytd,
        "slg_ytd": slg_ytd,
        "hr_per_pa_ytd": hr_per_pa_ytd,
        "r_per_pa_ytd": r_per_pa_ytd,
        "rbi_per_pa_ytd": rbi_per_pa_ytd,
        "sb_per_pa_ytd": sb_per_pa_ytd,
        "iso_ytd": iso_ytd,
        "bb_rate_ytd": bb_rate_ytd,
        "k_rate_ytd": k_rate_ytd,
        # Trail4w counts (actual build_snapshots output)
        "trail4w_pa": trail4w_pa,
        "trail4w_ab": trail4w_ab,
        "trail4w_h": trail4w_h,
        "trail4w_singles": trail4w_singles,
        "trail4w_doubles": trail4w_doubles,
        "trail4w_triples": trail4w_triples,
        "trail4w_hr": trail4w_hr,
        "trail4w_r": trail4w_r,
        "trail4w_rbi": trail4w_rbi,
        "trail4w_sb": trail4w_sb,
        "trail4w_bb": trail4w_bb,
        "trail4w_so": trail4w_so,
        "trail4w_hbp": trail4w_hbp,
        "trail4w_sf": trail4w_sf,
        "trail4w_sh": trail4w_sh,
        "trail4w_ibb": trail4w_ibb,
        "trail4w_cs": trail4w_cs,
    }
    row.update(kwargs)
    return row


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """IN_SEASON group is present with the expected features."""

    def test_in_season_group_exists(self):
        """FeatureGroup.IN_SEASON enum member is present."""
        assert hasattr(FeatureGroup, "IN_SEASON")
        assert FeatureGroup.IN_SEASON.value == "in_season"

    def test_24_in_season_features(self):
        """Exactly 24 features in the IN_SEASON group."""
        in_season = [f for f in ALL_FEATURES if f.group == FeatureGroup.IN_SEASON]
        assert len(in_season) == 24

    def test_expected_feature_names_registered(self):
        """All 24 expected in-season feature names are in ALL_FEATURES."""
        names = {f.name for f in ALL_FEATURES if f.group == FeatureGroup.IN_SEASON}
        expected = {
            # YTD passthroughs (10)
            "pa_ytd",
            "obp_ytd",
            "slg_ytd",
            "hr_per_pa_ytd",
            "r_per_pa_ytd",
            "rbi_per_pa_ytd",
            "sb_per_pa_ytd",
            "iso_ytd",
            "bb_rate_ytd",
            "k_rate_ytd",
            # Trail4w rates (10)
            "trail4w_pa",
            "trail4w_obp",
            "trail4w_slg",
            "trail4w_hr_per_pa",
            "trail4w_r_per_pa",
            "trail4w_rbi_per_pa",
            "trail4w_sb_per_pa",
            "trail4w_iso",
            "trail4w_bb_rate",
            "trail4w_k_rate",
            # Derived timing (2)
            "week_index",
            "pa_fraction",
            # IL stubs (2)
            "days_on_il_ytd",
            "has_il_data",
        }
        assert names == expected

    def test_feature_metadata_accessible(self):
        """Lookup works for in-season feature metadata."""
        meta = get_feature_metadata("pa_ytd")
        assert meta is not None
        assert meta.group == FeatureGroup.IN_SEASON

        meta_il = get_feature_metadata("has_il_data")
        assert meta_il is not None
        assert meta_il.ftype == FeatureType.INDICATOR

    def test_no_duplicate_names_globally(self):
        """Adding IN_SEASON did not introduce any duplicate feature names."""
        names = [f.name for f in ALL_FEATURES]
        assert len(names) == len(set(names))

    def test_toggle_off_excludes_in_season(self):
        """Disabling in_season removes all 24 features from get_feature_names."""
        enabled = {
            "batting": True,
            "statcast": True,
            "non_contact": True,
            "sprint_speed": True,
            "bat_speed": True,
            "park_factors": True,
            "team_stats": True,
            "age": True,
            "temporal": True,
            "in_season": False,
        }
        names = set(get_feature_names(enabled))
        for name in IN_SEASON_FEATURE_NAMES:
            assert name not in names, f"{name} leaked through toggle"

    def test_toggle_on_includes_in_season(self):
        """Enabling in_season adds all 24 features."""
        enabled = {
            "batting": True,
            "statcast": True,
            "non_contact": True,
            "sprint_speed": True,
            "bat_speed": True,
            "park_factors": True,
            "team_stats": True,
            "age": True,
            "temporal": True,
            "in_season": True,
        }
        names = set(get_feature_names(enabled))
        for name in IN_SEASON_FEATURE_NAMES:
            assert name in names

    def test_in_season_absent_from_config_stays_off(self):
        """IN_SEASON is opt-in: configs without the ``in_season`` key must
        omit all 24 in-season features, preserving pre-Phase-2 behaviour for
        the preseason MTL whose ``data.yaml`` never mentions the new group."""
        baseline = {
            "batting": True,
            "statcast": True,
            "non_contact": True,
            "sprint_speed": True,
            "bat_speed": True,
            "park_factors": True,
            "team_stats": True,
            "age": True,
            "temporal": True,
        }
        names_without_key = set(get_feature_names(baseline))
        # Exactly zero in-season features leak in when the key is absent.
        for name in IN_SEASON_FEATURE_NAMES:
            assert name not in names_without_key, (
                f"{name} leaked in despite in_season being absent from config"
            )

    def test_in_season_absent_equals_explicit_off(self):
        """Omitting the ``in_season`` key is identical to setting it False."""
        baseline = {
            "batting": True,
            "statcast": True,
            "non_contact": True,
            "sprint_speed": True,
            "bat_speed": True,
            "park_factors": True,
            "team_stats": True,
            "age": True,
            "temporal": True,
        }
        names_off = get_feature_names({**baseline, "in_season": False})
        names_absent = get_feature_names(baseline)
        assert names_off == names_absent


# ---------------------------------------------------------------------------
# compute_in_season_features: output schema
# ---------------------------------------------------------------------------


class TestComputeSchema:
    """Output DataFrame schema and basic invariants."""

    def test_returns_dataframe(self):
        df = pd.DataFrame([_make_snapshot_row()])
        out = compute_in_season_features(df)
        assert isinstance(out, pd.DataFrame)

    def test_all_24_columns_present(self):
        df = pd.DataFrame([_make_snapshot_row()])
        out = compute_in_season_features(df)
        for name in IN_SEASON_FEATURE_NAMES:
            assert name in out.columns, f"missing column {name}"

    def test_row_count_preserved(self):
        rows = [
            _make_snapshot_row(mlbam_id=100, iso_week=15),
            _make_snapshot_row(mlbam_id=100, iso_week=16),
            _make_snapshot_row(mlbam_id=200, iso_week=15),
        ]
        df = pd.DataFrame(rows)
        out = compute_in_season_features(df)
        assert len(out) == 3

    def test_row_order_preserved(self):
        """Caller relies on order to concat/merge; must match input index."""
        rows = [
            _make_snapshot_row(mlbam_id=200, iso_week=18),
            _make_snapshot_row(mlbam_id=100, iso_week=15),
            _make_snapshot_row(mlbam_id=100, iso_week=16),
        ]
        df = pd.DataFrame(rows)
        out = compute_in_season_features(df)
        # Index must match input, unchanged
        assert list(out.index) == list(df.index)
        # mlbam_id sequence preserved when present
        if "mlbam_id" in out.columns:
            assert list(out["mlbam_id"]) == [200, 100, 100]

    def test_numeric_dtypes(self):
        """All 24 columns should be numeric (int or float)."""
        df = pd.DataFrame([_make_snapshot_row()])
        out = compute_in_season_features(df)
        for name in IN_SEASON_FEATURE_NAMES:
            assert pd.api.types.is_numeric_dtype(out[name]), (
                f"{name} is not numeric: dtype={out[name].dtype}"
            )


# ---------------------------------------------------------------------------
# YTD passthroughs
# ---------------------------------------------------------------------------


class TestYtdPassthrough:
    def test_ytd_values_unchanged(self):
        row = _make_snapshot_row(
            pa_ytd=250.0,
            obp_ytd=0.340,
            slg_ytd=0.450,
            hr_per_pa_ytd=0.06,
            k_rate_ytd=0.25,
            iso_ytd=0.170,
        )
        df = pd.DataFrame([row])
        out = compute_in_season_features(df)
        assert out["pa_ytd"].iloc[0] == 250.0
        assert out["obp_ytd"].iloc[0] == pytest.approx(0.340)
        assert out["slg_ytd"].iloc[0] == pytest.approx(0.450)
        assert out["hr_per_pa_ytd"].iloc[0] == pytest.approx(0.06)
        assert out["k_rate_ytd"].iloc[0] == pytest.approx(0.25)
        assert out["iso_ytd"].iloc[0] == pytest.approx(0.170)


# ---------------------------------------------------------------------------
# Trail4w rate derivation (when only counts are present)
# ---------------------------------------------------------------------------


class TestTrail4wRateDerivation:
    """Trail4w rates must be derived from counts when rate columns are absent."""

    def test_trail4w_pa_passthrough(self):
        df = pd.DataFrame([_make_snapshot_row(trail4w_pa=72.0)])
        out = compute_in_season_features(df)
        assert out["trail4w_pa"].iloc[0] == 72.0

    def test_trail4w_hr_per_pa_derivation(self):
        df = pd.DataFrame([_make_snapshot_row(trail4w_pa=80.0, trail4w_hr=4.0)])
        out = compute_in_season_features(df)
        assert out["trail4w_hr_per_pa"].iloc[0] == pytest.approx(4.0 / 80.0)

    def test_trail4w_r_per_pa_derivation(self):
        df = pd.DataFrame([_make_snapshot_row(trail4w_pa=80.0, trail4w_r=12.0)])
        out = compute_in_season_features(df)
        assert out["trail4w_r_per_pa"].iloc[0] == pytest.approx(12.0 / 80.0)

    def test_trail4w_rbi_per_pa_derivation(self):
        df = pd.DataFrame([_make_snapshot_row(trail4w_pa=80.0, trail4w_rbi=14.0)])
        out = compute_in_season_features(df)
        assert out["trail4w_rbi_per_pa"].iloc[0] == pytest.approx(14.0 / 80.0)

    def test_trail4w_sb_per_pa_derivation(self):
        df = pd.DataFrame([_make_snapshot_row(trail4w_pa=80.0, trail4w_sb=2.0)])
        out = compute_in_season_features(df)
        assert out["trail4w_sb_per_pa"].iloc[0] == pytest.approx(2.0 / 80.0)

    def test_trail4w_bb_rate_derivation(self):
        df = pd.DataFrame([_make_snapshot_row(trail4w_pa=80.0, trail4w_bb=8.0)])
        out = compute_in_season_features(df)
        assert out["trail4w_bb_rate"].iloc[0] == pytest.approx(8.0 / 80.0)

    def test_trail4w_k_rate_derivation(self):
        df = pd.DataFrame([_make_snapshot_row(trail4w_pa=80.0, trail4w_so=18.0)])
        out = compute_in_season_features(df)
        assert out["trail4w_k_rate"].iloc[0] == pytest.approx(18.0 / 80.0)

    def test_trail4w_obp_derivation(self):
        """OBP = (H + BB + HBP) / (AB + BB + HBP + SF)."""
        df = pd.DataFrame(
            [
                _make_snapshot_row(
                    trail4w_pa=80.0,
                    trail4w_ab=68.0,
                    trail4w_h=19.0,
                    trail4w_bb=8.0,
                    trail4w_hbp=1.0,
                    trail4w_sf=1.0,
                )
            ]
        )
        out = compute_in_season_features(df)
        expected = (19.0 + 8.0 + 1.0) / (68.0 + 8.0 + 1.0 + 1.0)
        assert out["trail4w_obp"].iloc[0] == pytest.approx(expected)

    def test_trail4w_slg_derivation(self):
        """SLG = (1B + 2·2B + 3·3B + 4·HR) / AB."""
        df = pd.DataFrame(
            [
                _make_snapshot_row(
                    trail4w_ab=68.0,
                    trail4w_singles=12.0,
                    trail4w_doubles=3.0,
                    trail4w_triples=0.0,
                    trail4w_hr=4.0,
                )
            ]
        )
        out = compute_in_season_features(df)
        expected = (12.0 + 2 * 3.0 + 3 * 0.0 + 4 * 4.0) / 68.0
        assert out["trail4w_slg"].iloc[0] == pytest.approx(expected)

    def test_trail4w_iso_derivation(self):
        """ISO = SLG - AVG."""
        df = pd.DataFrame(
            [
                _make_snapshot_row(
                    trail4w_ab=68.0,
                    trail4w_h=19.0,
                    trail4w_singles=12.0,
                    trail4w_doubles=3.0,
                    trail4w_triples=0.0,
                    trail4w_hr=4.0,
                )
            ]
        )
        out = compute_in_season_features(df)
        slg = (12.0 + 2 * 3.0 + 3 * 0.0 + 4 * 4.0) / 68.0
        avg = 19.0 / 68.0
        assert out["trail4w_iso"].iloc[0] == pytest.approx(slg - avg)

    def test_trail4w_division_by_zero_gives_nan(self):
        """trail4w_pa == 0 must yield NaN per-PA rates, not inf or raise."""
        df = pd.DataFrame(
            [
                _make_snapshot_row(
                    trail4w_pa=0.0,
                    trail4w_hr=0.0,
                    trail4w_ab=0.0,
                    trail4w_bb=0.0,
                    trail4w_so=0.0,
                )
            ]
        )
        out = compute_in_season_features(df)
        assert np.isnan(out["trail4w_hr_per_pa"].iloc[0])
        assert np.isnan(out["trail4w_bb_rate"].iloc[0])
        assert np.isnan(out["trail4w_k_rate"].iloc[0])


class TestTrail4wRatePassthrough:
    """If pre-computed trail4w rate columns are present, use them."""

    def test_existing_trail4w_obp_passthrough(self):
        row = _make_snapshot_row()
        row["trail4w_obp"] = 0.350  # pre-computed
        df = pd.DataFrame([row])
        out = compute_in_season_features(df)
        assert out["trail4w_obp"].iloc[0] == pytest.approx(0.350)

    def test_existing_trail4w_slg_passthrough(self):
        """Pre-computed trail4w_slg wins over the derived formula."""
        row = _make_snapshot_row()
        row["trail4w_slg"] = 0.501
        df = pd.DataFrame([row])
        out = compute_in_season_features(df)
        assert out["trail4w_slg"].iloc[0] == pytest.approx(0.501)

    def test_existing_trail4w_iso_passthrough(self):
        """Pre-computed trail4w_iso bypasses SLG - AVG derivation."""
        row = _make_snapshot_row()
        row["trail4w_iso"] = 0.199
        df = pd.DataFrame([row])
        out = compute_in_season_features(df)
        assert out["trail4w_iso"].iloc[0] == pytest.approx(0.199)

    def test_existing_trail4w_per_pa_rate_passthrough(self):
        """Pre-computed trail4w per-PA rate columns bypass the counts."""
        row = _make_snapshot_row(trail4w_pa=80.0, trail4w_hr=4.0)
        # Set a value distinct from the derived 4/80 = 0.05
        row["trail4w_hr_per_pa"] = 0.123
        df = pd.DataFrame([row])
        out = compute_in_season_features(df)
        assert out["trail4w_hr_per_pa"].iloc[0] == pytest.approx(0.123)


# ---------------------------------------------------------------------------
# Derived timing features
# ---------------------------------------------------------------------------


class TestWeekIndex:
    def test_week_index_zero_for_first_week(self):
        """Earliest week of each (mlbam_id, season) has week_index=0."""
        rows = [
            _make_snapshot_row(mlbam_id=100, season=2024, iso_week=15),
            _make_snapshot_row(mlbam_id=100, season=2024, iso_week=17),
            _make_snapshot_row(mlbam_id=100, season=2024, iso_week=20),
        ]
        df = pd.DataFrame(rows)
        out = compute_in_season_features(df)
        assert out["week_index"].iloc[0] == 0
        assert out["week_index"].iloc[1] == 2
        assert out["week_index"].iloc[2] == 5

    def test_week_index_per_player(self):
        """Each (mlbam_id, season) gets its own 0-origin."""
        rows = [
            _make_snapshot_row(mlbam_id=100, season=2024, iso_week=15),
            _make_snapshot_row(mlbam_id=100, season=2024, iso_week=16),
            _make_snapshot_row(mlbam_id=200, season=2024, iso_week=18),
            _make_snapshot_row(mlbam_id=200, season=2024, iso_week=20),
        ]
        df = pd.DataFrame(rows)
        out = compute_in_season_features(df)
        out_p100 = out[out.index.isin([0, 1])]
        out_p200 = out[out.index.isin([2, 3])]
        assert list(out_p100["week_index"]) == [0, 1]
        assert list(out_p200["week_index"]) == [0, 2]

    def test_week_index_per_season(self):
        """A player's week_index resets each season."""
        rows = [
            _make_snapshot_row(mlbam_id=100, season=2023, iso_week=30),
            _make_snapshot_row(mlbam_id=100, season=2024, iso_week=15),
            _make_snapshot_row(mlbam_id=100, season=2024, iso_week=17),
        ]
        df = pd.DataFrame(rows)
        out = compute_in_season_features(df)
        assert out["week_index"].iloc[0] == 0  # 2023 first row
        assert out["week_index"].iloc[1] == 0  # 2024 first row
        assert out["week_index"].iloc[2] == 2  # 2024 second row


class TestPaFraction:
    def test_pa_fraction_uses_650_denominator(self):
        df = pd.DataFrame([_make_snapshot_row(pa_ytd=325.0)])
        out = compute_in_season_features(df)
        assert out["pa_fraction"].iloc[0] == pytest.approx(325.0 / 650.0)

    def test_pa_fraction_zero(self):
        df = pd.DataFrame([_make_snapshot_row(pa_ytd=0.0)])
        out = compute_in_season_features(df)
        assert out["pa_fraction"].iloc[0] == pytest.approx(0.0)

    def test_pa_fraction_full_season(self):
        df = pd.DataFrame([_make_snapshot_row(pa_ytd=650.0)])
        out = compute_in_season_features(df)
        assert out["pa_fraction"].iloc[0] == pytest.approx(1.0)

    def test_expected_season_pa_constant_is_650(self):
        """The documented design choice: 650 PA league-average regular."""
        assert EXPECTED_SEASON_PA == 650


# ---------------------------------------------------------------------------
# IL stubs
# ---------------------------------------------------------------------------


class TestIlStubs:
    def test_days_on_il_ytd_is_zero(self):
        df = pd.DataFrame([_make_snapshot_row() for _ in range(3)])
        out = compute_in_season_features(df)
        assert (out["days_on_il_ytd"] == 0).all()

    def test_has_il_data_is_zero(self):
        df = pd.DataFrame([_make_snapshot_row() for _ in range(3)])
        out = compute_in_season_features(df)
        assert (out["has_il_data"] == 0).all()

    def test_il_stubs_are_numeric(self):
        df = pd.DataFrame([_make_snapshot_row()])
        out = compute_in_season_features(df)
        assert pd.api.types.is_numeric_dtype(out["days_on_il_ytd"])
        assert pd.api.types.is_numeric_dtype(out["has_il_data"])


# ---------------------------------------------------------------------------
# Missing data / graceful handling
# ---------------------------------------------------------------------------


class TestMissingData:
    """Graceful handling of missing input columns and NaN values."""

    def test_missing_trail4w_counts_gives_nan_rates(self):
        """When trail4w count columns are absent, trail4w rate columns become NaN."""
        minimal_row = {
            "mlbam_id": 100,
            "season": 2024,
            "iso_year": 2024,
            "iso_week": 15,
            "pa_ytd": 200.0,
            "obp_ytd": 0.340,
            "slg_ytd": 0.440,
            "hr_per_pa_ytd": 0.05,
            "r_per_pa_ytd": 0.15,
            "rbi_per_pa_ytd": 0.14,
            "sb_per_pa_ytd": 0.02,
            "iso_ytd": 0.15,
            "bb_rate_ytd": 0.08,
            "k_rate_ytd": 0.22,
            # no trail4w columns at all
        }
        df = pd.DataFrame([minimal_row])
        out = compute_in_season_features(df)
        # Expect NaN — no crash
        assert np.isnan(out["trail4w_pa"].iloc[0])
        assert np.isnan(out["trail4w_hr_per_pa"].iloc[0])
        assert np.isnan(out["trail4w_obp"].iloc[0])
        assert np.isnan(out["trail4w_slg"].iloc[0])

    def test_nan_pa_ytd_passes_through(self):
        row = _make_snapshot_row(pa_ytd=np.nan)
        df = pd.DataFrame([row])
        out = compute_in_season_features(df)
        assert np.isnan(out["pa_ytd"].iloc[0])
        # pa_fraction depends on pa_ytd → NaN
        assert np.isnan(out["pa_fraction"].iloc[0])

    def test_partial_trail4w_columns(self):
        """Only some trail4w counts present → compute what we can, NaN for rest."""
        row = _make_snapshot_row(trail4w_pa=80.0, trail4w_hr=4.0)
        # Remove trail4w_bb so trail4w_bb_rate can't be computed
        del row["trail4w_bb"]
        df = pd.DataFrame([row])
        out = compute_in_season_features(df)
        # hr_per_pa still computable (both cols present)
        assert out["trail4w_hr_per_pa"].iloc[0] == pytest.approx(4.0 / 80.0)
        # bb_rate not computable
        assert np.isnan(out["trail4w_bb_rate"].iloc[0])

    def test_no_crash_on_empty_df(self):
        """Empty input → empty output with right schema."""
        df = pd.DataFrame(columns=["mlbam_id", "season", "iso_week", "pa_ytd"])
        out = compute_in_season_features(df)
        assert len(out) == 0
        for name in IN_SEASON_FEATURE_NAMES:
            assert name in out.columns


# ---------------------------------------------------------------------------
# Purity / side effect tests
# ---------------------------------------------------------------------------


class TestPurity:
    """compute_in_season_features should not mutate its input."""

    def test_input_columns_not_modified(self):
        rows = [
            _make_snapshot_row(mlbam_id=100, iso_week=15),
            _make_snapshot_row(mlbam_id=100, iso_week=17),
        ]
        df = pd.DataFrame(rows)
        original_cols = set(df.columns)
        original_row0 = df.iloc[0].copy()

        _ = compute_in_season_features(df)

        # Caller's DataFrame schema should be preserved
        assert set(df.columns) == original_cols
        # First row values unchanged
        for col in original_cols:
            v1, v2 = df.iloc[0][col], original_row0[col]
            if pd.isna(v1) and pd.isna(v2):
                continue
            assert v1 == v2
