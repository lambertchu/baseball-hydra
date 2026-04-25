"""Tests for the weekly snapshot pipeline (Phase 1 data layer).

Covers:
- Weekly Statcast aggregation (groupby ISO week, requires game_date)
- ISO week enumeration for season date ranges
- BRef column normalization
- Snapshot build end-to-end: ytd cumulative, ROS targets, min_ytd_pa filter
- Core invariants: PA-weighted sum identity, ytd + ros == season total,
  OBP/SLG formula correctness, no temporal leakage
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from unittest.mock import patch

from src.data.build_snapshots import (
    _add_ros_rates,
    _add_week_suffix,
    _add_ytd_rates,
    _apply_count_ytd_trail_ros,
    _derive_singles,
    _merge_weekly_sources,
    build_weekly_snapshots,
)
from src.data.fetch_game_logs import (
    _aggregate_statcast_batting_weekly,
    _normalize_bref_columns,
    _overlay_scaled_season_totals,
    fetch_batter_weekly_stats,
    iso_weeks_in_season,
)
from src.data.fetch_statcast import _aggregate_batter_statcast_weekly


def _prep_ytd(df: pd.DataFrame) -> pd.DataFrame:
    """Test helper: apply week-suffix rename + ytd/trail/ros in one step."""
    return _apply_count_ytd_trail_ros(_add_week_suffix(df))


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _make_statcast_bbe_rows(
    batter: int,
    game_date: str,
    n: int,
    ev_mean: float = 90.0,
    la_mean: float = 12.0,
    barrel_pct: float = 0.10,
    hard_hit_pct: float = 0.40,
) -> pd.DataFrame:
    """Create N BBE rows on a specific date for one batter."""
    rng = np.random.default_rng(hash((batter, game_date)) % 2**32)
    ev = rng.normal(ev_mean, 5.0, size=n)
    # Force some hard hits / barrels to hit the requested fractions approximately
    n_hard = int(round(n * hard_hit_pct))
    ev[:n_hard] = np.maximum(ev[:n_hard], 96.0)
    la = rng.normal(la_mean, 10.0, size=n)
    lsa = np.zeros(n)
    n_barrel = int(round(n * barrel_pct))
    lsa[:n_barrel] = 6  # barrel code
    return pd.DataFrame({
        "batter": batter,
        "game_date": pd.to_datetime(game_date),
        "launch_speed": ev,
        "launch_angle": la,
        "bb_type": "fly_ball",
        "game_type": "R",
        "launch_speed_angle": lsa,
        "estimated_woba_using_speedangle": rng.uniform(0.3, 0.5, size=n),
        "estimated_ba_using_speedangle": rng.uniform(0.2, 0.4, size=n),
        "estimated_slg_using_speedangle": rng.uniform(0.4, 0.7, size=n),
    })


def _make_weekly_batting_row(
    mlbam_id: int,
    iso_year: int,
    iso_week: int,
    pa: int,
    hr: int = 0,
    doubles: int = 0,
    triples: int = 0,
    singles: int = 0,
    bb: int = 0,
    so: int = 0,
    hbp: int = 0,
    sf: int = 0,
    r: int = 0,
    rbi: int = 0,
    sb: int = 0,
    cs: int = 0,
    ibb: int = 0,
    sh: int = 0,
    week_start: date | None = None,
    season: int | None = None,
) -> dict:
    """Build a single synthetic weekly batting row (post-normalization schema)."""
    h = singles + doubles + triples + hr
    ab = pa - bb - hbp - sf - sh - ibb
    return {
        "mlbam_id": mlbam_id,
        "name": f"Player_{mlbam_id}",
        "age": 28,
        "team": "NYY",
        "level": "Maj",
        "iso_year": iso_year,
        "iso_week": iso_week,
        "week_start_date": pd.Timestamp(week_start or date(iso_year, 1, 1)),
        "week_end_date": pd.Timestamp(week_start or date(iso_year, 1, 1)),
        "season": season if season is not None else iso_year,
        "pa": pa, "ab": ab, "h": h,
        "doubles": doubles, "triples": triples, "hr": hr,
        "r": r, "rbi": rbi,
        "bb": bb, "ibb": ibb, "so": so, "hbp": hbp, "sf": sf, "sh": sh,
        "sb": sb, "cs": cs,
    }


def _make_statcast_pa_row(
    batter: int,
    event: str,
    game_date: str = "2024-04-08",
    game_pk: int = 1,
    at_bat_number: int = 1,
    pitch_number: int = 1,
    on_1b: int | None = None,
    on_2b: int | None = None,
    on_3b: int | None = None,
    bat_score: int = 0,
    post_bat_score: int = 0,
    des: str = "",
) -> dict:
    """Build a minimal final-pitch Statcast row for batting aggregation tests."""
    return {
        "batter": batter,
        "game_date": pd.Timestamp(game_date),
        "game_pk": game_pk,
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
        "events": event,
        "description": "hit_into_play",
        "game_type": "R",
        "on_1b": on_1b,
        "on_2b": on_2b,
        "on_3b": on_3b,
        "bat_score": bat_score,
        "post_bat_score": post_bat_score,
        "des": des,
    }


# ---------------------------------------------------------------------------
# Weekly Statcast aggregation
# ---------------------------------------------------------------------------


class TestAggregateWeekly:

    def test_requires_game_date(self):
        """Weekly aggregation must fail clearly without game_date."""
        df = pd.DataFrame({
            "batter": [1, 2],
            "launch_speed": [90.0, 95.0],
            "launch_angle": [10.0, 15.0],
            "bb_type": ["fly_ball", "ground_ball"],
            "game_type": ["R", "R"],
        })
        with pytest.raises(ValueError, match="game_date"):
            _aggregate_batter_statcast_weekly(df)

    def test_groups_by_iso_week(self):
        """Same batter, different weeks → separate rows."""
        wk1 = _make_statcast_bbe_rows(100, "2024-04-08", n=20)  # ISO 2024-W15
        wk2 = _make_statcast_bbe_rows(100, "2024-04-15", n=18)  # ISO 2024-W16
        df = pd.concat([wk1, wk2], ignore_index=True)

        agg = _aggregate_batter_statcast_weekly(df, min_bbe=5)

        assert len(agg) == 2
        weeks = set(zip(agg["iso_year"], agg["iso_week"]))
        assert weeks == {(2024, 15), (2024, 16)}
        assert (agg["mlbam_id"] == 100).all()

    def test_min_bbe_filter(self):
        """Weeks below min_bbe are dropped."""
        wk1 = _make_statcast_bbe_rows(100, "2024-04-08", n=3)
        wk2 = _make_statcast_bbe_rows(100, "2024-04-15", n=20)
        df = pd.concat([wk1, wk2], ignore_index=True)

        agg = _aggregate_batter_statcast_weekly(df, min_bbe=5)

        assert len(agg) == 1
        assert agg["iso_week"].iloc[0] == 16

    def test_empty_input(self):
        """Empty raw data → empty aggregate."""
        agg = _aggregate_batter_statcast_weekly(
            pd.DataFrame(columns=[
                "batter", "game_date", "launch_speed", "launch_angle",
                "bb_type", "game_type",
            ])
        )
        assert agg.empty

    def test_rate_metrics_present(self):
        """Barrel, hard-hit, sweet-spot rates appear in output."""
        df = _make_statcast_bbe_rows(100, "2024-04-08", n=30)
        agg = _aggregate_batter_statcast_weekly(df, min_bbe=5)

        assert "barrel_rate" in agg.columns
        assert "hard_hit_rate" in agg.columns
        assert "sweet_spot_rate" in agg.columns
        assert (agg["barrel_rate"] >= 0).all()
        assert (agg["barrel_rate"] <= 1).all()


class TestStatcastWeeklyBattingFallback:
    """Tests for deriving BRef-like weekly batting logs from full Statcast."""

    def test_aggregates_pa_counts_runs_rbi_and_running_events(self):
        raw = pd.DataFrame([
            _make_statcast_pa_row(
                10,
                "single",
                game_pk=1,
                at_bat_number=1,
                bat_score=0,
                post_bat_score=0,
            ),
            _make_statcast_pa_row(
                20,
                "home_run",
                game_pk=1,
                at_bat_number=2,
                on_1b=10,
                bat_score=0,
                post_bat_score=2,
            ),
            _make_statcast_pa_row(
                10,
                "walk",
                game_pk=1,
                at_bat_number=3,
                bat_score=2,
                post_bat_score=2,
            ),
            _make_statcast_pa_row(
                10,
                "sac_fly",
                game_pk=1,
                at_bat_number=4,
                on_3b=20,
                bat_score=2,
                post_bat_score=3,
            ),
            _make_statcast_pa_row(
                30,
                "truncated_pa",
                game_pk=1,
                at_bat_number=5,
                on_1b=10,
                bat_score=3,
                post_bat_score=3,
                des="Player Ten caught stealing 2nd base, catcher to shortstop.",
            ),
        ])

        weekly = _aggregate_statcast_batting_weekly(raw, season=2024)
        by_id = weekly.set_index("mlbam_id")

        assert by_id.loc[10, "pa"] == 3
        assert by_id.loc[10, "ab"] == 1
        assert by_id.loc[10, "h"] == 1
        assert by_id.loc[10, "bb"] == 1
        assert by_id.loc[10, "sf"] == 1
        assert by_id.loc[10, "r"] == 1
        assert by_id.loc[10, "rbi"] == 1
        assert by_id.loc[10, "cs"] == 1

        assert by_id.loc[20, "pa"] == 1
        assert by_id.loc[20, "ab"] == 1
        assert by_id.loc[20, "h"] == 1
        assert by_id.loc[20, "hr"] == 1
        assert by_id.loc[20, "r"] == 2
        assert by_id.loc[20, "rbi"] == 2

    def test_season_total_overlay_preserves_weekly_shape_and_exact_totals(self):
        weekly = pd.DataFrame({
            "mlbam_id": [10, 10, 10],
            "iso_year": [2024, 2024, 2024],
            "iso_week": [14, 15, 16],
            "pa": [10, 20, 30],
            "r": [0, 1, 2],
            "rbi": [1, 1, 1],
            "sb": [0, 0, 0],
            "cs": [0, 0, 0],
        })
        season_totals = pd.DataFrame({
            "mlbam_id": [10],
            "r": [6],
            "rbi": [9],
            "sb": [6],
            "cs": [3],
        })

        out = _overlay_scaled_season_totals(weekly, season_totals)

        assert out["r"].tolist() == [0, 2, 4]
        assert out["rbi"].tolist() == [3, 3, 3]
        assert out["sb"].tolist() == [1, 2, 3]
        assert out["cs"].tolist() == [1, 1, 1]
        assert out[["r", "rbi", "sb", "cs"]].sum().to_dict() == {
            "r": 6,
            "rbi": 9,
            "sb": 6,
            "cs": 3,
        }


# ---------------------------------------------------------------------------
# ISO week enumeration
# ---------------------------------------------------------------------------


class TestIsoWeeksInSeason:

    def test_covers_season_span(self):
        # Fake "season": single week.
        weeks = iso_weeks_in_season(
            2024,
            season_dates={2024: ("2024-04-08", "2024-04-14")},
        )
        assert len(weeks) == 1
        iso_year, iso_week, wstart, wend = weeks[0]
        assert iso_year == 2024 and iso_week == 15
        assert wstart == date(2024, 4, 8)
        assert wend == date(2024, 4, 14)
        assert wstart.weekday() == 0  # Monday

    def test_multi_week(self):
        weeks = iso_weeks_in_season(
            2024,
            season_dates={2024: ("2024-04-01", "2024-04-21")},
        )
        # Full weeks 14, 15, 16 + partial start/end → expect 3 or 4
        assert 3 <= len(weeks) <= 4
        for _, _, wstart, wend in weeks:
            assert wstart.weekday() == 0
            assert (wend - wstart).days == 6

    def test_real_2024_season(self):
        weeks = iso_weeks_in_season(2024)
        assert 25 <= len(weeks) <= 35  # ~26 weeks in real seasons


# ---------------------------------------------------------------------------
# BRef column normalization
# ---------------------------------------------------------------------------


class TestNormalizeBref:

    def test_renames_core_columns(self):
        raw = pd.DataFrame({
            "Name": ["A"], "Tm": ["NYY"], "Age": ["28"], "Lev": ["Maj"],
            "PA": ["20"], "AB": ["18"], "R": ["3"], "H": ["5"],
            "2B": ["1"], "3B": ["0"], "HR": ["1"], "RBI": ["3"],
            "BB": ["2"], "IBB": ["0"], "SO": ["4"], "HBP": ["0"],
            "SH": ["0"], "SF": ["0"], "GDP": ["0"], "SB": ["0"], "CS": ["0"],
            "BA": ["0.278"], "OBP": ["0.350"], "SLG": ["0.500"], "OPS": ["0.850"],
            "mlbID": ["123456"],
        })
        out = _normalize_bref_columns(raw)
        for col in [
            "name", "team", "age", "level", "pa", "ab", "r", "h",
            "doubles", "triples", "hr", "rbi", "bb", "ibb", "so", "hbp",
            "sh", "sf", "gdp", "sb", "cs", "avg", "obp", "slg", "ops",
            "mlbam_id",
        ]:
            assert col in out.columns, f"missing {col}"
        assert pd.api.types.is_integer_dtype(out["pa"]) or pd.api.types.is_float_dtype(out["pa"])
        assert out["mlbam_id"].iloc[0] == 123456

    def test_drops_minor_league_rows(self):
        raw = pd.DataFrame({
            "Name": ["A", "B"], "Lev": ["Maj", "AAA"],
            "PA": ["10", "15"], "mlbID": ["1", "2"],
        })
        out = _normalize_bref_columns(raw)
        assert len(out) == 1
        assert out["name"].iloc[0] == "A"

    def test_accepts_mlb_and_maj_level_prefixes(self):
        """BRef has used both 'Maj-AL/Maj-NL' and 'MLB-AL/MLB-NL' over time."""
        raw = pd.DataFrame({
            "Name": ["A", "B", "C", "D", "E"],
            "Lev": ["Maj-AL", "MLB-NL", "MLB", "AAA", "AA"],
            "PA": ["10", "15", "12", "8", "5"],
            "mlbID": ["1", "2", "3", "4", "5"],
        })
        out = _normalize_bref_columns(raw)
        assert sorted(out["name"].tolist()) == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Snapshot builder internals
# ---------------------------------------------------------------------------


class TestDeriveSingles:

    def test_formula(self):
        df = pd.DataFrame({"h": [5], "doubles": [1], "triples": [0], "hr": [1]})
        out = _derive_singles(df)
        assert out["singles"].iloc[0] == 3

    def test_clips_to_zero(self):
        """Data errors (hr > h) shouldn't produce negative singles."""
        df = pd.DataFrame({"h": [1], "doubles": [0], "triples": [0], "hr": [2]})
        out = _derive_singles(df)
        assert out["singles"].iloc[0] == 0


class TestYtdCumulative:

    def test_cumsum_within_player(self):
        rows = [
            _make_weekly_batting_row(100, 2024, 15, pa=20, hr=1),
            _make_weekly_batting_row(100, 2024, 16, pa=22, hr=0),
            _make_weekly_batting_row(100, 2024, 17, pa=18, hr=2),
        ]
        df = _derive_singles(pd.DataFrame(rows))
        out = _prep_ytd(df)

        assert list(out["pa_ytd"]) == [20, 42, 60]
        assert list(out["hr_ytd"]) == [1, 1, 3]

    def test_ytd_separate_per_player(self):
        rows = [
            _make_weekly_batting_row(100, 2024, 15, pa=20),
            _make_weekly_batting_row(200, 2024, 15, pa=15),
            _make_weekly_batting_row(100, 2024, 16, pa=22),
            _make_weekly_batting_row(200, 2024, 16, pa=10),
        ]
        df = _derive_singles(pd.DataFrame(rows))
        out = _prep_ytd(df).sort_values(["mlbam_id", "iso_week"])

        p100 = out[out["mlbam_id"] == 100]
        p200 = out[out["mlbam_id"] == 200]
        assert list(p100["pa_ytd"]) == [20, 42]
        assert list(p200["pa_ytd"]) == [15, 25]


class TestYtdRates:

    def test_obp_formula(self):
        """OBP = (H + BB + HBP) / (AB + BB + HBP + SF)."""
        rows = [_make_weekly_batting_row(
            100, 2024, 15, pa=100,
            singles=20, doubles=5, triples=1, hr=4,
            bb=10, so=15, hbp=2, sf=3,
        )]
        df = _derive_singles(pd.DataFrame(rows))
        out = _add_ytd_rates(_prep_ytd(df))

        h = 20 + 5 + 1 + 4  # 30
        ab = 100 - 10 - 2 - 3 - 0 - 0  # 85
        expected_obp = (h + 10 + 2) / (ab + 10 + 2 + 3)
        assert out["obp_ytd"].iloc[0] == pytest.approx(expected_obp, rel=1e-6)

    def test_slg_formula(self):
        rows = [_make_weekly_batting_row(
            100, 2024, 15, pa=100,
            singles=20, doubles=5, triples=1, hr=4,
            bb=10, so=15, hbp=2, sf=3,
        )]
        df = _derive_singles(pd.DataFrame(rows))
        out = _add_ytd_rates(_prep_ytd(df))

        tb = 20 + 2 * 5 + 3 * 1 + 4 * 4  # 49
        ab = 100 - 10 - 2 - 3
        assert out["slg_ytd"].iloc[0] == pytest.approx(tb / ab, rel=1e-6)

    def test_per_pa_rates(self):
        rows = [_make_weekly_batting_row(
            100, 2024, 15, pa=100, hr=5, r=20, rbi=18, sb=3,
            singles=20, doubles=5, bb=10, so=15, hbp=2, sf=3,
        )]
        df = _derive_singles(pd.DataFrame(rows))
        out = _add_ytd_rates(_prep_ytd(df))

        assert out["hr_per_pa_ytd"].iloc[0] == pytest.approx(0.05, rel=1e-6)
        assert out["r_per_pa_ytd"].iloc[0] == pytest.approx(0.20, rel=1e-6)
        assert out["rbi_per_pa_ytd"].iloc[0] == pytest.approx(0.18, rel=1e-6)
        assert out["sb_per_pa_ytd"].iloc[0] == pytest.approx(0.03, rel=1e-6)


class TestRosTargets:

    def test_ytd_plus_ros_equals_season_total(self):
        """The core ROS invariant: ytd + ros == season total at every week."""
        rows = [
            _make_weekly_batting_row(100, 2024, 15, pa=25, hr=1, r=3, rbi=4, sb=1,
                                      singles=5, doubles=1, bb=2, so=5, hbp=0, sf=0),
            _make_weekly_batting_row(100, 2024, 16, pa=22, hr=2, r=4, rbi=5, sb=0,
                                      singles=4, doubles=2, bb=3, so=4, hbp=1, sf=0),
            _make_weekly_batting_row(100, 2024, 17, pa=28, hr=1, r=2, rbi=3, sb=2,
                                      singles=6, doubles=0, bb=4, so=6, hbp=0, sf=1),
            _make_weekly_batting_row(100, 2024, 18, pa=20, hr=0, r=1, rbi=2, sb=0,
                                      singles=3, doubles=1, bb=2, so=3, hbp=0, sf=0),
        ]
        df = _derive_singles(pd.DataFrame(rows))
        out = _add_ros_rates(_add_ytd_rates(_prep_ytd(df)))

        # For every count stat at every row: ytd + ros == season total
        for stat in ("pa", "hr", "r", "rbi", "sb"):
            season_total = out[f"{stat}_week"].sum()
            for _, row in out.iterrows():
                assert row[f"{stat}_ytd"] + row[f"ros_{stat}"] == pytest.approx(season_total)

    def test_ros_pa_nonneg(self):
        rows = [
            _make_weekly_batting_row(100, 2024, 15, pa=20),
            _make_weekly_batting_row(100, 2024, 16, pa=22),
        ]
        df = _derive_singles(pd.DataFrame(rows))
        out = _add_ros_rates(_add_ytd_rates(_prep_ytd(df)))
        assert (out["ros_pa"] >= 0).all()

    def test_last_week_ros_is_zero(self):
        rows = [
            _make_weekly_batting_row(100, 2024, 15, pa=20, hr=1),
            _make_weekly_batting_row(100, 2024, 16, pa=22, hr=2),
        ]
        df = _derive_singles(pd.DataFrame(rows))
        out = _add_ros_rates(_add_ytd_rates(_prep_ytd(df)))
        last = out.sort_values("iso_week").iloc[-1]
        assert last["ros_pa"] == 0
        assert last["ros_hr"] == 0


# ---------------------------------------------------------------------------
# Full pipeline via parquet round-trip
# ---------------------------------------------------------------------------


class TestBuildWeeklySnapshotsPipeline:

    def test_end_to_end(self, tmp_path):
        """Full pipeline: write weekly parquets, build snapshots, assert schema."""
        # Build synthetic weekly batting parquet for 2024
        rows = []
        for wk, pa, hr in [
            (14, 25, 1), (15, 22, 2), (16, 28, 1), (17, 20, 3),
        ]:
            rows.append(_make_weekly_batting_row(
                100, 2024, wk, pa=pa, hr=hr,
                singles=5, doubles=1, bb=2, so=5, hbp=0, sf=0,
                r=3, rbi=4, sb=1, season=2024,
            ))
        # Second player with a later debut (only 2 weeks of data)
        for wk, pa, hr in [(16, 15, 0), (17, 18, 1)]:
            rows.append(_make_weekly_batting_row(
                200, 2024, wk, pa=pa, hr=hr,
                singles=3, doubles=0, bb=1, so=4, hbp=0, sf=0,
                r=1, rbi=2, sb=0, season=2024,
            ))
        batting_wk = pd.DataFrame(rows)
        # Include singles already so _derive_singles is idempotent
        batting_wk = _derive_singles(batting_wk)
        batting_wk.to_parquet(tmp_path / "batting_week_2024.parquet", index=False)

        # Build with no statcast file (should log warning, still work)
        result = build_weekly_snapshots(
            2024,
            raw_dir=tmp_path,
            out_dir=tmp_path,
            min_ytd_pa=0,  # keep all rows for testing
        )

        # Schema contains core ytd / ros cols
        for c in ("pa_ytd", "hr_ytd", "obp_ytd", "slg_ytd",
                  "hr_per_pa_ytd", "ros_pa", "ros_hr", "ros_obp", "ros_slg",
                  "hr_per_pa_ytd", "ros_hr_per_pa", "trail4w_pa"):
            assert c in result.columns, f"missing column {c}"

        # Parquet round-trip succeeded
        round_trip = pd.read_parquet(tmp_path / "weekly_snapshots_2024.parquet")
        assert len(round_trip) == len(result)

        # PA-weighted sum identity: for each player,
        #   Σ(pa_week) == max(pa_ytd) == (pa_ytd + ros_pa)[any row]
        for pid, grp in result.groupby("mlbam_id"):
            season_pa = grp["pa_week"].sum()
            assert grp["pa_ytd"].max() == season_pa
            for _, row in grp.iterrows():
                assert row["pa_ytd"] + row["ros_pa"] == pytest.approx(season_pa)

    def test_min_ytd_pa_filter(self, tmp_path):
        rows = [
            _make_weekly_batting_row(100, 2024, 14, pa=10, season=2024),
            _make_weekly_batting_row(100, 2024, 15, pa=12, season=2024),
            _make_weekly_batting_row(100, 2024, 16, pa=15, season=2024),
            _make_weekly_batting_row(100, 2024, 17, pa=20, season=2024),
        ]
        batting_wk = _derive_singles(pd.DataFrame(rows))
        batting_wk.to_parquet(tmp_path / "batting_week_2024.parquet", index=False)

        result = build_weekly_snapshots(
            2024, raw_dir=tmp_path, min_ytd_pa=30,
        )
        # Rows with pa_ytd in {10, 22} are below 30 → dropped.
        # Rows with pa_ytd in {37, 57} are kept.
        assert len(result) == 2
        assert (result["pa_ytd"] >= 30).all()

    def test_no_leakage_ytd_uses_only_past(self, tmp_path):
        """ytd at week w must equal Σ(weekly values for weeks ≤ w)."""
        rows = [
            _make_weekly_batting_row(100, 2024, 14, pa=10, hr=1, season=2024),
            _make_weekly_batting_row(100, 2024, 15, pa=15, hr=2, season=2024),
            _make_weekly_batting_row(100, 2024, 16, pa=20, hr=3, season=2024),
        ]
        batting_wk = _derive_singles(pd.DataFrame(rows))
        batting_wk.to_parquet(tmp_path / "batting_week_2024.parquet", index=False)

        result = build_weekly_snapshots(
            2024, raw_dir=tmp_path, min_ytd_pa=0,
        ).sort_values("iso_week")

        assert list(result["hr_ytd"]) == [1, 3, 6]
        assert list(result["pa_ytd"]) == [10, 25, 45]


# ---------------------------------------------------------------------------
# Regression tests for PR #3 review feedback
# ---------------------------------------------------------------------------


class TestMergeDedup:
    """When both sides share non-key columns, the merge must not emit _x/_y suffixes."""

    def test_week_start_date_not_duplicated(self):
        batting = pd.DataFrame({
            "mlbam_id": [100], "iso_year": [2024], "iso_week": [15],
            "week_start_date": [pd.Timestamp("2024-04-08")],
            "week_end_date": [pd.Timestamp("2024-04-14")],
            "season": [2024], "pa": [20],
        })
        statcast = pd.DataFrame({
            "mlbam_id": [100], "iso_year": [2024], "iso_week": [15],
            # Statcast's week_start_date is the first-game date, not Monday
            "week_start_date": [pd.Timestamp("2024-04-09")],
            "season": [2024], "bbe_count": [10], "avg_exit_velocity": [90.0],
        })
        merged = _merge_weekly_sources(batting, statcast)
        assert "week_start_date_x" not in merged.columns
        assert "week_start_date_y" not in merged.columns
        # BRef (Monday) value is canonical
        assert merged["week_start_date"].iloc[0] == pd.Timestamp("2024-04-08")


class TestMidWeekTradeDedup:
    """BRef emits one row per team for a mid-week trade. The weekly merge
    must collapse those into a single row (summing counts) instead of
    crashing the ``one_to_one`` validation.
    """

    def test_duplicate_keys_get_aggregated(self):
        # Player 100 appears twice for week 15 — pre-trade (NYY) and post-trade (BOS).
        batting = pd.DataFrame({
            "mlbam_id": [100, 100, 200],
            "iso_year": [2024, 2024, 2024],
            "iso_week": [15, 15, 15],
            "team": ["NYY", "BOS", "CHC"],
            "week_start_date": [pd.Timestamp("2024-04-08")] * 3,
            "week_end_date": [pd.Timestamp("2024-04-14")] * 3,
            "season": [2024, 2024, 2024],
            "pa": [10, 15, 20],
            "ab": [9, 13, 18],
            "hr": [1, 0, 2],
            "bb": [1, 2, 2],
            "h": [3, 4, 6],
        })
        statcast = pd.DataFrame({
            "mlbam_id": [100, 200],
            "iso_year": [2024, 2024],
            "iso_week": [15, 15],
            "bbe_count": [5, 8],
            "avg_exit_velocity": [92.0, 88.0],
        })
        merged = _merge_weekly_sources(batting, statcast)
        # One row per (mlbam_id, iso_year, iso_week)
        assert len(merged) == 2
        player_100 = merged[merged["mlbam_id"] == 100].iloc[0]
        # Counts summed across the two teams (10 + 15 = 25 PA, etc.)
        assert player_100["pa"] == 25
        assert player_100["ab"] == 22
        assert player_100["hr"] == 1
        assert player_100["bb"] == 3
        assert player_100["h"] == 7

    def test_no_duplicates_is_noop(self):
        # Non-duplicate input should pass through unchanged (modulo
        # the usual merge).
        batting = pd.DataFrame({
            "mlbam_id": [100, 200],
            "iso_year": [2024, 2024],
            "iso_week": [15, 15],
            "pa": [20, 22],
            "hr": [1, 0],
            "season": [2024, 2024],
            "week_start_date": [pd.Timestamp("2024-04-08")] * 2,
            "week_end_date": [pd.Timestamp("2024-04-14")] * 2,
        })
        statcast = pd.DataFrame({
            "mlbam_id": [100, 200],
            "iso_year": [2024, 2024],
            "iso_week": [15, 15],
            "bbe_count": [5, 8],
            "avg_exit_velocity": [92.0, 88.0],
        })
        merged = _merge_weekly_sources(batting, statcast)
        assert len(merged) == 2
        assert set(merged["mlbam_id"]) == {100, 200}


class TestBbeValidMaskDenominator:
    """BBE-weighted YTD must not deflate when a week has BBEs but a NaN rate."""

    def test_ytd_ignores_nan_rate_weeks(self, tmp_path):
        batting_rows = [
            _make_weekly_batting_row(100, 2024, 15, pa=25, season=2024,
                                      singles=5, doubles=1, bb=2, so=5, hbp=0, sf=0),
            _make_weekly_batting_row(100, 2024, 16, pa=22, season=2024,
                                      singles=4, doubles=2, bb=3, so=4, hbp=0, sf=0),
            _make_weekly_batting_row(100, 2024, 17, pa=28, season=2024,
                                      singles=6, doubles=0, bb=4, so=6, hbp=0, sf=0),
        ]
        batting = _derive_singles(pd.DataFrame(batting_rows))
        batting.to_parquet(tmp_path / "batting_week_2024.parquet", index=False)

        statcast = pd.DataFrame({
            "mlbam_id": [100, 100, 100],
            "iso_year": [2024, 2024, 2024],
            "iso_week": [15, 16, 17],
            "season": [2024, 2024, 2024],
            "bbe_count": [10, 8, 12],
            # xwOBA valid in weeks 15 and 17, missing (NaN) in week 16 despite BBE > 0
            "estimated_woba_using_speedangle": [0.400, np.nan, 0.350],
            "avg_exit_velocity": [92.0, 91.0, 93.0],
        })
        statcast.to_parquet(tmp_path / "statcast_agg_week_2024.parquet", index=False)

        result = build_weekly_snapshots(
            2024, raw_dir=tmp_path, min_ytd_pa=0,
        ).sort_values("iso_week").reset_index(drop=True)

        # Week 15: first valid xwOBA  → ytd = 0.400
        # Week 16: xwOBA NaN; ytd must not be deflated by the week-16 BBEs.
        #          Valid denom = 10 (only week 15), numerator = 10*0.400
        #          → ytd remains 0.400
        # Week 17: valid denom = 10+12=22, numerator = 10*0.400 + 12*0.350 = 8.2
        #          → ytd = 8.2 / 22 ≈ 0.3727
        xwoba_ytd = result["estimated_woba_using_speedangle_ytd"].tolist()
        assert xwoba_ytd[0] == pytest.approx(0.400, rel=1e-6)
        assert xwoba_ytd[1] == pytest.approx(0.400, rel=1e-6)
        assert xwoba_ytd[2] == pytest.approx(8.2 / 22, rel=1e-6)

        # Sanity: avg_exit_velocity (never NaN) uses the full BBE denominator.
        ev_ytd = result["avg_exit_velocity_ytd"].tolist()
        assert ev_ytd[0] == pytest.approx(92.0)
        assert ev_ytd[1] == pytest.approx((10 * 92.0 + 8 * 91.0) / 18, rel=1e-6)
        assert ev_ytd[2] == pytest.approx(
            (10 * 92.0 + 8 * 91.0 + 12 * 93.0) / 30, rel=1e-6,
        )


class TestSeasonDatesRegularSeasonOnly:
    """``_SEASON_DATES`` must end on-or-before the real regular-season cutoff.

    The in-season pipeline treats the season-window end date as the boundary
    for ROS targets and ISO-week enumeration. Postseason dates would inflate
    the ytd counts for playoff teams and bleed playoff games into ROS targets.
    """

    def test_end_dates_are_regular_season_only(self):
        from src.data.fetch_statcast import _SEASON_DATES
        from datetime import date

        # Upper bound for regular-season end in any year: MLB has never
        # extended regular season past early October.
        for year, (_, end_s) in _SEASON_DATES.items():
            end = date.fromisoformat(end_s)
            # Every MLB regular season has ended on or before October 6th.
            assert end.month <= 10, f"{year}: season end {end} is past October"
            if end.month == 10:
                assert end.day <= 6, f"{year}: {end} extends into postseason"


class TestFetchGameLogsFailsHard:
    """A per-week fetch error must abort the season, not silently drop a week."""

    def test_one_week_error_raises(self, tmp_path):
        call_count = {"n": 0}

        def flaky_batting_stats_range(start_dt, end_dt):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("Simulated transient BRef failure")
            return pd.DataFrame({
                "Name": ["A"], "Lev": ["Maj-AL"],
                "PA": ["10"], "mlbID": ["1"],
            })

        fake_pb = type("FakePB", (), {"batting_stats_range": staticmethod(flaky_batting_stats_range)})()

        # Only enumerate a few weeks by using a short season window
        short_dates = {2024: ("2024-04-08", "2024-04-28")}
        with patch("src.data.fetch_game_logs._SEASON_DATES", short_dates), \
             patch.dict("sys.modules", {"pybaseball": fake_pb}):
            with pytest.raises(RuntimeError, match="Failed to fetch"):
                fetch_batter_weekly_stats(
                    2024, out_dir=tmp_path, delay=0.0,
                )

        assert not (tmp_path / "batting_week_2024.parquet").exists()
