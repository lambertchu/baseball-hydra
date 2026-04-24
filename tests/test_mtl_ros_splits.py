"""Tests for walk-forward splitting of weekly snapshots.

The split treats ``(mlbam_id, season)`` as the atomic unit: no player-season
may appear in more than one split.  Because the split uses season boundaries
and seasons are disjoint integers, the leakage guarantee is structural —
but the implementation asserts it post-hoc to fail loudly on regressions.
"""

from __future__ import annotations

import pandas as pd

from src.models.mtl_ros.splits import (
    SplitConfig,
    walk_forward_split,
)


# ---------------------------------------------------------------------------
# Fixture helper
# ---------------------------------------------------------------------------


def _make_snapshots() -> pd.DataFrame:
    """Spanning seasons 2020-2024, multiple weeks per (player, season)."""
    rows: list[dict] = []
    for season in range(2020, 2025):
        for player in (100, 200, 300):
            for week in (15, 18, 22, 26):
                rows.append(
                    {
                        "mlbam_id": player,
                        "season": season,
                        "iso_year": season,
                        "iso_week": week,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# walk_forward_split
# ---------------------------------------------------------------------------


class TestWalkForwardSplit:
    def test_walk_forward_no_leakage(self):
        """No (mlbam_id, season) key can appear in more than one split."""
        snaps = _make_snapshots()
        splits = walk_forward_split(
            snaps,
            train_end_season=2022,
            val_season=2023,
            test_season=2024,
        )
        train_keys = set(map(tuple, splits["train"][["mlbam_id", "season"]].values))
        val_keys = set(map(tuple, splits["val"][["mlbam_id", "season"]].values))
        test_keys = set(map(tuple, splits["test"][["mlbam_id", "season"]].values))

        assert train_keys & val_keys == set()
        assert train_keys & test_keys == set()
        assert val_keys & test_keys == set()

    def test_walk_forward_season_boundaries(self):
        snaps = _make_snapshots()
        splits = walk_forward_split(
            snaps,
            train_end_season=2022,
            val_season=2023,
            test_season=2024,
        )
        assert (splits["train"]["season"] <= 2022).all()
        assert (splits["val"]["season"] == 2023).all()
        assert (splits["test"]["season"] == 2024).all()

    def test_walk_forward_val_optional(self):
        snaps = _make_snapshots()
        splits = walk_forward_split(
            snaps,
            train_end_season=2022,
            test_season=2024,
        )
        assert set(splits.keys()) == {"train", "test"}
        assert (splits["train"]["season"] <= 2022).all()
        assert (splits["test"]["season"] == 2024).all()

    def test_walk_forward_train_only(self):
        snaps = _make_snapshots()
        splits = walk_forward_split(snaps, train_end_season=2022)
        assert set(splits.keys()) == {"train"}
        assert (splits["train"]["season"] <= 2022).all()

    def test_walk_forward_preserves_row_counts(self):
        """Sum of split sizes ≤ total, and equals total when all seasons covered."""
        snaps = _make_snapshots()
        splits = walk_forward_split(
            snaps,
            train_end_season=2022,
            val_season=2023,
            test_season=2024,
        )
        total = sum(len(v) for v in splits.values())
        # Seasons 2020-2024 all covered.
        assert total == len(snaps)


# ---------------------------------------------------------------------------
# SplitConfig
# ---------------------------------------------------------------------------


class TestSplitConfig:
    def test_split_config_roundtrip(self):
        snaps = _make_snapshots()
        cfg = SplitConfig.from_dict(
            {
                "train_end_season": 2022,
                "val_season": 2023,
                "test_season": 2024,
            }
        )
        via_config = cfg.build(snaps)
        direct = walk_forward_split(
            snaps,
            train_end_season=2022,
            val_season=2023,
            test_season=2024,
        )
        # Keys match and frames are equivalent.
        assert set(via_config.keys()) == set(direct.keys())
        for k in direct:
            pd.testing.assert_frame_equal(
                via_config[k].reset_index(drop=True),
                direct[k].reset_index(drop=True),
            )

    def test_split_config_custom_id_cols(self):
        snaps = _make_snapshots().rename(columns={"mlbam_id": "pid", "season": "yr"})
        cfg = SplitConfig(
            train_end_season=2022,
            val_season=2023,
            test_season=2024,
            id_col="pid",
            season_col="yr",
        )
        splits = cfg.build(snaps)
        assert (splits["train"]["yr"] <= 2022).all()
        assert (splits["val"]["yr"] == 2023).all()
        assert (splits["test"]["yr"] == 2024).all()
