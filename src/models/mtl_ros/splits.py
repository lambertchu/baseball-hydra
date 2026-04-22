"""Walk-forward splits for weekly snapshots.

The ROS training loop treats ``(mlbam_id, season)`` as the atomic split unit:
every cutoff for a given player-season must land in exactly one split so the
model never "sees" a player-season at train time and validates on another
cutoff from the same trajectory.  Because seasons are disjoint integers and
the splits are season-boundary filters, this holds structurally — but we
assert it post-hoc to fail loudly on regressions.

This is intentionally kept separate from :mod:`src.data.splits` (which is
tuned for season-level preseason data): the ROS pipeline has a very
different granularity (weekly cutoffs, atomic player-season, no ``val_year
> train_end`` invariant during production backtests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for a walk-forward split.

    Parameters
    ----------
    train_end_season:
        Inclusive upper bound on train seasons.
    val_season, test_season:
        Optional single-season holdouts.  Omit to skip that split.
    id_col, season_col:
        Column names holding the player identifier and season year.
    """

    train_end_season: int
    val_season: int | None = None
    test_season: int | None = None
    id_col: str = "mlbam_id"
    season_col: str = "season"

    @classmethod
    def from_dict(cls, d: dict) -> SplitConfig:
        return cls(
            train_end_season=d["train_end_season"],
            val_season=d.get("val_season"),
            test_season=d.get("test_season"),
            id_col=d.get("id_col", "mlbam_id"),
            season_col=d.get("season_col", "season"),
        )

    def build(self, snapshots: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Convenience wrapper around :func:`walk_forward_split`."""
        return walk_forward_split(
            snapshots,
            train_end_season=self.train_end_season,
            val_season=self.val_season,
            test_season=self.test_season,
            id_col=self.id_col,
            season_col=self.season_col,
        )


def walk_forward_split(
    snapshots: pd.DataFrame,
    train_end_season: int,
    val_season: int | None = None,
    test_season: int | None = None,
    id_col: str = "mlbam_id",
    season_col: str = "season",
) -> dict[str, pd.DataFrame]:
    """Split snapshots into train/val/test by season boundaries.

    * Train: rows with ``season <= train_end_season``.
    * Val: rows with ``season == val_season`` (if provided).
    * Test: rows with ``season == test_season`` (if provided).

    Returns a dict keyed by ``"train"``, ``"val"``, ``"test"``; optional
    keys are omitted when their season argument is ``None``.  The resulting
    frames are reset-indexed copies.

    Raises
    ------
    ValueError
        If the post-hoc atomic-unit check finds any ``(id_col, season_col)``
        key in more than one split (should be impossible given disjoint
        season boundaries, but guards against silent regressions).
    """
    if season_col not in snapshots.columns:
        raise KeyError(f"Missing season column {season_col!r}")
    if id_col not in snapshots.columns:
        raise KeyError(f"Missing id column {id_col!r}")

    out: dict[str, pd.DataFrame] = {
        "train": snapshots.loc[snapshots[season_col] <= train_end_season]
        .copy()
        .reset_index(drop=True),
    }
    if val_season is not None:
        out["val"] = (
            snapshots.loc[snapshots[season_col] == val_season]
            .copy()
            .reset_index(drop=True)
        )
    if test_season is not None:
        out["test"] = (
            snapshots.loc[snapshots[season_col] == test_season]
            .copy()
            .reset_index(drop=True)
        )

    _assert_no_leakage(out, id_col=id_col, season_col=season_col)
    return out


def _assert_no_leakage(
    splits: dict[str, pd.DataFrame],
    id_col: str,
    season_col: str,
) -> None:
    """Post-hoc guard: no ``(id, season)`` key appears in two splits."""
    keysets: dict[str, set[tuple]] = {}
    for name, frame in splits.items():
        if len(frame) == 0:
            keysets[name] = set()
            continue
        keysets[name] = set(map(tuple, frame[[id_col, season_col]].to_numpy().tolist()))
    names = list(keysets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = keysets[a] & keysets[b]
            if overlap:
                raise ValueError(
                    f"walk_forward_split leaked {len(overlap)} "
                    f"({id_col}, {season_col}) keys between splits "
                    f"{a!r} and {b!r}; sample: {sorted(overlap)[:3]}"
                )


def cutoff_sample_subsample(
    snapshots: pd.DataFrame,
    rng: np.random.Generator,
    k_cutoffs_per_player: int | None = None,
) -> pd.DataFrame:
    """Optionally down-sample cutoffs per ``(player, season)``.

    Current implementation: ``k_cutoffs_per_player=None`` → passthrough.
    The signature is reserved for a future mode that samples ``k`` cutoffs
    per player-season to keep epoch-size manageable on large snapshot tables.
    """
    if k_cutoffs_per_player is None:
        return snapshots
    # Placeholder: actual subsampling will be added in a later change.
    raise NotImplementedError(
        "cutoff_sample_subsample with k_cutoffs_per_player != None is not "
        "implemented yet; pass None for the current passthrough behavior."
    )
