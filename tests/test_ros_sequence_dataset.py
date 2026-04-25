"""Tests for Phase 3 ROS cutoff sequence dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.ros.dataset import ROSCutoffSequenceDataset
from src.models.ros.features import compute_weekly_sequence_features


def _snapshot_rows() -> pd.DataFrame:
    rows: list[dict] = []
    for week, pa_ytd, hr_week in [(14, 60.0, 1.0), (15, 85.0, 2.0), (16, 110.0, 9.0)]:
        rows.append(
            {
                "mlbam_id": 7,
                "season": 2025,
                "iso_year": 2025,
                "iso_week": week,
                "pa_week": 25.0,
                "ab_week": 22.0,
                "h_week": 7.0,
                "singles_week": 4.0,
                "doubles_week": 1.0,
                "triples_week": 0.0,
                "hr_week": hr_week,
                "r_week": 4.0,
                "rbi_week": 5.0,
                "bb_week": 2.0,
                "so_week": 6.0,
                "hbp_week": 1.0,
                "sf_week": 0.0,
                "sb_week": 1.0,
                "cs_week": 0.0,
                "pa_ytd": pa_ytd,
                "ros_pa": 300.0 - pa_ytd,
                "ros_obp": 0.330,
                "ros_slg": 0.440,
                "ros_hr_per_pa": 0.040,
                "ros_r_per_pa": 0.120,
                "ros_rbi_per_pa": 0.130,
                "ros_sb_per_pa": 0.020,
            }
        )
    return pd.DataFrame(rows)


def _make_dataset(rows: pd.DataFrame) -> ROSCutoffSequenceDataset:
    seq = compute_weekly_sequence_features(rows).fillna(0.0)
    phase2_x = pd.DataFrame(
        np.ones((len(rows), 4)), columns=[f"f{i}" for i in range(4)]
    )
    targets = rows[
        [
            "ros_obp",
            "ros_slg",
            "ros_hr_per_pa",
            "ros_r_per_pa",
            "ros_rbi_per_pa",
            "ros_sb_per_pa",
        ]
    ]
    return ROSCutoffSequenceDataset(
        snapshots=rows,
        sequence_features=seq,
        phase2_features=phase2_x,
        targets=targets,
        pa_target=rows["ros_pa"],
        max_seq_len=8,
    )


def test_dataset_slices_only_history_up_to_cutoff() -> None:
    rows = _snapshot_rows()
    ds = _make_dataset(rows)

    before = ds[1]["seq"].clone()

    mutated = rows.copy()
    mutated.loc[2, "hr_week"] = 99.0
    ds_mutated = _make_dataset(mutated)
    after = ds_mutated[1]["seq"].clone()

    np.testing.assert_allclose(before.numpy(), after.numpy())
    assert int(ds[1]["seq_mask"].sum().item()) == 2


def test_dataset_preserves_row_key_alignment() -> None:
    rows = _snapshot_rows()
    ds = _make_dataset(rows)

    assert ds.row_keys[0] == (7, 2025, 2025, 14)
    assert ds.row_keys[1] == (7, 2025, 2025, 15)
    assert ds[1]["phase2_x"].shape == (4,)
    assert ds[1]["target"].shape == (6,)
    assert ds[1]["pa_target"].shape == (1,)
