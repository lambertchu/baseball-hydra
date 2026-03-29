"""Tests for the external projections fetch, load, and merge pipeline.

Covers:
- HTTP fetch with mocked responses (JSON API path)
- Caching and force-redownload logic
- Graceful error handling
- Schema normalization
- Loading cached CSVs
- Merging projections into the results table
- Comparison CSV output
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.fetch_projections import (
    DISPLAY_NAMES,
    PROJECTION_SYSTEMS,
    _normalize_columns,
    fetch_all_projections,
    fetch_projections,
    load_projections,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_projection_json(n: int = 5, system: str = "steamer") -> bytes:
    """Create a fake FanGraphs API JSON response."""
    rows = []
    for i in range(n):
        rows.append({
            "playerid": 1000 + i,
            "PlayerName": f"Player {i}",
            "PA": 500 + i * 10,
            "OBP": round(0.320 + i * 0.005, 3),
            "SLG": round(0.420 + i * 0.010, 3),
            "HR": 20 + i,
            "R": 70 + i * 2,
            "RBI": 65 + i * 3,
            "SB": 5 + i,
        })
    return json.dumps(rows).encode("utf-8")


def _make_projection_df(n: int = 5, system: str = "steamer") -> pd.DataFrame:
    """Create a projection DataFrame in the normalized schema."""
    return pd.DataFrame({
        "idfg": [1000 + i for i in range(n)],
        "name": [f"Player {i}" for i in range(n)],
        "pa": [500 + i * 10 for i in range(n)],
        "obp": [round(0.320 + i * 0.005, 3) for i in range(n)],
        "slg": [round(0.420 + i * 0.010, 3) for i in range(n)],
        "hr": [20 + i for i in range(n)],
        "r": [70 + i * 2 for i in range(n)],
        "rbi": [65 + i * 3 for i in range(n)],
        "sb": [5 + i for i in range(n)],
        "season": 2026,
        "projection_system": system,
    })


def _write_projection_csv(
    tmp_path: Path, system: str = "steamer", year: int = 2026, n: int = 5,
) -> Path:
    """Write a sample projection CSV file and return its path."""
    df = _make_projection_df(n=n, system=system)
    path = tmp_path / f"{system}_{year}.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Fetch tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestFetchProjections:
    """Tests for fetch_projections with mocked HTTP."""

    def test_fetch_caches_csv(self, tmp_path: Path) -> None:
        """Successful fetch writes CSV and returns path."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = _make_projection_json()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("src.data.fetch_projections.urllib.request.urlopen", return_value=mock_resp):
            path = fetch_projections("steamer", 2026, out_dir=tmp_path, delay=0)

        assert path is not None
        assert path.exists()
        assert path.name == "steamer_2026.csv"
        df = pd.read_csv(path)
        assert "idfg" in df.columns
        assert "projection_system" in df.columns
        assert len(df) == 5

    def test_fetch_skips_existing(self, tmp_path: Path) -> None:
        """No-op when cached file exists and force=False."""
        _write_projection_csv(tmp_path, "steamer", 2026)

        with patch("src.data.fetch_projections.urllib.request.urlopen") as mock_open:
            path = fetch_projections("steamer", 2026, out_dir=tmp_path, delay=0)
            mock_open.assert_not_called()

        assert path is not None
        assert path.exists()

    def test_fetch_force_redownloads(self, tmp_path: Path) -> None:
        """Re-fetches when force=True even if file exists."""
        _write_projection_csv(tmp_path, "steamer", 2026, n=3)

        mock_resp = MagicMock()
        mock_resp.read.return_value = _make_projection_json(n=7)
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("src.data.fetch_projections.urllib.request.urlopen", return_value=mock_resp):
            path = fetch_projections("steamer", 2026, out_dir=tmp_path, force=True, delay=0)

        assert path is not None
        df = pd.read_csv(path)
        assert len(df) == 7

    def test_fetch_handles_api_error(self, tmp_path: Path) -> None:
        """Returns None and logs warning on HTTP error."""
        with patch(
            "src.data.fetch_projections.urllib.request.urlopen",
            side_effect=Exception("HTTP 403"),
        ):
            path = fetch_projections("steamer", 2026, out_dir=tmp_path, delay=0)

        assert path is None

    def test_fetch_normalizes_schema(self, tmp_path: Path) -> None:
        """Column names are normalized to lowercase project schema."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = _make_projection_json()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("src.data.fetch_projections.urllib.request.urlopen", return_value=mock_resp):
            path = fetch_projections("steamer", 2026, out_dir=tmp_path, delay=0)

        df = pd.read_csv(path)
        expected = {"idfg", "name", "pa", "obp", "slg", "hr", "r", "rbi", "sb", "season", "projection_system"}
        assert set(df.columns) == expected

    def test_fetch_all_projections(self, tmp_path: Path) -> None:
        """fetch_all_projections calls fetch for each system."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = _make_projection_json()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("src.data.fetch_projections.urllib.request.urlopen", return_value=mock_resp):
            results = fetch_all_projections(
                2026, systems=["steamer", "zips"], out_dir=tmp_path, delay=0,
            )

        assert len(results) == 2
        assert results["steamer"] is not None
        assert results["zips"] is not None

    def test_fetch_unknown_system(self, tmp_path: Path) -> None:
        """Unknown system returns None."""
        path = fetch_projections("unknown_system", 2026, out_dir=tmp_path, delay=0)
        assert path is None


# ---------------------------------------------------------------------------
# Load tests (pure data, no HTTP)
# ---------------------------------------------------------------------------

class TestLoadProjections:
    """Tests for load_projections reading cached files."""

    def test_load_reads_cached_files(self, tmp_path: Path) -> None:
        """Reads and concatenates available parquets."""
        _write_projection_csv(tmp_path, "steamer", 2026, n=3)
        _write_projection_csv(tmp_path, "zips", 2026, n=4)

        df = load_projections(2026, systems=["steamer", "zips"], out_dir=tmp_path)
        assert df is not None
        assert len(df) == 7
        assert set(df["projection_system"].unique()) == {"steamer", "zips"}

    def test_load_handles_missing_systems(self, tmp_path: Path) -> None:
        """Returns partial data when some systems are missing."""
        _write_projection_csv(tmp_path, "steamer", 2026, n=3)

        df = load_projections(
            2026, systems=["steamer", "thebat"], out_dir=tmp_path,
        )
        assert df is not None
        assert len(df) == 3
        assert df["projection_system"].unique().tolist() == ["steamer"]

    def test_load_returns_none_when_empty(self, tmp_path: Path) -> None:
        """Returns None when no projection files exist."""
        df = load_projections(2026, systems=["steamer"], out_dir=tmp_path)
        assert df is None


# ---------------------------------------------------------------------------
# Normalize columns test
# ---------------------------------------------------------------------------

class TestNormalizeColumns:
    """Tests for _normalize_columns."""

    def test_renames_fangraphs_columns(self) -> None:
        raw = pd.DataFrame({
            "playerid": [1, 2],
            "PlayerName": ["A", "B"],
            "PA": [500, 400],
            "OBP": [0.350, 0.300],
            "SLG": [0.450, 0.400],
            "HR": [25, 15],
            "R": [80, 60],
            "RBI": [75, 50],
            "SB": [10, 5],
        })
        df = _normalize_columns(raw)
        assert "idfg" in df.columns
        assert "name" in df.columns
        assert "obp" in df.columns
        assert "playerid" not in df.columns


# ---------------------------------------------------------------------------
# Merge / comparison tests
# ---------------------------------------------------------------------------

class TestMergeProjections:
    """Tests for _merge_projections helper in generate_projections."""

    @pytest.fixture()
    def _results(self) -> pd.DataFrame:
        """Minimal results table with _idfg column (as built in main())."""
        return pd.DataFrame({
            "_idfg": [100, 200, 300],
            "Player": ["Alice", "Bob", "Carol"],
            "Team": ["NYY", "LAD", "HOU"],
            "Ensemble OBP": [0.350, 0.320, 0.300],
            "Ensemble SLG": [0.500, 0.450, 0.400],
            "Ensemble HR": [30, 20, 15],
            "Ensemble R": [90, 75, 60],
            "Ensemble RBI": [85, 70, 55],
            "Ensemble SB": [10, 8, 5],
        })

    @pytest.fixture()
    def _projections(self) -> pd.DataFrame:
        """Two projection systems for three players."""
        return pd.DataFrame({
            "idfg": [100, 200, 300, 100, 200, 300],
            "name": ["Alice", "Bob", "Carol"] * 2,
            "obp": [0.345, 0.315, 0.295, 0.340, 0.310, 0.290],
            "slg": [0.490, 0.440, 0.395, 0.485, 0.435, 0.390],
            "hr": [28, 18, 14, 27, 17, 13],
            "r": [88, 73, 58, 86, 71, 56],
            "rbi": [83, 68, 53, 81, 66, 51],
            "sb": [9, 7, 4, 8, 6, 3],
            "projection_system": ["steamer"] * 3 + ["zips"] * 3,
            "season": [2026] * 6,
        })

    def test_merge_adds_projection_columns(self, _results, _projections) -> None:
        """Projection columns appear in merged results."""
        import importlib
        mod = importlib.import_module("scripts.generate_projections")
        merged = mod._merge_projections(_results, _projections)
        assert "Stmr OBP" in merged.columns
        assert "ZiPS HR" in merged.columns

    def test_merge_joins_on_idfg(self, _results, _projections) -> None:
        import importlib
        mod = importlib.import_module("scripts.generate_projections")
        merged = mod._merge_projections(_results, _projections)
        # Alice (idfg=100) should have Steamer OBP = 0.345
        assert merged.loc[0, "Stmr OBP"] == pytest.approx(0.345)

    def test_merge_handles_missing_players(self, _results) -> None:
        """Unmatched players get NaN projections."""
        import importlib
        mod = importlib.import_module("scripts.generate_projections")
        # Projections only for player 100 (Alice).
        projections = pd.DataFrame({
            "idfg": [100],
            "name": ["Alice"],
            "obp": [0.345],
            "slg": [0.490],
            "hr": [28],
            "r": [88],
            "rbi": [83],
            "sb": [9],
            "projection_system": ["steamer"],
            "season": [2026],
        })
        merged = mod._merge_projections(_results, projections)
        assert pd.isna(merged.loc[1, "Stmr OBP"])  # Bob not in projections

    def test_round_and_clip_on_projections(self, _results, _projections) -> None:
        """Rate stats rounded to 3dp, count stats clipped >= 0."""
        import importlib
        mod = importlib.import_module("scripts.generate_projections")
        merged = mod._merge_projections(_results, _projections)
        # OBP should be 3 decimals.
        obp_val = merged.loc[0, "Stmr OBP"]
        assert round(obp_val, 3) == obp_val
        # HR should be int >= 0.
        hr_val = merged.loc[0, "Stmr HR"]
        assert hr_val >= 0
        assert hr_val == int(hr_val)

    def test_merge_survives_sort(self) -> None:
        """Projections join correctly even after results are sorted."""
        import importlib
        mod = importlib.import_module("scripts.generate_projections")
        # Build results in non-alphabetical order, then sort.
        results = pd.DataFrame({
            "_idfg": [300, 100, 200],
            "Player": ["Carol", "Alice", "Bob"],
            "Ensemble OBP": [0.300, 0.350, 0.320],
            "Ensemble SLG": [0.400, 0.500, 0.450],
            "Ensemble HR": [15, 30, 20],
            "Ensemble R": [60, 90, 75],
            "Ensemble RBI": [55, 85, 70],
            "Ensemble SB": [5, 10, 8],
        })
        # Sort by OPS descending (like main() does).
        results["_ops"] = results["Ensemble OBP"] + results["Ensemble SLG"]
        results = results.sort_values("_ops", ascending=False).reset_index(drop=True)
        results = results.drop(columns=["_ops"])

        projections = pd.DataFrame({
            "idfg": [100, 200, 300],
            "name": ["Alice", "Bob", "Carol"],
            "hr": [40, 25, 10],
            "obp": [0.400, 0.350, 0.280],
            "slg": [0.550, 0.480, 0.370],
            "r": [100, 80, 55],
            "rbi": [110, 75, 45],
            "sb": [12, 9, 3],
            "projection_system": ["steamer"] * 3,
            "season": [2026] * 3,
        })
        merged = mod._merge_projections(results, projections)
        # After sort: Alice (idfg=100) is row 0, Bob (200) is row 1, Carol (300) is row 2.
        assert merged.loc[0, "Player"] == "Alice"
        assert merged.loc[0, "Stmr HR"] == 40
        assert merged.loc[1, "Player"] == "Bob"
        assert merged.loc[1, "Stmr HR"] == 25
        assert merged.loc[2, "Player"] == "Carol"
        assert merged.loc[2, "Stmr HR"] == 10

    def test_no_projections_graceful(self) -> None:
        """load_projections returns None when no files exist."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            result = load_projections(2026, systems=["steamer"], out_dir=td)
        assert result is None
