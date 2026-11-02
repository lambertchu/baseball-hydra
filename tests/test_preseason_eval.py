"""End-to-end tests for the reusable preseason evaluation on synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.eval.preseason import STATS, load_config, run
from src.eval.preseason_verify import verify

YEAR = 2027
N = 40


def _actuals(rng: np.random.Generator) -> pd.DataFrame:
    pa = rng.integers(50, 700, N).astype(float)
    return pd.DataFrame(
        {
            "idfg": np.arange(1000, 1000 + N),
            "mlbam_id": np.arange(500000, 500000 + N),
            "Player": [f"Player {i}" for i in range(N)],
            "OBP": rng.uniform(0.28, 0.40, N).round(3),
            "SLG": rng.uniform(0.35, 0.55, N).round(3),
            "HR": (pa * rng.uniform(0.01, 0.06, N)).round(),
            "R": (pa * rng.uniform(0.08, 0.16, N)).round(),
            "RBI": (pa * rng.uniform(0.08, 0.16, N)).round(),
            "SB": (pa * rng.uniform(0.0, 0.05, N)).round(),
            "PA": pa,
        }
    )


def _noisy(truth: pd.DataFrame, rng: np.random.Generator, scale: float) -> pd.DataFrame:
    out = truth.copy()
    for s in STATS:
        sd = truth[s].std() * scale
        out[s] = (truth[s] + rng.normal(0, sd, N)).clip(lower=0)
    out["PA"] = (truth.PA + rng.normal(0, 120 * scale, N)).clip(lower=20)
    return out


def _write_inputs(tmp: Path, alternate: bool, perfect: str | None) -> Path:
    rng = np.random.default_rng(0)
    truth = _actuals(rng)
    hydra = _noisy(truth, rng, 0.5)
    naive = _noisy(truth, rng, 1.0)
    export = pd.DataFrame(
        {
            "idfg": truth.idfg,
            "mlbam_id": truth.mlbam_id,
            "Player": truth.Player,
            f"{YEAR - 1} PA": naive.PA.round(),
            f"{YEAR} PA (proj)": hydra.PA.round(1),
            **{f"{YEAR - 1} {s}": naive[s] for s in STATS},
            **{f"MTL {s}": hydra[s] for s in STATS},
        }
    )
    export.to_csv(tmp / "hydra.csv", index=False)
    if alternate:
        export[["idfg", "Player", *[f"MTL {s}" for s in STATS]]].to_csv(
            tmp / "alt.csv", index=False
        )

    ext = tmp / "ext"
    ext.mkdir()
    # Schema saved by src.data.fetch_projections (normalized lowercase).
    steamer = truth if perfect == "steamer" else _noisy(truth, rng, 0.6)
    pd.DataFrame(
        {
            "idfg": truth.idfg,
            "name": truth.Player,
            **{s.lower(): steamer[s] for s in [*STATS, "PA"]},
        }
    ).to_csv(ext / f"steamer_{YEAR}.csv", index=False)
    # Manual FanGraphs export schema; one player has no forecast.
    zips = _noisy(truth, rng, 0.7)
    pd.DataFrame(
        {
            "PlayerName": truth.Player,
            "PlayerId": truth.idfg.astype(str),
            **{s: zips[s] for s in [*STATS, "PA"]},
        }
    ).iloc[1:].to_csv(ext / f"zips_{YEAR}.csv", index=False)

    # The last Hydra player has no season record in either source.
    played = truth.iloc[:-1]
    fg = {
        "data": [
            {
                "playerid": int(r.idfg),
                "Name": f"<a href='#'>{r.Player}</a>",
                "xMLBAMID": int(r.mlbam_id),
                "Season": YEAR,
                **{s: float(getattr(r, s)) for s in [*STATS, "PA"]},
            }
            for r in played.itertuples()
        ],
        "totalCount": len(played),
    }
    (tmp / "fg.json").write_text(json.dumps(fg))
    names = {
        "OBP": "obp",
        "SLG": "slg",
        "HR": "homeRuns",
        "R": "runs",
        "RBI": "rbi",
        "SB": "stolenBases",
        "PA": "plateAppearances",
    }
    splits = [
        {
            "player": {"id": int(r.mlbam_id), "fullName": r.Player},
            "stat": {
                v: f"{getattr(r, k):.3f}" if k in STATS[:2] else int(getattr(r, k))
                for k, v in names.items()
            },
        }
        for r in played.itertuples()
    ]
    (tmp / "mlb.json").write_text(
        json.dumps({"stats": [{"splits": splits, "totalSplits": len(splits)}]})
    )

    cfg = {
        "year": YEAR,
        "output_dir": str(tmp / "out"),
        "seed": 1,
        "bootstrap_samples": 50,
        "minimum_actual_pa": 200,
        "sensitivity_minimum_pa": [1, 500],
        "hydra": str(tmp / "hydra.csv"),
        "external_dir": str(ext),
        "systems": {"steamer": "Steamer", "zips": "ZiPS"},
        "actuals": str(tmp / "fg.json"),
        "crosscheck": str(tmp / "mlb.json"),
    }
    if alternate:
        cfg["hydra_alternate"] = str(tmp / "alt.csv")
    path = tmp / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.mark.parametrize("alternate", [True, False])
def test_run_writes_outputs_that_verify(tmp_path: Path, alternate: bool) -> None:
    cfg = load_config(_write_inputs(tmp_path, alternate, perfect=None))
    out = tmp_path / "out"
    manifest = run(cfg, out)

    assert manifest["hydra_id_source"] == "export"
    assert manifest["hydra_players"] == N
    assert manifest["zero_actual_players"] == [f"Player {N - 1}"]
    assert manifest["excluded_from_common"] == ["Player 0"]
    assert (out / "earlier_hydra.csv").exists() == alternate
    summary = pd.read_csv(out / "summary_metrics.csv")
    assert set(summary.model) == {"Hydra", f"Naive {YEAR - 1}", "Steamer", "ZiPS"}
    assert "primary_200pa" in set(summary.cohort)
    assert ("earlier_hydra_export_200pa" in set(summary.cohort)) == alternate
    assert verify(cfg, out) > 0


def test_report_text_follows_results(tmp_path: Path) -> None:
    cfg = load_config(_write_inputs(tmp_path, alternate=False, perfect="steamer"))
    out = tmp_path / "out"
    run(cfg, out)
    report = (out / "evaluation_report.md").read_text()

    leaderboard = pd.read_csv(out / "rmse_leaderboard.csv", index_col=0)
    assert leaderboard.index[0] == "Steamer"
    assert "Hydra ranks 2nd of 4 by mean RMSE" in report
    assert "The best other system by equal-stat score is Steamer" in report
    assert "the paired bootstrap supports a difference" in report
    assert f"Saved {YEAR} preseason forecasts" in report


def test_missing_public_system_fails_clearly(tmp_path: Path) -> None:
    path = _write_inputs(tmp_path, alternate=False, perfect=None)
    (tmp_path / "ext" / f"zips_{YEAR}.csv").unlink()
    with pytest.raises(FileNotFoundError, match="zips"):
        run(load_config(path), tmp_path / "out")
