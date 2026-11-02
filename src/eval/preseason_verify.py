"""Independently verify a preseason evaluation from its committed outputs.

Needs no network, model checkpoints or source caches. Uses scikit-learn rather
than ``src.eval`` so a shared bug cannot hide in both paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

STATS = ["OBP", "SLG", "HR", "R", "RBI", "SB"]


def verify(cfg: dict, out: Path) -> int:
    """Recompute every saved metric from the player-level snapshots."""
    year, min_pa = cfg["year"], cfg["minimum_actual_pa"]
    naive_model = f"Naive {year - 1}"
    primary_name, skill_name = f"primary_{min_pa}pa", f"skill_per_pa_{min_pa}pa"
    alternate_name = f"earlier_hydra_export_{min_pa}pa"
    pairwise_prefix = f"pairwise_{min_pa}pa_"
    players = pd.read_csv(out / "player_comparison.csv").set_index("idfg")
    has_alt = (out / "earlier_hydra.csv").exists()
    if has_alt:
        earlier = pd.read_csv(out / "earlier_hydra.csv").set_index("idfg")
    metrics = pd.read_csv(out / "metrics_by_target.csv")
    summary = pd.read_csv(out / "summary_metrics.csv")
    coverage = pd.read_csv(out / "coverage.csv")
    models = coverage.Model.tolist()
    assert players.index.is_unique
    if has_alt:
        assert earlier.index.is_unique
        assert players.index.sort_values().equals(earlier.index.sort_values())
    assert not metrics.duplicated(["cohort", "model", "target"]).any()
    valid = {}
    for model in models:
        columns = [f"{model} {s}" for s in [*STATS, "PA"]]
        valid[model] = players[columns].notna().all(axis=1)
        valid[model] &= players[f"{model} PA"] > 0
        row = coverage[coverage.Model.eq(model)].iloc[0]
        assert valid[model].sum() == row["Hydra overlap"]
        assert (valid[model] & players.PA.ge(min_pa)).sum() == row[
            f"Overlap >={min_pa} PA"
        ]
    common = pd.DataFrame(valid).all(axis=1)
    primary = common & players.PA.ge(min_pa)
    pd.testing.assert_series_equal(common, players.common_all_models, check_names=False)
    pd.testing.assert_series_equal(primary, players[primary_name], check_names=False)

    # Recheck all official-source rows, including absence-based zero outcomes.
    fg = pd.read_csv(out / f"actuals_{year}_fangraphs.csv").set_index("idfg")
    assert fg.index.is_unique
    joined: pd.DataFrame | list = []
    missing = ~players.index.isin(fg.index)
    if (out / f"actuals_{year}_mlb.csv").exists():
        joined = _check_mlb(out, fg, year)
        assert not players.loc[missing, "mlbam_id"].isin(joined.mlbam_id).any()
    matched_actuals = fg.reindex(players.index)[[*STATS, "PA"]]
    matched_actuals.loc[missing, [*STATS[2:], "PA"]] = 0
    np.testing.assert_allclose(matched_actuals, players[[*STATS, "PA"]], atol=1e-12)

    def arrays(cohort: str, model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = primary
        if cohort == "all_common_counts":
            mask = common
        elif cohort.startswith("sensitivity_"):
            threshold = int(cohort.removeprefix("sensitivity_").removesuffix("pa"))
            mask = common & players.PA.ge(threshold)
        elif cohort.startswith(pairwise_prefix):
            other = cohort.removeprefix(pairwise_prefix)
            mask = valid[other] & players.PA.ge(min_pa)
        elif cohort not in {primary_name, skill_name, alternate_name}:
            raise ValueError(f"Unknown cohort: {cohort}")
        subset = players.loc[mask].sort_index()
        stats = STATS[2:] if cohort == "all_common_counts" else STATS
        truth = subset[stats].to_numpy(float, copy=True)
        pred = subset[[f"{model} {s}" for s in stats]].to_numpy(float, copy=True)
        naive_pred = subset[[f"{naive_model} {s}" for s in stats]].to_numpy(
            float, copy=True
        )
        if cohort == alternate_name and model == "Hydra":
            pred = earlier.loc[subset.index, stats].to_numpy(float, copy=True)
        if cohort == skill_name:
            truth[:, 2:] /= subset.PA.to_numpy()[:, None]
            pred[:, 2:] /= subset[f"{model} PA"].to_numpy()[:, None]
            naive_pred[:, 2:] /= subset[f"{naive_model} PA"].to_numpy()[:, None]
        return truth, pred, naive_pred

    def values(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        nonzero = truth != 0
        return {
            "rmse": float(mean_squared_error(truth, pred) ** 0.5),
            "mae": float(mean_absolute_error(truth, pred)),
            "r2": float(r2_score(truth, pred)),
            "mape": float(
                100 * np.abs((truth[nonzero] - pred[nonzero]) / truth[nonzero]).mean()
            ),
            "bias_pred_minus_actual": float((pred - truth).mean()),
            "mape_n": int(nonzero.sum()),
        }

    checked = 0
    for (cohort, model), rows in metrics.groupby(["cohort", "model"], sort=False):
        truth, pred, naive = arrays(cohort, model)
        stats = STATS[2:] if cohort == "all_common_counts" else STATS
        results = []
        for j, stat in enumerate(stats):
            target = stat + (
                "/PA" if cohort == skill_name and stat in STATS[2:] else ""
            )
            row = rows[rows.target.eq(target)].iloc[0]
            expected = values(truth[:, j], pred[:, j])
            expected["rmse_relative_to_naive"] = expected["rmse"] / np.sqrt(
                np.mean((truth[:, j] - naive[:, j]) ** 2)
            )
            assert row.n == len(truth)
            for key, value in expected.items():
                np.testing.assert_allclose(row[key], value, rtol=1e-10, atol=1e-12)
                checked += 1
            results.append(expected)
        aggregate = summary[summary.cohort.eq(cohort) & summary.model.eq(model)].iloc[0]
        assert aggregate.n == len(truth)
        for key in ["rmse", "mae", "r2", "mape", "rmse_relative_to_naive"]:
            column = (
                "mean_relative_rmse"
                if key == "rmse_relative_to_naive"
                else "mean_" + key
            )
            np.testing.assert_allclose(
                aggregate[column],
                np.mean([v[key] for v in results]),
                rtol=1e-10,
                atol=1e-12,
            )
            checked += 1

    for row in pd.read_csv(out / "playing_time_metrics.csv").to_dict("records"):
        subset = players.loc[primary if row["cohort"] == primary_name else common]
        expected = values(subset.PA.to_numpy(), subset[f"{row['model']} PA"].to_numpy())
        assert row["n"] == len(subset)
        for key in ["rmse", "mae", "r2", "mape", "bias_pred_minus_actual"]:
            np.testing.assert_allclose(row[key], expected[key], rtol=1e-10, atol=1e-12)
            checked += 1

    for row in pd.read_csv(out / "paired_bootstrap.csv").to_dict("records"):
        truth, pred, naive = arrays(row["cohort"], row["model"])
        _, hydra, _ = arrays(row["cohort"], "Hydra")
        scale = np.sqrt(np.mean((truth - naive) ** 2, axis=0))
        rng = np.random.default_rng(cfg["seed"])
        sample = rng.integers(
            0, len(truth), size=(cfg["bootstrap_samples"], len(truth))
        )
        errors = np.square(pred - truth)
        hydra_errors = np.square(hydra - truth)
        delta = (np.sqrt(errors[sample].mean(axis=1)) / scale).mean(axis=1)
        delta -= (np.sqrt(hydra_errors[sample].mean(axis=1)) / scale).mean(axis=1)
        point = np.mean(np.sqrt(errors.mean(axis=0)) / scale) - np.mean(
            np.sqrt(hydra_errors.mean(axis=0)) / scale
        )
        np.testing.assert_allclose(
            [row["delta_model_minus_hydra"], row["ci95_low"], row["ci95_high"]],
            [point, *np.quantile(delta, [0.025, 0.975])],
            rtol=1e-10,
            atol=1e-12,
        )
        checked += 3
    print(
        f"Verified {checked} metric/interval values, cohort membership, and"
        f" {len(joined)} MLB/FanGraphs matches."
    )
    return checked


def _check_mlb(out: Path, fg: pd.DataFrame, year: int) -> pd.DataFrame:
    """Recheck the saved FanGraphs/MLB cross-check against both snapshots."""
    mlb = pd.read_csv(out / f"actuals_{year}_mlb.csv")
    assert mlb.mlbam_id.is_unique
    joined = fg.merge(
        mlb,
        left_on="xMLBAMID",
        right_on="mlbam_id",
        validate="one_to_one",
        suffixes=("_fg", "_mlb"),
    )
    assert len(joined) == len(mlb)
    saved_checks = pd.read_csv(out / "source_crosscheck.csv").set_index("stat")
    for stat in [*STATS, "PA"]:
        diff = (joined[f"{stat}_fg"] - joined[f"{stat}_mlb"]).abs()
        tolerance = 0.00051 if stat in STATS[:2] else 0
        np.testing.assert_allclose(
            diff.max(), saved_checks.loc[stat, "maximum_difference"], atol=1e-12
        )
        assert (diff > tolerance).sum() == saved_checks.loc[
            stat, "outside_rounding_tolerance"
        ]
        assert len(diff) == saved_checks.loc[stat, "n"]
    return joined
