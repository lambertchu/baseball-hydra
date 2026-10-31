"""Independently verify the committed 2026 analysis without local source caches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
STATS = ["OBP", "SLG", "HR", "R", "RBI", "SB"]


def main() -> None:
    cfg = yaml.safe_load((HERE / "evaluation.yaml").read_text())
    players = pd.read_csv(HERE / "player_comparison.csv").set_index("idfg")
    earlier = pd.read_csv(HERE / "earlier_hydra.csv").set_index("idfg")
    metrics = pd.read_csv(HERE / "metrics_by_target.csv")
    summary = pd.read_csv(HERE / "summary_metrics.csv")
    coverage = pd.read_csv(HERE / "coverage.csv")
    models = coverage.Model.tolist()
    assert players.index.is_unique and earlier.index.is_unique
    assert players.index.sort_values().equals(earlier.index.sort_values())
    assert not metrics.duplicated(["cohort", "model", "target"]).any()
    valid = {}
    for model in models:
        columns = [f"{model} {s}" for s in [*STATS, "PA"]]
        valid[model] = players[columns].notna().all(axis=1)
        valid[model] &= players[f"{model} PA"] > 0
        row = coverage[coverage.Model.eq(model)].iloc[0]
        assert valid[model].sum() == row["Hydra overlap"]
        assert (valid[model] & players.PA.ge(cfg["minimum_actual_pa"])).sum() == row[
            "Overlap >=200 PA"
        ]
    common = pd.DataFrame(valid).all(axis=1)
    primary = common & players.PA.ge(cfg["minimum_actual_pa"])
    pd.testing.assert_series_equal(common, players.common_all_models, check_names=False)
    pd.testing.assert_series_equal(primary, players.primary_200pa, check_names=False)

    # Recheck all official-source rows, including absence-based zero outcomes.
    fg = pd.read_csv(HERE / "actuals_2026_fangraphs.csv").set_index("idfg")
    mlb = pd.read_csv(HERE / "actuals_2026_mlb.csv")
    assert fg.index.is_unique and mlb.mlbam_id.is_unique
    joined = fg.merge(
        mlb,
        left_on="xMLBAMID",
        right_on="mlbam_id",
        validate="one_to_one",
        suffixes=("_fg", "_mlb"),
    )
    assert len(joined) == len(mlb)
    saved_checks = pd.read_csv(HERE / "source_crosscheck.csv").set_index("stat")
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
    matched_actuals = fg.reindex(players.index)[[*STATS, "PA"]]
    missing = ~players.index.isin(fg.index)
    assert not players.loc[missing, "mlbam_id"].isin(mlb.mlbam_id).any()
    matched_actuals.loc[missing, [*STATS[2:], "PA"]] = 0
    np.testing.assert_allclose(matched_actuals, players[[*STATS, "PA"]], atol=1e-12)

    def arrays(cohort: str, model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = primary
        if cohort == "all_common_counts":
            mask = common
        elif cohort.startswith("sensitivity_"):
            threshold = int(cohort.removeprefix("sensitivity_").removesuffix("pa"))
            mask = common & players.PA.ge(threshold)
        elif cohort.startswith("pairwise_200pa_"):
            other = cohort.removeprefix("pairwise_200pa_")
            mask = valid[other] & players.PA.ge(cfg["minimum_actual_pa"])
        elif cohort not in {
            "primary_200pa",
            "skill_per_pa_200pa",
            "earlier_hydra_export_200pa",
        }:
            raise ValueError(f"Unknown cohort: {cohort}")
        subset = players.loc[mask].sort_index()
        stats = STATS[2:] if cohort == "all_common_counts" else STATS
        truth = subset[stats].to_numpy(float, copy=True)
        pred = subset[[f"{model} {s}" for s in stats]].to_numpy(float, copy=True)
        naive = subset[[f"Naive 2025 {s}" for s in stats]].to_numpy(float, copy=True)
        if cohort == "earlier_hydra_export_200pa" and model == "Hydra":
            pred = earlier.loc[subset.index, stats].to_numpy(float, copy=True)
        if cohort == "skill_per_pa_200pa":
            truth[:, 2:] /= subset.PA.to_numpy()[:, None]
            pred[:, 2:] /= subset[f"{model} PA"].to_numpy()[:, None]
            naive[:, 2:] /= subset["Naive 2025 PA"].to_numpy()[:, None]
        return truth, pred, naive

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
                "/PA" if cohort == "skill_per_pa_200pa" and stat in STATS[2:] else ""
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

    for row in pd.read_csv(HERE / "playing_time_metrics.csv").to_dict("records"):
        subset = players.loc[primary if row["cohort"] == "primary_200pa" else common]
        expected = values(subset.PA.to_numpy(), subset[f"{row['model']} PA"].to_numpy())
        assert row["n"] == len(subset)
        for key in ["rmse", "mae", "r2", "mape", "bias_pred_minus_actual"]:
            np.testing.assert_allclose(row[key], expected[key], rtol=1e-10, atol=1e-12)
            checked += 1

    for row in pd.read_csv(HERE / "paired_bootstrap.csv").to_dict("records"):
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
        f"Verified {checked} metric/interval values, cohort membership, and {len(joined)} MLB/FanGraphs matches."
    )


if __name__ == "__main__":
    main()
