"""Score saved forecasts without training or overwriting any input.

Run from repository root: .venv/bin/python data/reports/preseason_2026/evaluate.py
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.eval.metrics import compute_metrics  # noqa: E402
from src.eval.pa_projection import project_pa  # noqa: E402
from src.features.temporal import compute_temporal_features  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent
STATS = ["OBP", "SLG", "HR", "R", "RBI", "SB"]
COUNTS = STATS[2:]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(df: pd.DataFrame, digits: int = 4) -> str:
    def fmt(v: object) -> str:
        return f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)

    rows = [
        "| " + " | ".join(map(str, df.columns)) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    rows += [
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join(rows)


def main() -> None:
    cfg = yaml.safe_load((OUT / "evaluation.yaml").read_text())
    source_paths = [
        ROOT / cfg[k]
        for k in [
            "hydra",
            "hydra_earlier",
            "merged",
            "previous_actuals",
            "actuals",
            "crosscheck",
        ]
    ]
    source_paths += [
        ROOT / cfg["external_dir"] / f"{s}_{cfg['year']}.csv" for s in cfg["systems"]
    ]
    inputs = [
        {
            "path": str(p.relative_to(ROOT)),
            "sha256": digest(p),
            "modified_utc": datetime.fromtimestamp(
                p.stat().st_mtime, timezone.utc
            ).isoformat(),
        }
        for p in source_paths
    ]
    hydra = pd.read_csv(ROOT / cfg["hydra"])
    prev = pd.read_parquet(ROOT / cfg["previous_actuals"]).query("pa >= 300")
    mapped = hydra.merge(
        prev[["name", "idfg", "pa"]],
        left_on=["Player", "2025 PA"],
        right_on=["name", "pa"],
        how="left",
        validate="one_to_one",
    )
    assert mapped.idfg.notna().all() and len(mapped) == len(hydra)
    mapped.idfg = mapped.idfg.astype(int)
    mapped = mapped.set_index("idfg")
    merged = pd.read_parquet(ROOT / cfg["merged"])
    assert merged.season.max() < cfg["year"]
    temporal = compute_temporal_features(merged, stats=["pa"])
    pa_frame = temporal[temporal.season.eq(cfg["year"] - 1)].set_index("idfg")
    pa = project_pa(pa_frame).reindex(mapped.index)
    assert pa.notna().all()
    predictions = {}
    for label, prefix in [("Hydra", "MTL"), ("Naive 2025", "2025")]:
        df = mapped[[f"{prefix} {s}" for s in STATS]].copy()
        df.columns = STATS
        df["PA"] = pa if label == "Hydra" else mapped["2025 PA"]
        predictions[label] = df
    earlier = pd.read_csv(ROOT / cfg["hydra_earlier"])
    earlier = earlier.merge(
        mapped.reset_index()[["idfg", "Player"]],
        on="Player",
        how="left",
        validate="one_to_one",
    ).set_index("idfg")
    assert earlier.index.notna().all()
    earlier = earlier[[f"MTL {s}" for s in STATS]].rename(
        columns={f"MTL {s}": s for s in STATS}
    )
    earlier["PA"] = pa
    earlier.to_csv(OUT / "earlier_hydra.csv", index_label="idfg")
    for system, label in cfg["systems"].items():
        df = pd.read_csv(ROOT / cfg["external_dir"] / f"{system}_{cfg['year']}.csv")
        df["idfg"] = pd.to_numeric(df.PlayerId, errors="coerce")
        df = df.dropna(subset=["idfg"])
        df.idfg = df.idfg.astype(int)
        assert not df.idfg.duplicated().any()
        df = df.set_index("idfg")[STATS + ["PA"]].apply(pd.to_numeric)
        predictions[label] = df.dropna().query("PA > 0")

    raw = json.loads((ROOT / cfg["actuals"]).read_text())
    actual = pd.DataFrame(raw["data"])
    assert len(actual) == raw["totalCount"] and actual.Season.eq(cfg["year"]).all()
    assert not actual.playerid.duplicated().any()
    actual["Player"] = actual.Name.map(
        lambda s: html.unescape(re.sub("<[^>]+>", "", s))
    )
    actual = actual.set_index("playerid")
    actual.index = actual.index.astype(int)
    official_raw = json.loads((ROOT / cfg["crosscheck"]).read_text())["stats"][0]
    assert len(official_raw["splits"]) == official_raw["totalSplits"]
    official = pd.DataFrame(
        [
            {
                "mlbam_id": s["player"]["id"],
                "Player": s["player"]["fullName"],
                **s["stat"],
            }
            for s in official_raw["splits"]
        ]
    )
    assert not official.mlbam_id.duplicated().any()
    official.rename(
        columns={
            "obp": "OBP",
            "slg": "SLG",
            "homeRuns": "HR",
            "runs": "R",
            "rbi": "RBI",
            "stolenBases": "SB",
            "plateAppearances": "PA",
        }
    )[["mlbam_id", "Player", *STATS, "PA"]].to_csv(
        OUT / "actuals_2026_mlb.csv", index=False
    )
    check = actual.merge(
        official,
        left_on="xMLBAMID",
        right_on="mlbam_id",
        validate="one_to_one",
        suffixes=("_fg", "_mlb"),
    )
    checks = []
    for fg, mlb in [
        ("PA", "plateAppearances"),
        ("HR", "homeRuns"),
        ("R", "runs"),
        ("RBI", "rbi"),
        ("SB", "stolenBases"),
        ("OBP", "obp"),
        ("SLG", "slg"),
    ]:
        delta = (check[fg].astype(float) - check[mlb].astype(float)).abs()
        tolerance = 0.00051 if fg in STATS[:2] else 0
        checks.append(
            {
                "stat": fg,
                "n": len(check),
                "maximum_difference": delta.max(),
                "outside_rounding_tolerance": int((delta > tolerance).sum()),
            }
        )
        if fg != "PA":
            assert (delta <= tolerance).all(), fg
    pd.DataFrame(checks).to_csv(OUT / "source_crosscheck.csv", index=False)

    missing = mapped.index.difference(actual.index)
    missing_mlb = pa_frame.loc[missing, "mlbam_id"]
    assert not missing_mlb.isin(official.mlbam_id).any()
    truth = actual.reindex(mapped.index)[STATS + ["PA"]].copy()
    truth.loc[missing, COUNTS + ["PA"]] = 0
    truth.insert(0, "Player", mapped.Player)
    truth.to_csv(OUT / "actuals_hydra_players.csv", index_label="idfg")
    actual[["Player", "xMLBAMID"] + STATS + ["PA"]].to_csv(
        OUT / "actuals_2026_fangraphs.csv", index_label="idfg"
    )
    common = mapped.index
    for pred in predictions.values():
        common = common.intersection(pred.index)
    common = common.sort_values()
    primary = common[truth.loc[common, "PA"].ge(cfg["minimum_actual_pa"])]
    cover = []
    for label, pred in predictions.items():
        available = mapped.index.intersection(pred.index)
        cover.append(
            {
                "Model": label,
                "Saved players": len(pred),
                "Hydra overlap": len(available),
                "Overlap >=200 PA": int(truth.loc[available, "PA"].ge(200).sum()),
                "Missing Hydra players": "; ".join(
                    mapped.loc[mapped.index.difference(pred.index), "Player"]
                ),
            }
        )
    pd.DataFrame(cover).to_csv(OUT / "coverage.csv", index=False)
    matched = truth.copy()
    matched["mlbam_id"] = pa_frame.loc[matched.index, "mlbam_id"].astype(int)
    matched["common_all_models"] = matched.index.isin(common)
    matched["primary_200pa"] = matched.index.isin(primary)
    for label, pred in predictions.items():
        matched = matched.join(pred.add_prefix(label + " "))
    matched.to_csv(OUT / "player_comparison.csv", index_label="idfg")

    metrics, aggregate, bootstrap = [], [], []
    cohorts = [
        ("primary_200pa", primary, STATS, False),
        ("all_common_counts", common, COUNTS, False),
        ("skill_per_pa_200pa", primary, STATS, True),
    ]
    for threshold in cfg["sensitivity_minimum_pa"]:
        ids = common[truth.loc[common, "PA"].ge(threshold)]
        cohorts.append((f"sensitivity_{threshold}pa", ids, STATS, False))

    def evaluate(
        cohort: str,
        ids: pd.Index,
        stats: list[str],
        rates: bool,
        forecasts: dict[str, pd.DataFrame],
    ) -> None:
        yt = truth.loc[ids, stats].to_numpy(float, copy=True)
        count_positions = [i for i, s in enumerate(stats) if s in COUNTS]
        if rates:
            yt[:, count_positions] /= truth.loc[ids, "PA"].to_numpy()[:, None]
        pred_arrays = {}
        for label, df in forecasts.items():
            yp = df.loc[ids, stats].to_numpy(float, copy=True)
            if rates:
                yp[:, count_positions] /= df.loc[ids, "PA"].to_numpy()[:, None]
            assert np.isfinite(yt).all() and np.isfinite(yp).all()
            pred_arrays[label] = yp
        scale = np.sqrt(np.mean((yt - pred_arrays["Naive 2025"]) ** 2, axis=0))
        assert (scale > 0).all()
        for label, yp in pred_arrays.items():
            result = compute_metrics(yt, yp, stats)
            per_rmse = []
            for i, s in enumerate(stats):
                m = result["per_target"][s]
                per_rmse.append(m["rmse"])
                metrics.append(
                    {
                        "cohort": cohort,
                        "model": label,
                        "n": len(ids),
                        "target": s + ("/PA" if rates and s in COUNTS else ""),
                        **m,
                        "bias_pred_minus_actual": float(np.mean(yp[:, i] - yt[:, i])),
                        "mape_n": int((yt[:, i] != 0).sum()),
                        "rmse_relative_to_naive": m["rmse"] / scale[i],
                    }
                )
            aggregate.append(
                {
                    "cohort": cohort,
                    "model": label,
                    "n": len(ids),
                    "mean_relative_rmse": float(np.mean(np.array(per_rmse) / scale)),
                    **{"mean_" + k: v for k, v in result["aggregate"].items()},
                }
            )
        if cohort in {"primary_200pa", "all_common_counts", "skill_per_pa_200pa"}:
            rng = np.random.default_rng(cfg["seed"])
            indices = rng.integers(0, len(ids), (cfg["bootstrap_samples"], len(ids)))
            hydra_errors = (pred_arrays["Hydra"] - yt) ** 2
            hydra_scores = np.mean(
                np.sqrt(hydra_errors[indices].mean(axis=1)) / scale, axis=1
            )
            for label, yp in pred_arrays.items():
                if label == "Hydra":
                    continue
                score = np.mean(
                    np.sqrt(((yp - yt) ** 2)[indices].mean(axis=1)) / scale, axis=1
                )
                delta = score - hydra_scores
                bootstrap.append(
                    {
                        "cohort": cohort,
                        "model": label,
                        "delta_model_minus_hydra": float(
                            np.mean(np.sqrt(np.mean((yp - yt) ** 2, axis=0)) / scale)
                            - np.mean(np.sqrt(hydra_errors.mean(axis=0)) / scale)
                        ),
                        "ci95_low": float(np.quantile(delta, 0.025)),
                        "ci95_high": float(np.quantile(delta, 0.975)),
                    }
                )

    for cohort, ids, stats, rates in cohorts:
        evaluate(cohort, ids, stats, rates, predictions)
    earlier_predictions = {**predictions, "Hydra": earlier}
    evaluate("earlier_hydra_export_200pa", primary, STATS, False, earlier_predictions)
    for label, pred in predictions.items():
        ids = mapped.index.intersection(pred.index)
        ids = ids[truth.loc[ids, "PA"].ge(cfg["minimum_actual_pa"])]
        evaluate(
            "pairwise_200pa_" + label,
            ids,
            STATS,
            False,
            {
                "Hydra": predictions["Hydra"],
                "Naive 2025": predictions["Naive 2025"],
                label: pred,
            },
        )
    met = pd.DataFrame(metrics)
    agg = pd.DataFrame(aggregate)
    met.to_csv(OUT / "metrics_by_target.csv", index=False)
    agg.to_csv(OUT / "summary_metrics.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(OUT / "paired_bootstrap.csv", index=False)
    pa_metrics = []
    for cohort, ids in [("primary_200pa", primary), ("all_common", common)]:
        for label, df in predictions.items():
            yt, yp = truth.loc[ids, "PA"].to_numpy(), df.loc[ids, "PA"].to_numpy()
            m = compute_metrics(yt, yp, ["PA"])["per_target"]["PA"]
            pa_metrics.append(
                {
                    "cohort": cohort,
                    "model": label,
                    "n": len(ids),
                    **m,
                    "bias_pred_minus_actual": float(np.mean(yp - yt)),
                }
            )
    pd.DataFrame(pa_metrics).to_csv(OUT / "playing_time_metrics.csv", index=False)

    order = (
        agg.query("cohort == 'primary_200pa'")
        .sort_values("mean_relative_rmse")
        .model.tolist()
    )
    rmse_table = (
        met.query("cohort == 'primary_200pa'")
        .pivot(index="model", columns="target", values="rmse")
        .loc[order, STATS]
    )
    score = agg.query("cohort == 'primary_200pa'").set_index("model").loc[order]
    rmse_table["Equal-stat score"] = score.mean_relative_rmse
    rmse_table["Mean RMSE"] = score.mean_rmse
    rmse_table.to_csv(OUT / "rmse_leaderboard.csv")
    ratio = (
        met.query("cohort == 'primary_200pa'")
        .pivot(index="model", columns="target", values="rmse_relative_to_naive")
        .loc[order, STATS]
    )
    fig, ax = plt.subplots(figsize=(11, 6.3))
    im = ax.imshow(ratio, cmap="RdYlGn_r", vmin=0.65, vmax=1.15, aspect="auto")
    ax.set_xticks(range(6), STATS)
    ax.set_yticks(range(len(order)), order)
    for i in range(len(order)):
        for j in range(6):
            ax.text(
                j, i, f"{ratio.iloc[i, j]:.2f}", ha="center", va="center", fontsize=11
            )
    ax.set_title(
        f"2026 preseason accuracy | {len(primary)} shared batters with 200+ PA\nRMSE / naive 2025 RMSE; lower is better",
        fontsize=14,
        pad=18,
    )
    fig.colorbar(im, ax=ax, label="Relative error (1.00 = repeat 2025)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_heatmap.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, s in zip(axes.flat, STATS):
        yt, yp = truth.loc[primary, s], predictions["Hydra"].loc[primary, s]
        ax.scatter(yt, yp, alpha=0.55, s=16)
        low, high = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        ax.set(title=s, xlabel="2026 actual", ylabel="Saved Hydra forecast")
    fig.suptitle("Hydra calibration: same 200+ PA cohort")
    fig.tight_layout()
    fig.savefig(OUT / "hydra_calibration.png", dpi=150)
    plt.close(fig)
    residual = predictions["Hydra"].loc[primary, STATS] - truth.loc[primary, STATS]
    residual.corr().to_csv(OUT / "hydra_residual_correlations.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = residual.corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(6), STATS)
    ax.set_yticks(range(6), STATS)
    for i in range(6):
        for j in range(6):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(corr.iloc[i, j]) > 0.65 else "black",
            )
    ax.set_title("Hydra residual correlations | 198 shared batters")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / "hydra_residual_correlations.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, s in zip(axes.flat, STATS):
        from scipy.stats import gaussian_kde

        ax.hist(residual[s], bins=24, alpha=0.65, density=True)
        grid = np.linspace(residual[s].min(), residual[s].max(), 200)
        ax.plot(grid, gaussian_kde(residual[s])(grid), color="navy")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set(title=s, xlabel="Forecast minus actual", ylabel="Density")
    fig.tight_layout()
    fig.savefig(OUT / "hydra_residuals.png", dpi=150)
    plt.close(fig)
    largest = []
    for s in STATS:
        ids = residual[s].abs().nlargest(8).index
        for pid in ids:
            largest.append(
                {
                    "target": s,
                    "idfg": int(pid),
                    "Player": mapped.loc[pid, "Player"],
                    "prediction": predictions["Hydra"].loc[pid, s],
                    "actual": truth.loc[pid, s],
                    "error": residual.loc[pid, s],
                    "actual_pa": truth.loc[pid, "PA"],
                }
            )
    pd.DataFrame(largest).to_csv(OUT / "hydra_largest_errors.csv", index=False)

    manifest = {
        "year": cfg["year"],
        "analysis_date": "2026-10-29",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": digest(OUT / "evaluation.yaml"),
        "script_sha256": digest(Path(__file__)),
        "inputs": inputs,
        "hydra_players": len(mapped),
        "common_players": len(common),
        "primary_players": len(primary),
        "zero_actual_players": mapped.loc[missing, "Player"].tolist(),
        "excluded_from_common": mapped.loc[
            mapped.index.difference(common), "Player"
        ].tolist(),
        "seed": cfg["seed"],
        "bootstrap_samples": cfg["bootstrap_samples"],
        "verification_snapshots": {
            name: digest(OUT / name)
            for name in [
                "player_comparison.csv",
                "earlier_hydra.csv",
                "actuals_2026_fangraphs.csv",
                "actuals_2026_mlb.csv",
            ]
        },
        "sources": {
            "fangraphs": "https://www.fangraphs.com/leaders/major-league?pos=all&stats=bat&lg=all&qual=0&type=8&season=2026&season1=2026&ind=0",
            "mlb": "https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting&season=2026&sportIds=1&gameType=R&playerPool=ALL&limit=2000",
        },
    }
    for entry in inputs:
        assert digest(ROOT / entry["path"]) == entry["sha256"]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    report = [
        "# Saved 2026 preseason forecasts vs 2026 regular-season actuals",
        "",
        f"Evaluated October 29, 2026. Hydra: {len(mapped)} players; shared by all systems: {len(common)}; primary cohort (200+ actual PA): {len(primary)}.",
        "",
        "## Findings",
        "",
        "Hydra ranks first on the existing 200+ PA benchmark: mean RMSE 9.418 versus Steamer 9.478 and ZiPS 10.124. Its equal-stat score is 0.8336 versus Steamer 0.8365; this small difference is not established by the paired bootstrap (Steamer minus Hydra 95% interval: -0.0280 to +0.0316). Hydra has the lowest HR, R and RBI RMSE; THE BAT X leads OBP, ATC leads SLG, and Steamer leads SB. All nine systems beat naive persistence on the aggregate score.",
        "",
        "This is not an unconditional Hydra win. Including all 240 shared projected players in the counting-stat evaluation puts Steamer first, followed by ATC, THE BAT X, THE BAT, and Hydra fifth. Hydra's mean count RMSE is 16.486 versus Steamer 14.549 (13.3% higher). On all shared players, PA RMSE is 192.2 for Hydra versus 151.0 for Steamer; Hydra overpredicts PA by 77.1 per player on average. The per-PA diagnostic puts Hydra and ATC nearly level at the top. This points to availability/playing-time projection as a larger improvement opportunity than the underlying rate model; it does not by itself establish a causal error decomposition.",
        "",
        "The conclusion is sensitive to the evaluation population: Steamer wins among all 234 shared batters with at least one PA, while ATC leads among the 100 with 500+ PA. The earlier March 29 Hydra export scores 9.437 mean RMSE and 0.8352 equal-stat score, leaving its narrow 200+ PA lead over Steamer intact. These are evaluations of saved forecasts with a timing caveat, not an independently proven Opening-Day freeze.",
        "",
        "## Scope and provenance",
        "",
        "The primary Hydra forecast is the saved projections_mtl_2026.csv export, dated April 12. Public CSVs are dated March 12. A different March 29 Hydra export is evaluated separately. File modification dates are evidence of local snapshots, not proof of immutable pre-Opening-Day forecasts. The available merged model inputs stop at 2025, but model-development timing cannot be independently established from these artifacts. No model was retrained and no forecasts were refreshed.",
        "",
        "Hydra values retain their saved precision (3 decimals for OBP/SLG; integer counts); public exports retain original precision. IDs were recovered through exact name plus 2025 PA matches, then all comparisons use FanGraphs player IDs. Hydra projected PA is reconstructed using the existing temporal-feature and Marcel functions on the saved 2016–2025 merged frame. Its saved priors parquet contains different forecasts and is not substituted for the requested CSV.",
        "",
        "Actuals are FanGraphs season=2026, season1=2026, qual=0, month=0 regular-season totals, all teams combined. All 1,472 returned records were retained. Independent MLB gameType=R season totals contain 751 players; all 751 match FanGraphs on all four target counts exactly and on OBP/SLG within 0.0005 rounding. PA differs by one for Ryan Jeffers and Sean Keys; this report consistently uses FanGraphs PA. FanGraphs' browser page returned HTTP 403; its data API succeeded.",
        "",
        "No batting record in either complete source: "
        + ", ".join(manifest["zero_actual_players"])
        + ". Their season counts and PA are zero; undefined rates are never scored as zero.",
        "",
        "Excluded from the all-model intersection because a public system lacks a valid forecast: "
        + ", ".join(manifest["excluded_from_common"])
        + ". Pairwise coverage checks are provided separately.",
        "",
        "## Primary RMSE comparison",
        "",
        "Every model is scored on identical players with at least 200 actual PA, consistent with the existing benchmark threshold. Lower RMSE is better. Equal-stat score is the mean of six target RMSEs divided by the corresponding naive-persistence RMSE; 1.00 means repeat 2025. This gives all targets equal influence. The repository's mixed-unit Mean RMSE is also retained, but is dominated by runs/RBI and should not be the sole overall ranking.",
        "",
        table(rmse_table.reset_index()),
        "",
        "## Other required metrics on the same primary cohort",
        "",
        "MAE (lower is better):",
        "",
        table(
            met.query("cohort == 'primary_200pa'")
            .pivot(index="model", columns="target", values="mae")
            .loc[order, STATS]
            .reset_index()
        ),
        "",
        "R² (higher is better; negative means worse than the cohort mean):",
        "",
        table(
            met.query("cohort == 'primary_200pa'")
            .pivot(index="model", columns="target", values="r2")
            .loc[order, STATS]
            .reset_index()
        ),
        "",
        "MAPE, percent (zero actuals excluded; unstable for small count outcomes):",
        "",
        table(
            met.query("cohort == 'primary_200pa'")
            .pivot(index="model", columns="target", values="mape")
            .loc[order, STATS]
            .reset_index(),
            digits=2,
        ),
        "",
        "## Robustness and playing time",
        "",
        "The full-roster count evaluation includes low-PA and zero-PA players, avoiding the survivor filter in the primary table. The skill-only diagnostic divides each system's predicted counts by its own projected PA and actual counts by actual PA. Actual playing time is not substituted into season forecasts. Hydra's skill rates are approximate because the saved counts are rounded.",
        "",
        table(
            agg[~agg.cohort.str.startswith("pairwise")][
                [
                    "cohort",
                    "model",
                    "n",
                    "mean_relative_rmse",
                    "mean_rmse",
                    "mean_mae",
                    "mean_r2",
                ]
            ]
        ),
        "",
        "## Paired uncertainty versus Hydra",
        "",
        "2,000 paired player bootstrap samples per cohort, seed 42. Delta is other model score minus Hydra score, using that cohort's fixed naive-RMSE normalization. Positive means Hydra is better. An interval crossing zero does not establish a difference; intervals are exploratory and not adjusted for multiple comparisons.",
        "",
        table(pd.DataFrame(bootstrap)),
        "",
        "## Coverage",
        "",
        table(pd.DataFrame(cover)),
        "",
        "## Files and metric definitions",
        "",
        "metrics_by_target.csv contains RMSE, MAE, R², MAPE, signed bias, valid MAPE denominator counts, and normalized error for every target/cohort/model. MAPE excludes zero actuals, so count-stat MAPE, especially steals, is supplementary. summary_metrics.csv contains means. player_comparison.csv contains every input forecast and matched actual; playing_time_metrics.csv isolates PA errors. Accuracy, calibration and residual plots are included. Training curves are not regenerated because this is a frozen-forecast evaluation. manifest.json records every input hash, code version, config and seed. Reproduce from repository root with `.venv/bin/python data/reports/preseason_2026/evaluate.py`.",
    ]
    report += [
        "",
        "## Reproduction and charts",
        "",
        "See [verification instructions and source requirements](README.md). "
        "The offline verifier reproduces the numerical results from committed "
        "snapshots; rebuilding with evaluate.py requires the original local caches.",
        "",
        "![Accuracy against naive persistence](accuracy_heatmap.png)",
        "",
        "![Hydra calibration](hydra_calibration.png)",
        "",
        "![Hydra residual distributions](hydra_residuals.png)",
        "",
        "![Hydra residual correlations](hydra_residual_correlations.png)",
    ]
    (OUT / "evaluation_report.md").write_text("\n".join(report) + "\n")
    print(rmse_table.round(4).to_string())
    print(
        "\nCOHORTS\n",
        agg[~agg.cohort.str.startswith("pairwise")].round(4).to_string(index=False),
    )
    print("\nBOOTSTRAP\n", pd.DataFrame(bootstrap).round(4).to_string(index=False))
    print(
        "\nCOUNTS",
        json.dumps(
            {
                k: manifest[k]
                for k in ["hydra_players", "common_players", "primary_players"]
            }
        ),
    )


if __name__ == "__main__":
    main()
