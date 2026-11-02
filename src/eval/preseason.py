"""Score saved preseason forecasts against actual regular-season results.

Reusable across seasons: every year-specific value comes from the evaluation
config (``configs/preseason_eval_{year}.yaml``). Inputs are read, never
modified; all outputs go to the configured output directory.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

from src.data.fetch_projections import load_projections
from src.eval.metrics import compute_metrics
from src.eval.pa_projection import project_pa
from src.features.temporal import compute_temporal_features

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
STATS = ["OBP", "SLG", "HR", "R", "RBI", "SB"]
COUNTS = STATS[2:]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def naive_label(cfg: dict) -> str:
    return f"Naive {cfg['year'] - 1}"


def cohort_names(cfg: dict) -> dict[str, str]:
    m = cfg["minimum_actual_pa"]
    return {
        "primary": f"primary_{m}pa",
        "counts": "all_common_counts",
        "skill": f"skill_per_pa_{m}pa",
        "alternate": f"earlier_hydra_export_{m}pa",
        "pairwise": f"pairwise_{m}pa_",
    }


def _path(cfg: dict, key: str) -> Path:
    return ROOT / cfg[key]


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


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def load_hydra(cfg: dict) -> tuple[pd.DataFrame, pd.Series, str]:
    """Return the Hydra export indexed by idfg, projected PA, and the ID source.

    Exports from ``generate_projections.py`` carry ``idfg`` and projected PA.
    Older exports without them need ``legacy_id_match`` in the config.
    """
    year, prev = cfg["year"], cfg["year"] - 1
    hydra = pd.read_csv(_path(cfg, "hydra"))
    mlbam = None
    if "idfg" in hydra.columns:
        mapped = hydra.set_index("idfg")
        pa = mapped[f"{year} PA (proj)"].astype(float)
        mlbam = mapped["mlbam_id"]
        source = "export"
    else:
        legacy = cfg["legacy_id_match"]
        prev_actuals = pd.read_parquet(ROOT / legacy["previous_actuals"]).query(
            f"pa >= {legacy['previous_min_pa']}"
        )
        mapped = hydra.merge(
            prev_actuals[["name", "idfg", "pa"]],
            left_on=["Player", f"{prev} PA"],
            right_on=["name", "pa"],
            how="left",
            validate="one_to_one",
        )
        assert mapped.idfg.notna().all() and len(mapped) == len(hydra)
        mapped.idfg = mapped.idfg.astype(int)
        mapped = mapped.set_index("idfg")
        merged = pd.read_parquet(ROOT / legacy["merged"])
        assert merged.season.max() < year
        temporal = compute_temporal_features(merged, stats=["pa"])
        pa_frame = temporal[temporal.season.eq(prev)].set_index("idfg")
        pa = project_pa(pa_frame).reindex(mapped.index)
        mlbam = pa_frame.loc[mapped.index, "mlbam_id"]
        source = "name_and_prior_pa_match"
    assert mapped.index.is_unique and pa.notna().all()
    mapped["mlbam_id"] = mlbam.astype(int).to_numpy()
    return mapped, pa, source


def load_alternate(cfg: dict, mapped: pd.DataFrame, pa: pd.Series) -> pd.DataFrame:
    prefix = cfg.get("hydra_prefix", "MTL")
    alt = pd.read_csv(_path(cfg, "hydra_alternate"))
    if "idfg" in alt.columns:
        alt = alt.set_index("idfg")
        assert alt.index.isin(mapped.index).all()
    else:
        alt = alt.merge(
            mapped.reset_index()[["idfg", "Player"]],
            on="Player",
            how="left",
            validate="one_to_one",
        ).set_index("idfg")
    assert alt.index.notna().all()
    alt = alt[[f"{prefix} {s}" for s in STATS]].rename(
        columns={f"{prefix} {s}": s for s in STATS}
    )
    alt["PA"] = pa
    return alt


def load_public(cfg: dict) -> dict[str, pd.DataFrame]:
    """Load public systems through the project's projection loader."""
    year = cfg["year"]
    out = {}
    for system, label in cfg["systems"].items():
        # One system per call: concatenating systems would upcast int columns.
        df = load_projections(year, [system], ROOT / cfg["external_dir"])
        if df is None:
            raise FileNotFoundError(f"No saved {system} projections for {year}")
        df["idfg"] = pd.to_numeric(df.idfg, errors="coerce")
        df = df.dropna(subset=["idfg"])
        df.idfg = df.idfg.astype(int)
        assert not df.idfg.duplicated().any()
        cols = {s.lower(): s for s in STATS} | {"pa": "PA"}
        df = df.set_index("idfg")[list(cols)].rename(columns=cols)
        out[label] = df.apply(pd.to_numeric).dropna().query("PA > 0")
    return out


def load_fangraphs_actuals(path: Path, year: int) -> pd.DataFrame:
    raw = json.loads(path.read_text())
    actual = pd.DataFrame(raw["data"])
    assert len(actual) == raw["totalCount"] and actual.Season.eq(year).all()
    assert not actual.playerid.duplicated().any()
    actual["Player"] = actual.Name.map(
        lambda s: html.unescape(re.sub("<[^>]+>", "", s))
    )
    actual = actual.set_index("playerid")
    actual.index = actual.index.astype(int)
    return actual


MLB_RENAME = {
    "obp": "OBP",
    "slg": "SLG",
    "homeRuns": "HR",
    "runs": "R",
    "rbi": "RBI",
    "stolenBases": "SB",
    "plateAppearances": "PA",
}


def load_mlb_actuals(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text())["stats"][0]
    assert len(raw["splits"]) == raw["totalSplits"]
    official = pd.DataFrame(
        [
            {
                "mlbam_id": s["player"]["id"],
                "Player": s["player"]["fullName"],
                **s["stat"],
            }
            for s in raw["splits"]
        ]
    )
    assert not official.mlbam_id.duplicated().any()
    return official


def crosscheck(actual: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    """Compare FanGraphs and MLB season totals; fail on any target mismatch."""
    check = actual.merge(
        official,
        left_on="xMLBAMID",
        right_on="mlbam_id",
        validate="one_to_one",
        suffixes=("_fg", "_mlb"),
    )
    checks = []
    for fg in ["PA", "HR", "R", "RBI", "SB", "OBP", "SLG"]:
        mlb = {v: k for k, v in MLB_RENAME.items()}[fg]
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
    return pd.DataFrame(checks)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def run(cfg: dict, out: Path) -> dict:
    """Score every configured forecast and write tables, plots and manifest."""
    out.mkdir(parents=True, exist_ok=True)
    year, naive = cfg["year"], naive_label(cfg)
    names = cohort_names(cfg)
    min_pa = cfg["minimum_actual_pa"]
    has_alt = bool(cfg.get("hydra_alternate"))
    has_check = bool(cfg.get("crosscheck"))

    source_paths = [_path(cfg, "hydra")]
    if has_alt:
        source_paths.append(_path(cfg, "hydra_alternate"))
    if cfg.get("legacy_id_match"):
        legacy = cfg["legacy_id_match"]
        source_paths += [ROOT / legacy["merged"], ROOT / legacy["previous_actuals"]]
    source_paths.append(_path(cfg, "actuals"))
    if has_check:
        source_paths.append(_path(cfg, "crosscheck"))
    source_paths += [
        ROOT / cfg["external_dir"] / f"{s}_{year}.csv" for s in cfg["systems"]
    ]
    inputs = [
        {
            "path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
            "sha256": digest(p),
            "modified_utc": datetime.fromtimestamp(
                p.stat().st_mtime, timezone.utc
            ).isoformat(),
        }
        for p in source_paths
    ]

    mapped, pa, id_source = load_hydra(cfg)
    prefix = cfg.get("hydra_prefix", "MTL")
    predictions = {}
    for label, col_prefix in [("Hydra", prefix), (naive, str(year - 1))]:
        df = mapped[[f"{col_prefix} {s}" for s in STATS]].copy()
        df.columns = STATS
        df["PA"] = pa if label == "Hydra" else mapped[f"{year - 1} PA"]
        predictions[label] = df
    if has_alt:
        earlier = load_alternate(cfg, mapped, pa)
        earlier.to_csv(out / "earlier_hydra.csv", index_label="idfg")
    predictions |= load_public(cfg)

    actual = load_fangraphs_actuals(_path(cfg, "actuals"), year)
    missing = mapped.index.difference(actual.index)
    if has_check:
        official = load_mlb_actuals(_path(cfg, "crosscheck"))
        official.rename(columns=MLB_RENAME)[
            ["mlbam_id", "Player", *STATS, "PA"]
        ].to_csv(out / f"actuals_{year}_mlb.csv", index=False)
        crosscheck(actual, official).to_csv(out / "source_crosscheck.csv", index=False)
        # A Hydra player absent from FanGraphs must also be absent from MLB.
        assert not mapped.loc[missing, "mlbam_id"].isin(official.mlbam_id).any()

    truth = actual.reindex(mapped.index)[STATS + ["PA"]].copy()
    truth.loc[missing, COUNTS + ["PA"]] = 0
    truth.insert(0, "Player", mapped.Player)
    truth.to_csv(out / "actuals_hydra_players.csv", index_label="idfg")
    actual[["Player", "xMLBAMID"] + STATS + ["PA"]].to_csv(
        out / f"actuals_{year}_fangraphs.csv", index_label="idfg"
    )
    common = mapped.index
    for pred in predictions.values():
        common = common.intersection(pred.index)
    common = common.sort_values()
    primary = common[truth.loc[common, "PA"].ge(min_pa)]
    cover = []
    for label, pred in predictions.items():
        available = mapped.index.intersection(pred.index)
        cover.append(
            {
                "Model": label,
                "Saved players": len(pred),
                "Hydra overlap": len(available),
                f"Overlap >={min_pa} PA": int(
                    truth.loc[available, "PA"].ge(min_pa).sum()
                ),
                "Missing Hydra players": "; ".join(
                    mapped.loc[mapped.index.difference(pred.index), "Player"]
                ),
            }
        )
    pd.DataFrame(cover).to_csv(out / "coverage.csv", index=False)
    matched = truth.copy()
    matched["mlbam_id"] = mapped.loc[matched.index, "mlbam_id"].astype(int)
    matched["common_all_models"] = matched.index.isin(common)
    matched[f"primary_{min_pa}pa"] = matched.index.isin(primary)
    for label, pred in predictions.items():
        matched = matched.join(pred.add_prefix(label + " "))
    matched.to_csv(out / "player_comparison.csv", index_label="idfg")

    metrics, aggregate, bootstrap = [], [], []
    cohorts = [
        (names["primary"], primary, STATS, False),
        (names["counts"], common, COUNTS, False),
        (names["skill"], primary, STATS, True),
    ]
    for threshold in cfg["sensitivity_minimum_pa"]:
        ids = common[truth.loc[common, "PA"].ge(threshold)]
        cohorts.append((f"sensitivity_{threshold}pa", ids, STATS, False))
    bootstrapped = {names["primary"], names["counts"], names["skill"]}

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
        scale = np.sqrt(np.mean((yt - pred_arrays[naive]) ** 2, axis=0))
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
        if cohort in bootstrapped:
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
    if has_alt:
        evaluate(
            names["alternate"], primary, STATS, False, {**predictions, "Hydra": earlier}
        )
    for label, pred in predictions.items():
        ids = mapped.index.intersection(pred.index)
        ids = ids[truth.loc[ids, "PA"].ge(min_pa)]
        evaluate(
            names["pairwise"] + label,
            ids,
            STATS,
            False,
            {"Hydra": predictions["Hydra"], naive: predictions[naive], label: pred},
        )
    met = pd.DataFrame(metrics)
    agg = pd.DataFrame(aggregate)
    met.to_csv(out / "metrics_by_target.csv", index=False)
    agg.to_csv(out / "summary_metrics.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(out / "paired_bootstrap.csv", index=False)
    pa_metrics = []
    for cohort, ids in [(names["primary"], primary), ("all_common", common)]:
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
    pd.DataFrame(pa_metrics).to_csv(out / "playing_time_metrics.csv", index=False)

    primary_agg = agg.query(f"cohort == '{names['primary']}'")
    order = primary_agg.sort_values("mean_relative_rmse").model.tolist()
    primary_met = met.query(f"cohort == '{names['primary']}'")
    rmse_table = primary_met.pivot(index="model", columns="target", values="rmse").loc[
        order, STATS
    ]
    score = primary_agg.set_index("model").loc[order]
    rmse_table["Equal-stat score"] = score.mean_relative_rmse
    rmse_table["Mean RMSE"] = score.mean_rmse
    rmse_table.to_csv(out / "rmse_leaderboard.csv")
    ratio = primary_met.pivot(
        index="model", columns="target", values="rmse_relative_to_naive"
    ).loc[order, STATS]
    residual = predictions["Hydra"].loc[primary, STATS] - truth.loc[primary, STATS]
    _plots(out, cfg, order, ratio, primary, truth, predictions["Hydra"], residual)
    largest = []
    for s in STATS:
        for pid in residual[s].abs().nlargest(8).index:
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
    pd.DataFrame(largest).to_csv(out / "hydra_largest_errors.csv", index=False)

    snapshots = ["player_comparison.csv", f"actuals_{year}_fangraphs.csv"]
    snapshots += ["earlier_hydra.csv"] if has_alt else []
    snapshots += [f"actuals_{year}_mlb.csv"] if has_check else []
    manifest = {
        "year": year,
        "analysis_date": datetime.now(timezone.utc).date().isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": digest(Path(cfg["_config_path"])),
        "script_sha256": digest(Path(__file__)),
        "hydra_id_source": id_source,
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
        "verification_snapshots": {name: digest(out / name) for name in snapshots},
        "sources": cfg.get("sources", {}),
    }
    # Inputs must not change while the evaluation runs.
    for path, entry in zip(source_paths, inputs):
        assert digest(path) == entry["sha256"]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "evaluation_report.md").write_text(render_report(cfg, out))
    return manifest


def _plots(
    out: Path,
    cfg: dict,
    order: list[str],
    ratio: pd.DataFrame,
    primary: pd.Index,
    truth: pd.DataFrame,
    hydra: pd.DataFrame,
    residual: pd.DataFrame,
) -> None:
    year, min_pa = cfg["year"], cfg["minimum_actual_pa"]
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
        f"{year} preseason accuracy | {len(primary)} shared batters with {min_pa}+ PA"
        f"\nRMSE / naive {year - 1} RMSE; lower is better",
        fontsize=14,
        pad=18,
    )
    fig.colorbar(
        im, ax=ax, label=f"Relative error (1.00 = repeat {year - 1})", shrink=0.8
    )
    fig.tight_layout()
    fig.savefig(out / "accuracy_heatmap.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, s in zip(axes.flat, STATS):
        yt, yp = truth.loc[primary, s], hydra.loc[primary, s]
        ax.scatter(yt, yp, alpha=0.55, s=16)
        low, high = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        ax.set(title=s, xlabel=f"{year} actual", ylabel="Saved Hydra forecast")
    fig.suptitle(f"Hydra calibration: same {min_pa}+ PA cohort")
    fig.tight_layout()
    fig.savefig(out / "hydra_calibration.png", dpi=150)
    plt.close(fig)

    corr = residual.corr()
    corr.to_csv(out / "hydra_residual_correlations.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
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
    ax.set_title(f"Hydra residual correlations | {len(primary)} shared batters")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "hydra_residual_correlations.png", dpi=150)
    plt.close(fig)

    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, s in zip(axes.flat, STATS):
        ax.hist(residual[s], bins=24, alpha=0.65, density=True)
        grid = np.linspace(residual[s].min(), residual[s].max(), 200)
        ax.plot(grid, gaussian_kde(residual[s])(grid), color="navy")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set(title=s, xlabel="Forecast minus actual", ylabel="Density")
    fig.tight_layout()
    fig.savefig(out / "hydra_residuals.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _join(items: list[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _rank(df: pd.DataFrame, column: str, model: str) -> int:
    return df.sort_values(column).model.tolist().index(model) + 1


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10)
    return f"{n}{suffix or 'th'}"


def render_report(cfg: dict, out: Path) -> str:
    """Build the Markdown report only from the result files in ``out``.

    Every number and ranking in the text comes from those files, so the report
    stays correct when forecasts, systems, thresholds or the season change.
    """
    year, min_pa, naive = cfg["year"], cfg["minimum_actual_pa"], naive_label(cfg)
    names = cohort_names(cfg)
    manifest = json.loads((out / "manifest.json").read_text())
    met = pd.read_csv(out / "metrics_by_target.csv")
    agg = pd.read_csv(out / "summary_metrics.csv")
    boot = pd.read_csv(out / "paired_bootstrap.csv")
    cover = pd.read_csv(out / "coverage.csv").fillna("")
    pa_met = pd.read_csv(out / "playing_time_metrics.csv")
    leaderboard = pd.read_csv(out / "rmse_leaderboard.csv", index_col=0)
    order = leaderboard.index.tolist()

    def cohort(name: str) -> pd.DataFrame:
        return agg[agg.cohort.eq(name)]

    primary = cohort(names["primary"])
    n_primary = int(primary.n.iloc[0])
    systems = primary[~primary.model.eq(naive)]
    hydra = primary.set_index("model").loc["Hydra"]
    by_rmse = primary.sort_values("mean_rmse")
    by_score = primary.sort_values("mean_relative_rmse")
    rmse_top = [
        f"{m} {v:.3f}" for m, v in zip(by_rmse.model[:3], by_rmse.mean_rmse[:3])
    ]
    findings = [
        f"On the primary benchmark ({n_primary} shared batters with {min_pa}+ actual"
        f" PA), Hydra ranks {_ordinal(_rank(primary, 'mean_rmse', 'Hydra'))} of"
        f" {len(primary)} by mean RMSE (top three: {_join(rmse_top)}) and"
        f" {_ordinal(_rank(primary, 'mean_relative_rmse', 'Hydra'))} by equal-stat"
        f" score ({hydra.mean_relative_rmse:.4f})."
    ]
    rival = by_score[~by_score.model.isin(["Hydra", naive])].iloc[0]
    ci = boot[boot.cohort.eq(names["primary"]) & boot.model.eq(rival.model)].iloc[0]
    settled = ci.ci95_low > 0 or ci.ci95_high < 0
    findings[0] += (
        f" The best other system by equal-stat score is {rival.model}"
        f" ({rival.mean_relative_rmse:.4f}); the paired bootstrap"
        f" {'supports' if settled else 'does not establish'} a difference"
        f" ({rival.model} minus Hydra 95% interval: {ci.ci95_low:+.4f} to"
        f" {ci.ci95_high:+.4f})."
    )
    rmse_by_target = met[met.cohort.eq(names["primary"])].pivot(
        index="model", columns="target", values="rmse"
    )
    leaders: dict[str, list[str]] = {}
    for s in STATS:
        leaders.setdefault(rmse_by_target[s].idxmin(), []).append(s)
    findings[0] += (
        " Lowest RMSE by target: "
        + "; ".join(f"{m} leads {_join(s)}" for m, s in leaders.items())
        + "."
    )
    beat = int(systems.mean_relative_rmse.lt(1).sum())
    findings[0] += (
        f" {beat} of {len(systems)} systems beat naive persistence on the"
        " equal-stat score."
    )

    counts = cohort(names["counts"]).sort_values("mean_rmse")
    count_leader = counts.iloc[0]
    hydra_counts = counts.set_index("model").loc["Hydra"]
    pa_all = pa_met[pa_met.cohort.eq("all_common")].set_index("model")
    skill = cohort(names["skill"])
    gap = hydra_counts.mean_rmse / count_leader.mean_rmse - 1
    findings.append(
        f"Including all {int(counts.n.iloc[0])} shared projected players in the"
        f" counting-stat evaluation puts {count_leader.model} first and Hydra"
        f" {_ordinal(_rank(counts, 'mean_rmse', 'Hydra'))} (mean count RMSE"
        f" {hydra_counts.mean_rmse:.3f} versus {count_leader.mean_rmse:.3f},"
        f" {gap:+.1%}). On all shared players, PA RMSE is"
        f" {pa_all.loc['Hydra', 'rmse']:.1f} for Hydra versus"
        f" {pa_all.loc[count_leader.model, 'rmse']:.1f} for {count_leader.model};"
        f" Hydra's mean PA bias is {pa_all.loc['Hydra', 'bias_pred_minus_actual']:+.1f}"
        f" per player. On the per-PA skill diagnostic Hydra ranks"
        f" {_ordinal(_rank(skill, 'mean_relative_rmse', 'Hydra'))} by equal-stat"
        " score. A gap between the season-total and per-PA ranks points at"
        " playing-time projection, but does not by itself establish a causal"
        " error decomposition."
    )

    sensitivity = []
    for threshold in cfg["sensitivity_minimum_pa"]:
        c = cohort(f"sensitivity_{threshold}pa").sort_values("mean_relative_rmse")
        sensitivity.append(
            f"{c.model.iloc[0]} leads the {int(c.n.iloc[0])} shared batters with"
            f" {threshold}+ PA"
        )
    third = (
        "Sensitivity to the evaluation population (equal-stat score): "
        + "; ".join(sensitivity)
        + "."
    )
    alt = cohort(names["alternate"])
    if len(alt):
        alt_hydra = alt.set_index("model").loc["Hydra"]
        third += (
            f" The alternate Hydra export scores {alt_hydra.mean_rmse:.3f} mean RMSE"
            f" and {alt_hydra.mean_relative_rmse:.4f} equal-stat score, ranking"
            f" {_ordinal(_rank(alt, 'mean_relative_rmse', 'Hydra'))} on the"
            " primary cohort."
        )
    findings.append(third)

    dated = [
        f"{Path(e['path']).name} ({e['modified_utc'][:10]})" for e in manifest["inputs"]
    ]
    id_source = manifest.get("hydra_id_source")
    if id_source is None:  # manifests written before this key existed
        id_source = "legacy" if cfg.get("legacy_id_match") else "export"
    id_note = (
        "Player IDs and projected PA come from the Hydra export."
        if id_source == "export"
        else f"The Hydra export has no player IDs, so IDs were recovered through"
        f" exact name plus {year - 1} PA matches. Projected PA was reconstructed"
        " with the Marcel formula on the saved merged frame."
    )
    provenance = [
        f"Input files and their local modification dates: {_join(dated)}."
        " Modification dates are evidence of local snapshots, not proof of"
        " immutable pre-Opening-Day forecasts. No model was retrained and no"
        " forecasts were refreshed.",
        id_note + " All comparisons use FanGraphs player IDs.",
    ]
    fg = pd.read_csv(out / f"actuals_{year}_fangraphs.csv")
    actual_note = (
        f"Actuals are FanGraphs {year} regular-season totals, all teams combined"
        f" ({len(fg):,} records retained)."
    )
    check_path = out / "source_crosscheck.csv"
    if check_path.exists():
        check = pd.read_csv(check_path).set_index("stat")
        mlb = pd.read_csv(out / f"actuals_{year}_mlb.csv")
        joined = fg.merge(mlb, left_on="xMLBAMID", right_on="mlbam_id")
        pa_diff = joined[joined.PA_x.ne(joined.PA_y)].Player_x.tolist()
        actual_note += (
            f" Independent MLB season totals contain {len(mlb)} players; all"
            f" {int(check.loc['HR', 'n'])} match FanGraphs on the four target counts"
            " exactly and on OBP/SLG within 0.0005 rounding."
        )
        if pa_diff:
            actual_note += (
                f" PA differs for {_join(pa_diff)}; this report uses FanGraphs PA."
            )
    provenance.append(actual_note)

    def pivot(value: str, digits: int = 4) -> str:
        return table(
            met[met.cohort.eq(names["primary"])]
            .pivot(index="model", columns="target", values=value)
            .loc[order, STATS]
            .reset_index(),
            digits=digits,
        )

    summary_cols = [
        "cohort",
        "model",
        "n",
        "mean_relative_rmse",
        "mean_rmse",
        "mean_mae",
        "mean_r2",
    ]
    report = [
        f"# Saved {year} preseason forecasts vs {year} regular-season actuals",
        "",
        f"Evaluated {manifest['analysis_date']}. Hydra: {manifest['hydra_players']}"
        f" players; shared by all systems: {manifest['common_players']}; primary"
        f" cohort ({min_pa}+ actual PA): {manifest['primary_players']}.",
        "",
        "## Findings",
        "",
        *[line for f in findings for line in (f, "")],
        "## Scope and provenance",
        "",
        *[line for p in provenance for line in (p, "")],
        "No batting record in the actuals: "
        + (", ".join(manifest["zero_actual_players"]) or "none")
        + ". Their season counts and PA are zero; undefined rates are never scored"
        " as zero.",
        "",
        "Excluded from the all-model intersection because a public system lacks a"
        " valid forecast: "
        + (", ".join(manifest["excluded_from_common"]) or "none")
        + ". Pairwise coverage checks are provided separately.",
        "",
        "## Primary RMSE comparison",
        "",
        f"Every model is scored on identical players with at least {min_pa} actual"
        " PA. Lower RMSE is better. Equal-stat score is the mean of six target RMSEs"
        f" divided by the corresponding naive-persistence RMSE; 1.00 means repeat"
        f" {year - 1}. This gives all targets equal influence. The repository's"
        " mixed-unit Mean RMSE is also retained, but is dominated by runs/RBI and"
        " should not be the sole overall ranking.",
        "",
        table(leaderboard.reset_index()),
        "",
        "## Other required metrics on the same primary cohort",
        "",
        "MAE (lower is better):",
        "",
        pivot("mae"),
        "",
        "R² (higher is better; negative means worse than the cohort mean):",
        "",
        pivot("r2"),
        "",
        "MAPE, percent (zero actuals excluded; unstable for small count outcomes):",
        "",
        pivot("mape", digits=2),
        "",
        "## Robustness and playing time",
        "",
        "The full-roster count evaluation includes low-PA and zero-PA players,"
        " avoiding the survivor filter in the primary table. The skill-only"
        " diagnostic divides each system's predicted counts by its own projected PA"
        " and actual counts by actual PA. Actual playing time is not substituted"
        " into season forecasts. Hydra's skill rates are approximate because the"
        " saved counts are rounded.",
        "",
        table(agg[~agg.cohort.str.startswith("pairwise")][summary_cols]),
        "",
        "## Paired uncertainty versus Hydra",
        "",
        f"{manifest['bootstrap_samples']:,} paired player bootstrap samples per"
        f" cohort, seed {manifest['seed']}. Delta is other model score minus Hydra"
        " score, using that cohort's fixed naive-RMSE normalization. Positive means"
        " Hydra is better. An interval crossing zero does not establish a"
        " difference; intervals are exploratory and not adjusted for multiple"
        " comparisons.",
        "",
        table(boot),
        "",
        "## Coverage",
        "",
        table(cover),
        "",
    ]
    if cfg.get("notes"):
        report += ["## Analyst notes", ""]
        report += [line for n in cfg["notes"] for line in (n.strip(), "")]
    report += [
        "## Files and metric definitions",
        "",
        "metrics_by_target.csv contains RMSE, MAE, R², MAPE, signed bias, valid MAPE"
        " denominator counts, and normalized error for every target/cohort/model."
        " MAPE excludes zero actuals, so count-stat MAPE, especially steals, is"
        " supplementary. summary_metrics.csv contains means. player_comparison.csv"
        " contains every input forecast and matched actual; playing_time_metrics.csv"
        " isolates PA errors. manifest.json records every input hash, code version,"
        " config and seed.",
        "",
        "## Reproduction and charts",
        "",
        "See [verification instructions and source requirements](README.md).",
        "",
        "![Accuracy against naive persistence](accuracy_heatmap.png)",
        "",
        "![Hydra calibration](hydra_calibration.png)",
        "",
        "![Hydra residual distributions](hydra_residuals.png)",
        "",
        "![Hydra residual correlations](hydra_residual_correlations.png)",
    ]
    return "\n".join(report) + "\n"
