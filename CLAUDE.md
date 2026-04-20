# Baseball Hydra: MLB Batter Season Stat Predictions

This folder implements a **multi-task learning (MTL) neural network that predicts next-season counting and rate stats for MLB batters** using historical performance, Statcast quality metrics, biometric data, ballpark factors, and team context.

**LLMs must follow this document as the authoritative build plan and working agreement.**
For setup, CLI usage, data format, and training commands, see [README.md](README.md).

---

## 1) Project Goal

Given a batter's historical profile (stats, Statcast metrics, biometric data, park, team), predict their **next-season stat line** across six target variables.

### Target Variables

| Index | Target | Internal Type | Description         |
| ----- | ------ | ------------- | ------------------- |
| 0     | `OBP`  | Rate          | On-base percentage  |
| 1     | `SLG`  | Rate          | Slugging percentage |
| 2     | `HR`   | Per-PA rate   | Home runs / PA      |
| 3     | `R`    | Per-PA rate   | Runs / PA           |
| 4     | `RBI`  | Per-PA rate   | RBI / PA            |
| 5     | `SB`   | Per-PA rate   | Stolen bases / PA   |

All targets are predicted as per-PA rates internally. At inference, count stat rates are multiplied by projected PA (Marcel formula: `PA_proj = 0.5 × PA_y1 + 0.1 × PA_y2 + 200`) to produce season counting stats. This rate-first approach aligns with how all major projection systems (ZiPS, Steamer, PECOTA, Marcel) decompose predictions and naturally handles the 2020 shortened season (60 games) since per-PA rates are season-length invariant.

---

## 2) Repository Principles

1. **No temporal leakage** — splits are strictly chronological. Features from year Y predict targets from year Y+1. Never use future data.
2. **Simplicity first** — add complexity only when metrics justify it.
3. **Reproducibility** — every run must be reproducible from a single config file (seed + data hash + code version).
4. **Config-driven** — all hyperparameters live in YAML configs, not in code.
5. **Multi-target evaluation** — always report per-target metrics. Aggregate metrics (mean RMSE) are secondary.
6. **Domain awareness** — respect baseball domain knowledge (aging curves, park factors, stabilisation rates, rate vs count distinction).

---

## 3) Data Contract

### 3.1 Data Sources

| Source                      | API / Method                                             | Features                                                                                                                                                                                          | Years          |
| --------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| **FanGraphs batting stats** | `pybaseball.batting_stats()`                             | PA, AB, H, 2B, 3B, HR, R, RBI, SB, CS, BB, SO, HBP, AVG, OBP, SLG, OPS, wOBA, wRC+, WAR, Age                                                                                                      | 2016-2025      |
| **Statcast raw BBE**        | `pybaseball.statcast()` → `statcast_raw_{year}.parquet`  | Raw pitch-level batted-ball events (LA, EV, spray coords, xwOBA, xBA, xSLG)                                                                                                                       | 2016-2025      |
| **Statcast aggregated**     | Local raw → `statcast_agg_{year}.parquet` (API fallback) | avg_exit_velocity, ev_p95, max_exit_velocity, avg_launch_angle, barrel_rate, hard_hit_rate, sweet_spot_rate                                                                                       | 2016-2025      |
| **Sprint speed**            | Baseball Savant sprint speed leaderboard / `pybaseball`  | sprint_speed (ft/s)                                                                                                                                                                               | 2016-2025      |
| **Bat speed**               | Baseball Savant bat speed leaderboard                    | avg_bat_speed, avg_swing_speed                                                                                                                                                                    | 2024-2025 only |
| **Ballpark factors**        | FanGraphs park factors                                   | park_factor (runs, HR, etc.)                                                                                                                                                                      | 2016-2025      |
| **Team stats**              | `pybaseball.team_batting()` or FanGraphs                 | team_runs, team_ops, team_sb                                                                                                                                                                      | 2016-2025      |
| **Public projections**      | FanGraphs JSON API (`/api/projections`) + manual CSV     | OBP, SLG, HR, R, RBI, SB (Steamer, ZiPS, The Bat, The Bat X) — comparison only, not in training pipeline. **API only serves current-year projections; historical projections are not available.** | 2022+          |

### 3.2 Key Constraints

- **Minimum PA threshold**: 200 PA per season for inclusion in training data. For 2026 predictions, require 300+ PA in 2025.
- **Bat speed is optional**: available only for 2024-2025. Must be handled as a feature with missingness (imputation or mask). Models must train and predict correctly with or without bat speed.
- **ID mapping**: FanGraphs uses `IDfg`, Statcast uses `mlbam_id`. Use `pybaseball.playerid_reverse_lookup()` to bridge them.
- **Season boundaries**: regular season only (`game_type == 'R'`). No spring training, postseason, or All-Star data.
- **Target alignment**: features from year Y predict targets from year Y+1. The feature row for (player, 2024) has target values from (player, 2025).
- **2020 shortened season (60 games)**: handled via rate-based targets — count targets (HR, R, RBI, SB) are stored as per-PA rates, making them comparable across seasons regardless of length. Raw counting stat features (`hr`, `sb`, `cs`) and their temporal derivatives are excluded from the model (via `exclude_features` in `data.yaml`); their per-PA rate equivalents are used instead. PA projection uses `season_games: {2020: 60}` to scale 2020 PA to 162-game pace for the Marcel formula.

### 3.3 Output Schema

The merged dataset for modeling has this structure:

```
player_id | season | age | [batting_features] | [statcast_features] | [speed_features] | [bat_speed_features] | [park_factors] | [team_features] | target_OBP | target_SLG | target_HR | target_R | target_RBI | target_SB
```

Each row = one player-season. Targets are next-season values.

---

## 4) Feature Engineering

### 4.1 Batting Features (from FanGraphs)

| Feature           | Source Column | Notes                                                  |
| ----------------- | ------------- | ------------------------------------------------------ |
| `pa`              | PA            | Plate appearances (also used for weighting)            |
| `bb_rate`         | BB/PA         | Walk rate                                              |
| `k_rate`          | SO/PA         | Strikeout rate                                         |
| `iso`             | SLG - AVG     | Isolated power                                         |
| `babip`           | BABIP         | Batting avg on balls in play                           |
| `avg`             | AVG           | Batting average                                        |
| `obp`             | OBP           | On-base percentage (also a feature for other targets)  |
| `slg`             | SLG           | Slugging (also a feature for other targets)            |
| `hr`              | HR            | Home runs (raw)                                        |
| `sb`              | SB            | Stolen bases (raw)                                     |
| `cs`              | CS            | Caught stealing                                        |
| `sb_rate`         | SB/(SB+CS)    | Stolen base success rate                               |
| `woba`            | wOBA          | Weighted on-base average                               |
| `wrc_plus`        | wRC+          | Weighted runs created plus (park-adjusted)             |
| `hbp_rate`        | HBP/PA        | Hit-by-pitch rate                                      |
| `contact_rate`    | 1-K%-BB%-HBP% | Contact rate (fraction of PA producing ball in play)   |
| `hr_per_pa`       | HR/PA         | Home runs per plate appearance (rate decomposition)    |
| `r_per_pa`        | R/PA          | Runs per plate appearance (rate decomposition)         |
| `rbi_per_pa`      | RBI/PA        | RBI per plate appearance (rate decomposition)          |
| `sb_per_pa`       | SB/PA         | Stolen bases per plate appearance (rate decomposition) |
| `sb_attempt_rate` | (SB+CS)/PA    | Stolen base attempt rate (willingness to run)          |
| `ibb_rate`        | IBB/PA        | Intentional walk rate                                  |
| `ubb_rate`        | (BB-IBB)/PA   | Unintentional walk rate                                |
| `singles_rate`    | 1B/PA         | Singles rate                                           |
| `doubles_rate`    | 2B/PA         | Doubles rate                                           |
| `triples_rate`    | 3B/PA         | Triples rate                                           |
| `extra_base_rate` | (2B+3B+HR)/PA | Extra-base hit rate                                    |
| `cs_rate`         | CS/PA         | Caught stealing rate                                   |

### 4.2 Statcast Quality Features

| Feature                                | Description                                                |
| -------------------------------------- | ---------------------------------------------------------- |
| `bbe_count`                            | Batted-ball events count                                   |
| `avg_exit_velocity`                    | Average exit velocity on batted balls (mph)                |
| `ev_p95`                               | 95th percentile exit velocity (mph)                        |
| `max_exit_velocity`                    | Absolute maximum exit velocity (mph)                       |
| `avg_launch_angle`                     | Average launch angle (degrees)                             |
| `barrel_rate`                          | Barrel rate (% of batted ball events)                      |
| `hard_hit_rate`                        | Hard hit rate (EV >= 95 mph)                               |
| `sweet_spot_rate`                      | Sweet spot rate (LA 8-32 degrees)                          |
| `estimated_woba_using_speedangle`      | Expected wOBA from Statcast speed-angle model              |
| `estimated_ba_using_speedangle`        | Expected batting average from speed-angle model            |
| `estimated_slg_using_speedangle`       | Expected slugging from speed-angle model                   |
| `has_*` (11 indicators)               | Missingness indicators for each Statcast metric above      |

### 4.2b Non-Contact Features (stabilisation-regressed rates)

Apply regression-to-the-mean using stabilisation points from FanGraphs research. These rates are more predictive than raw observed rates, especially for low-PA batters.

| Feature                | Description                               | Stabilisation PA |
| ---------------------- | ----------------------------------------- | ---------------- |
| `regressed_k_rate`     | Stabilisation-regressed strikeout rate    | 60               |
| `regressed_bb_rate`    | Stabilisation-regressed walk rate         | 120              |
| `regressed_hbp_rate`   | Stabilisation-regressed hit-by-pitch rate | 300              |
| `regressed_babip`      | Stabilisation-regressed BABIP             | 1200             |
| `regressed_iso`        | Stabilisation-regressed isolated power    | 160              |
| `regressed_hr_per_bbe` | Stabilisation-regressed HR per BBE        | 60 BBE           |

### 4.3 Biometric / Speed Features

| Feature           | Description                                                              | Availability   |
| ----------------- | ------------------------------------------------------------------------ | -------------- |
| `sprint_speed`    | Sprint speed (ft/s) from Baseball Savant                                 | 2016-2025      |
| `avg_bat_speed`   | Average bat speed (mph)                                                  | 2024-2025 only |
| `avg_swing_speed` | Average swing speed (mph)                                                | 2024-2025 only |
| `squared_up_rate` | Squared-up contact rate                                                  | 2024-2025 only |
| `blast_rate`      | Blast rate                                                               | 2024-2025 only |
| `fast_swing_rate` | Fast swing rate                                                          | 2024-2025 only |
| `bat_tracking_*`  | 5 tracking count features (swings, BBE, blasts, squared_up, fast_swings) | 2024-2025 only |
| `has_*`           | 10 missingness indicators for bat speed/tracking features                | All years      |

### 4.4 Context Features

| Feature              | Description                                                                                            |
| -------------------- | ------------------------------------------------------------------------------------------------------ |
| `age`                | Player age at season midpoint                                                                          |
| `age_squared`        | age^2 for non-linear aging curve                                                                       |
| `age_delta_speed`    | Piecewise aging delta for speed stats (peak ~23, steep decline). Excluded by default — see note below. |
| `age_delta_power`    | Piecewise aging delta for power stats (peak ~27, plateau to 30). Excluded by default.                  |
| `age_delta_patience` | Piecewise aging delta for plate discipline (peak ~30, slow decline). Excluded by default.              |

> **Note on aging delta features**: These are computed in the pipeline but excluded via `configs/data.yaml` `exclude_features`. Ablation testing showed they hurt benchmark RMSE (+1.04%) — the piecewise curves overfit to assumed aging patterns on the small training set (~1,800 rows). The generic `age + age_squared` features remain active and are sufficient.

| Feature                 | Description                                        |
| ----------------------- | -------------------------------------------------- |
| `park_factor_runs`      | Ballpark factor for runs                           |
| `park_factor_hr`        | Ballpark factor for home runs                      |
| `team_runs_per_game`    | Team's runs scored per game (lineup context)       |
| `team_ops`              | Team OPS (lineup protection proxy)                 |
| `team_sb`               | Team stolen bases (running game context)           |
| `team_sb_per_game`      | Team stolen bases per game                         |
| `sb_rule_era`           | Binary indicator (1 if season >= 2023 rule change) |
| `sb_era_x_speed`        | SB rule era × sprint speed interaction             |
| `sb_era_x_attempt_rate` | SB rule era × SB attempt rate interaction          |
| `speed_age_interaction` | Sprint speed × (age - 27) for speed-aging decline  |

### 4.5 Derived / Temporal Features

| Feature               | Description                                           |
| --------------------- | ----------------------------------------------------- |
| `prev_year_{stat}`    | Prior year's value for each target stat               |
| `weighted_avg_{stat}` | PA-weighted average of Y-1, Y-2, Y-3 (weights: 5/3/2) |
| `trend_{stat}`        | Y-1 minus Y-2 (momentum signal)                       |

Generated for all 6 target stats (obp, slg, hr, r, rbi, sb), 3 expected Statcast stats (xwOBA, xBA, xSLG), and 6 per-PA rate stats (hr_per_pa, r_per_pa, rbi_per_pa, sb_per_pa, sb_attempt_rate, iso) = 45 temporal features total. Weights are configurable per stat via `temporal_stat_weights` in `configs/data.yaml` (e.g., slow-stabilizing stats like BABIP use equal 3/3/3 weighting; SB-related stats use [7,2,1] to down-weight pre-2023 rule change data).

### 4.6 Missing Data Strategy

| Feature Group               | Strategy                                                                        |
| --------------------------- | ------------------------------------------------------------------------------- |
| Bat speed (pre-2024)        | Impute with league-average bat speed, add `has_bat_speed` indicator column      |
| Sprint speed (rare missing) | Impute with position-group median                                               |
| Statcast metrics            | Require non-null; drop player-seasons with < 50 batted ball events              |
| Non-contact rates           | Stabilisation-regressed toward league average (handles low-PA automatically)    |
| Multi-year features         | If Y-2 or Y-3 missing (e.g., rookie), use available years with adjusted weights |

---

## 5) Modeling Approach — MTL (Multi-Task Learning) Neural Network

**Architecture:**

- Shared-backbone architecture ([256, 128, 64] → per-target heads).
- Three-group head architecture: rate heads (OBP, SLG) run first; detached predictions feed count-rate heads (HR/PA, R/PA, RBI/PA); speed heads (SB/PA) receive backbone only.
- All 6 targets are predicted as per-PA rates internally. Count stats are converted to season totals at inference via Marcel PA projection.
- Multi-seed ensemble (5 seeds), Huber loss with homoscedastic uncertainty weighting, cosine annealing warm restarts, mixup augmentation, sample-level recency weighting.
- **Sample recency weighting**: exponential decay `weight(season) = exp(-λ(max_season - season))` with λ=0.30 (half-life ~2.3 years), normalized to mean=1.0. Weights recent seasons' patterns more heavily when fitting model parameters, complementing per-player temporal features. All major projection systems (Marcel, ZiPS, Steamer) use analogous recency weighting.
- Config: `configs/mtl.yaml`
- Benchmarks at **8.882 mean RMSE** (pooled 2022-2025), **2.9% ahead of ZiPS** (9.151).

**Why rate-based targets**: Aligns with how every major projection system works (ZiPS, Steamer, PECOTA, Marcel). Per-PA rates are more predictable year-to-year (HR rate r=0.79 vs raw counts which depend on playing time), cleanly separate skill from availability, and naturally handle the 2020 shortened season (60 games) since rates are season-length invariant.

**Why MTL**: HR/PA correlates with SLG; SB/PA correlates with sprint speed; R/PA and RBI/PA correlate with lineup position and OBP. Multi-task sharing captures these cross-stat dependencies.

---

## 6) Evaluation

### 6.1 Required Metrics (every run must report)

| Metric   | Scope             | Notes                                        |
| -------- | ----------------- | -------------------------------------------- |
| **RMSE** | Per-target + mean | Primary metric                               |
| **MAE**  | Per-target + mean | Robust to outliers                           |
| **R²**   | Per-target + mean | Explained variance                           |
| **MAPE** | Per-target        | Mean absolute percentage error (for context) |

### 6.2 Baseline Comparison

Every evaluation run must be compared against the **naive persistence** baseline: predict Y+1 = Y (last year's stats).

### 6.3 Evaluation Splits

- **Train**: 2016-2022 seasons (features) -> 2017-2023 targets
- **Validation**: 2023 features -> 2024 targets
- **Test**: 2024 features -> 2025 targets
- **Production (2026)**: retrain on all data (2016-2025 features -> 2017-2025 targets + 2025 features -> 2026 predictions)

### 6.4 Analysis Outputs

- Calibration plots: predicted vs actual scatter for each target
- Residual distributions: histograms with KDE per target
- MTL training curves (loss + validation RMSE)
- Cross-target residual correlation heatmaps

---

## 7) Current Implementation Status

Preseason pipeline is shipped end-to-end: data ingestion, feature engineering, MTL model, holdout/backtest evaluation, 2026 predictions, and public projection benchmarking. An in-season data layer (weekly snapshots) and ROS evaluation harness are in place as the foundation for the rest-of-season projector — no in-season model consumes them yet.

### 7.1 What Was Built

| Component                  | Files                                                       | Description                                                                                                                                                                |
| -------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data Pipeline**          | `src/data/fetch_*.py`, `merge.py`, `splits.py`              | FanGraphs + Statcast + speed + context data download, merge on (player_id, season), target alignment (Y+1), chronological splits                                           |
| **Feature Engineering**    | `src/features/*.py`                                         | Registered features across 9 groups (some excluded via config): batting, statcast, non-contact (stabilisation-regressed), speed, bat speed, age, park factors, team stats, temporal |
| **Non-Contact**            | `src/features/non_contact.py`                               | Stabilisation-regressed rates (K%, BB%, HBP%, BABIP, ISO, HR/BBE)                                                                                                          |
| **MTL Network**            | `src/models/mtl/`                                           | Multi-task neural network: three-group rate/count/speed heads, 5-seed ensemble, Huber loss, sample recency weighting, optional target winsorization, holdout/backtest workflows |
| **Evaluation**             | `src/eval/metrics.py`, `report.py`, `plots.py`              | RMSE/MAE/R²/MAPE, naive baseline comparison, calibration/residual plots                                                                                                    |
| **Projection & Benchmark** | `scripts/generate_projections.py`, `benchmark_vs_public.py` | 2026 projections with ensemble, multi-year rolling benchmark vs public projections                                                                                          |
| **Public Projections**     | `src/data/fetch_projections.py`                             | Fetch Steamer/ZiPS/The Bat/The Bat X from FanGraphs API, merge with our projections for side-by-side comparison                                                             |
| **Weekly Snapshot Layer**  | `src/data/fetch_game_logs.py`, `build_snapshots.py`, `fetch_statcast.py` (`_aggregate_batter_statcast_weekly`) | Per-(player, ISO-week) BRef batting logs + Statcast BBE aggregates → weekly snapshots with `*_week`, `*_ytd`, `trail4w_*`, and `ros_*` columns. Raw Statcast retains `game_date`. Data layer only — not yet consumed by any model. |
| **ROS Evaluation Harness** | `src/eval/ros_metrics.py`, `scripts/benchmark_ros.py` | Pinball loss, PIT coverage, PA-checkpoint row selection, plus a rolling ROS benchmark at 50/100/200/400 PA checkpoints with three baselines: `persist_observed`, `frozen_preseason`, `marcel_blend`. |

### 7.2 Current Reporting Outputs

- Holdout evaluation report: `data/reports/mtl_report.json`
- Backtest evaluation report: `data/reports/mtl_backtest_report.json`

### 7.3 Benchmark Results

Pooled RMSE across 2022-2025 (4 years, 1,063 player-seasons, rolling retrain):

| System  | OBP    | SLG    | HR   | R     | RBI   | SB   | Mean   |
| ------- | ------ | ------ | ---- | ----- | ----- | ---- | ------ |
| **MTL** | 0.0306 | 0.0621 | 7.49 | 19.01 | 19.83 | 6.87 | 8.882  |
| Steamer | 0.0294 | 0.0602 | 7.96 | 20.94 | 21.55 | 6.53 | 9.512  |
| ZiPS    | 0.0289 | 0.0594 | 7.45 | 19.71 | 21.24 | 6.41 | 9.151  |
| Naive   | 0.0352 | 0.0726 | 8.65 | 22.37 | 23.23 | 7.35 | 10.284 |

MTL is **2.9% ahead of ZiPS** and **6.6% ahead of Steamer** on aggregate mean RMSE. MTL wins on HR (7.49), R (19.01), and RBI (19.83). Remaining gap to ZiPS is concentrated in OBP (+5.9%), SLG (+4.5%), and SB (+7.2%).

**Ablation results** (which MTL features help):

- Sample recency weighting (λ=0.30): **-0.32%** isolated, **-2.11%** combined with ensemble
- Multi-seed ensemble (H10): **-1.68%** — largest single-factor improvement
- Three-group heads (speed_head_indices): **-0.50%** combined with SB feature/temporal changes
- Two-stage rate→count (H12): +0.31% — marginal, kept for architectural clarity
- Target winsorization (H13): -0.11% — negligible effect
- Stat-specific aging curves (H14): +0.00% — excluded (no benefit on small dataset)

### 7.4 CLI Reference

```bash
# Data pipeline
uv run python -m src.data.fetch_all --seasons 2016-2025            # All sources (raw + agg per season)
uv run python -m src.data.fetch_raw_statcast --seasons 2016-2025   # Raw BBE data only
uv run python -m src.data.fetch_statcast --seasons 2016-2025       # Aggregate from local raw files
uv run python -m src.data.fetch_statcast --seasons 2016-2025 --from-api  # Force pybaseball API fetch
uv run python -m src.data.fetch_statcast --seasons 2016-2025 --force     # Re-aggregate
uv run python -m src.data.merge
uv run python -m src.data.fetch_projections --year 2026                  # Public projections (Steamer/ZiPS/Bat/BatX)
uv run python -m src.data.fetch_projections --year 2026 --systems steamer zips  # Specific systems only

# Weekly snapshot data layer (in-season ROS pipeline, Phase 1)
uv run python -m src.data.fetch_game_logs --seasons 2016-2026            # BRef per-(batter, ISO-week) batting logs
uv run python -m src.data.build_snapshots --seasons 2016-2026            # Merge weekly BRef + Statcast → weekly_snapshots_{year}.parquet

# ROS benchmark (PA checkpoints 50/100/200/400)
uv run python scripts/benchmark_ros.py --years 2023 2024 2025                   # persist_observed only (no retraining)
uv run python scripts/benchmark_ros.py --years 2023 2024 2025 --retrain         # + frozen_preseason + marcel_blend (retrains MTL per year)

# Training
uv run python -m src.models.mtl.train --config configs/mtl.yaml              # MTL holdout
uv run python -m src.models.mtl.train --config configs/mtl.yaml --backtest --device cpu  # rolling backtest

# Generate 2026 projections
uv run python scripts/generate_projections.py                          # MTL projections
uv run python scripts/generate_projections.py --retrain                # Retrain from scratch
uv run python scripts/generate_projections.py --with-public            # Compare with public projections
uv run python scripts/generate_projections.py --fetch-public           # Fetch + compare in one step

# Benchmark vs public projections (multi-year rolling evaluation)
uv run python scripts/benchmark_vs_public.py                                 # Full 2022-2025 benchmark
uv run python scripts/benchmark_vs_public.py --years 2024 2025               # Specific years
uv run python scripts/benchmark_vs_public.py --no-retrain                    # Saved models, latest year only
uv run python scripts/benchmark_vs_public.py --with-plots                    # + comparison plots

# Tests
uv run pytest tests/ -v
```

---

## 8) Directory Structure

```
baseball-hydra/
├── CLAUDE.md                          # This file — authoritative build plan
├── README.md                          # Setup, usage, quick start
├── pyproject.toml                     # Dependencies (uv)
├── configs/
│   ├── data.yaml                      # Data pipeline config
│   └── mtl.yaml                      # MTL config
├── data/
│   ├── .gitignore                     # Ignores *.parquet, *.csv, *.pkl, *.pt, *.json, *.png, *.npz
│   ├── raw/                           # Per-year cached parquet files
│   ├── external_projections/          # Public projections (Steamer, ZiPS, etc.) as CSV
│   ├── models/                        # Trained model artifacts
│   ├── reports/                       # Evaluation report JSON files
│   │   ├── benchmark/                 # Multi-year benchmark vs public projections
│   │   └── benchmark_ros/             # ROS benchmark outputs + preseason/ MTL cache
│   └── projections/                   # Our model projection CSV files
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetch_batting.py           # FanGraphs batting stats download
│   │   ├── fetch_raw_statcast.py      # Raw Statcast BBE download (pitch-level, retains game_date)
│   │   ├── fetch_statcast.py          # Statcast aggregation (season + ISO-week, API fallback)
│   │   ├── fetch_game_logs.py         # BRef per-(batter, ISO-week) batting logs
│   │   ├── fetch_speed.py             # Sprint speed + bat speed download
│   │   ├── fetch_context.py           # Park factors + team stats download
│   │   ├── fetch_projections.py        # FanGraphs public projections (Steamer/ZiPS/Bat/BatX)
│   │   ├── fetch_all.py               # Unified CLI: download all sources
│   │   ├── merge.py                   # Merge all sources -> modeling dataset
│   │   ├── build_snapshots.py         # Weekly BRef + Statcast → weekly_snapshots_{year}.parquet
│   │   └── splits.py                  # Chronological split logic
│   ├── features/
│   │   ├── __init__.py
│   │   ├── batting.py                 # Derived batting features
│   │   ├── statcast.py                # Statcast quality features
│   │   ├── non_contact.py             # Stabilisation-regressed non-contact rates
│   │   ├── temporal.py                # Multi-year weighted averages, trends
│   │   ├── context.py                 # Park factors, team stats, age
│   │   ├── pipeline.py                # End-to-end feature pipeline
│   │   └── registry.py                # Feature name registry (some excluded via config)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── utils.py                   # Shared: align_features, model configs, train_model_for_year
│   │   └── mtl/
│   │       ├── __init__.py
│   │       ├── model.py               # MTLForecaster
│   │       ├── loss.py                # Multi-task losses
│   │       ├── dataset.py             # PyTorch Dataset
│   │       └── train.py               # CLI: holdout/backtest MTL
│   └── eval/
│       ├── __init__.py
│       ├── metrics.py                 # RMSE, MAE, R², MAPE
│       ├── ros_metrics.py             # ROS pinball/PIT metrics, PA-checkpoint selection
│       ├── report.py                  # Generate evaluation reports (JSON + console)
│       └── plots.py                   # Calibration, residual, comparison plots
├── scripts/
│   ├── generate_projections.py         # Generate 2026 season projections
│   ├── benchmark_vs_public.py         # Multi-year rolling benchmark vs public projections
│   ├── benchmark_ros.py               # Rolling ROS benchmark at PA checkpoints
│   └── run_ablation.py                # Ablation study for MTL config variants
└── tests/
    ├── __init__.py
    ├── test_data_pipeline.py          # Data merging + splits
    ├── test_features.py               # Feature engineering
    ├── test_non_contact_features.py   # Non-contact features
    ├── test_mtl.py                   # MTL core tests
    ├── test_backtest.py               # Backtest fold and report behavior
    ├── test_projections.py            # Public projections fetch, load, merge
    ├── test_plots_and_predictions.py   # Plots and prediction helpers
    ├── test_weekly_snapshots.py        # Weekly snapshot pipeline (ISO weeks, ytd/ros invariants)
    ├── test_ros_metrics.py             # ROS metrics (pinball, PIT, PA checkpoints)
    └── test_benchmark_ros.py           # ROS benchmark baselines + evaluation flow
```

---

## 9) Code Standards

- **Formatter / linter**: `ruff` + `black` (line length 88)
- **Type hints**: required on all public functions
- **Config-driven**: all hyperparameters in YAML; no magic numbers in training code
- **Deterministic seeds**: set `torch`, `numpy`, and `random` seeds from a single config value
- **Logging**: all training runs log config, seed, git commit hash (if available), data file hash

---

## 10) Testing

Tests are spread across multiple test files in `tests/`. Run with `uv run pytest tests/`.

Required test coverage:

- **Data pipeline**: merge correctness, target alignment (Y+1), no temporal leakage, correct splits
- **Features**: computation correctness, missing data handling, no future data in features
- **Model smoke tests**: train on tiny synthetic data, verify output shapes and types, checkpoint save/load roundtrip
- **Evaluation**: metric computation against known values
- **Plots and predictions**: plot functions return correct Figure objects, prediction helpers

---

## 11) gstack

Use the `/browse` skill from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

Available skills: `/office-hours`, `/plan-ceo-review`, `/plan-eng-review`, `/plan-design-review`, `/design-consultation`, `/review`, `/ship`, `/land-and-deploy`, `/canary`, `/benchmark`, `/browse`, `/qa`, `/qa-only`, `/design-review`, `/setup-browser-cookies`, `/setup-deploy`, `/retro`, `/investigate`, `/document-release`, `/codex`, `/cso`, `/autoplan`, `/careful`, `/freeze`, `/guard`, `/unfreeze`, `/gstack-upgrade`.
