# Baseball Hydra

**An MLB projection system built using a multi-task learning (MTL) neural network**

Baseball Hydra predicts batter stats — OBP, SLG, HR, R, RBI, and SB — in two flavors:

1. **Preseason (full-season)** — a multi-task learning (MTL) neural network trained on a decade of FanGraphs + Statcast + Baseball Savant data, benchmarked against ZiPS and Steamer over 1,063 player-seasons (2022-2025).
2. **In-season rest-of-season (ROS)** — a closed-form Beta-Binomial shrinkage baseline that blends the preseason MTL prior with year-to-date counts from a weekly BRef + Statcast snapshot pipeline, evaluated at 50/100/200/400 PA checkpoints across 2023-2025.

The codebase was developed using Claude Code. [Here's the link](https://lambertchu.com/blog/vibe-coding-baseball) to my blog post describing my process of building this model.

### Why multi-task learning?

Traditional projection systems model each stat independently. MTL trains a single neural network with a shared backbone that feeds into stat-specific prediction heads, so the model learns cross-stat relationships automatically: exit velocity informs HR projections, sprint speed shapes SB predictions, and OBP context flows into R and RBI estimates. The result is a model where improving one prediction can improve them all.

### How the preseason MTL stacks up

Benchmarked over 1,063 player-seasons (2022-2025, rolling retrain):

| System      | OBP    | SLG    | HR   | R     | RBI   | SB   | Mean RMSE |
| ----------- | ------ | ------ | ---- | ----- | ----- | ---- | --------- |
| **MTL**     | 0.0306 | 0.0621 | 7.49 | 19.01 | 19.83 | 6.87 | **8.882** |
| ZiPS        | 0.0289 | 0.0594 | 7.45 | 19.71 | 21.24 | 6.41 | 9.151     |
| Steamer     | 0.0294 | 0.0602 | 7.96 | 20.94 | 21.55 | 6.53 | 9.512     |
| Last season | 0.0352 | 0.0726 | 8.65 | 22.37 | 23.23 | 7.35 | 10.284    |

**2.9% ahead of ZiPS** and **6.6% ahead of Steamer** on aggregate mean RMSE, with the largest gains on counting stats (R and RBI).

### How the ROS pipeline stacks up

Pooled mean RMSE across 2023-2025 weekly snapshots at each PA checkpoint:

| PA checkpoint | PersistObs | FrozenPre | MarcelBlend | **Shrinkage (prod)** | Phase2 (parked) |
| ------------- | ---------- | --------- | ----------- | -------------------- | --------------- |
| 50            | 0.0591     | 0.0309    | 0.0313      | **0.0303**           | 0.0314          |
| 100           | 0.0481     | 0.0312    | 0.0322      | **0.0307**           | 0.0321          |
| 200           | 0.0418     | 0.0340    | 0.0346      | **0.0332**           | 0.0348          |
| 400           | 0.0414     | 0.0386    | 0.0387      | **0.0376**           | 0.0401          |

The closed-form Bayesian shrinkage baseline wins every PA checkpoint. An experimental Phase 2 quantile-head MTL (`src/models/mtl_ros/`) is fully implemented, tested, and benchmarked but underperforms shrinkage by ~4-5% — most likely due to weekly snapshot data being backfilled only for 2023-2025 (~27k cutoff rows). The Phase 2 code stays in-tree for a future retry once 2016-2022 snapshots are backfilled.

## Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (a Python package manager).

## Quick Start

```bash
uv sync   # Install dependencies

# Step 1: Download all data (7 sources per season)
uv run python -m src.data.fetch_all --seasons 2016-2025

# Step 2: Merge into modeling dataset
uv run python -m src.data.merge

# Step 3: Train MTL
uv run python -m src.models.mtl.train --config configs/mtl.yaml   # holdout evaluation
uv run python -m src.models.mtl.train --config configs/mtl.yaml --backtest --device cpu   # rolling backtest

# Step 4: Generate 2026 projections
uv run python scripts/generate_projections.py   # MTL projections

# Step 5 (optional): Fetch public projections and compare
uv run python -m src.data.fetch_projections --year 2026   # download Steamer/ZiPS/Bat/BatX
uv run python scripts/generate_projections.py --with-public   # side-by-side with public projections

# Step 6 (optional): Multi-year benchmark vs public projections
uv run python scripts/benchmark_vs_public.py   # full 2022-2025 benchmark
uv run python scripts/benchmark_vs_public.py --years 2025   # quick smoke test
uv run python scripts/benchmark_vs_public.py --with-plots   # + comparison plots

# Step 7 (optional): Build weekly snapshots for the in-season ROS pipeline
uv run python -m src.data.fetch_game_logs --seasons 2016-2026   # BRef weekly batting logs
uv run python -m src.data.build_snapshots --seasons 2016-2026   # Weekly snapshots with ytd + ros targets

# Step 8 (optional): ROS benchmark at PA checkpoints (50 / 100 / 200 / 400)
uv run python scripts/benchmark_ros.py --years 2023 2024 2025                           # persist_observed baseline (no retraining)
uv run python scripts/benchmark_ros.py --years 2023 2024 2025 --retrain                 # + frozen_preseason, marcel_blend, shrinkage, phase2 (retrains preseason MTL + Phase 2 ensemble per year)
uv run python scripts/benchmark_ros.py --years 2023 2024 2025 --fit-shrinkage-tau       # fit per-stat τ₀ via leave-one-year-out cross-fitting
uv run python scripts/benchmark_ros.py --years 2023 2024 2025 --retrain --pit-plot      # + PIT histograms for shrinkage & phase2

# Step 9 (optional): Phase 2 ROS quantile MTL — train and generate current-year projections
uv run python -m src.models.mtl_ros.train --config configs/mtl_ros.yaml                 # train + eval on 2023/2024 snapshots
uv run python scripts/generate_ros_projections.py --year 2026                           # 2026 ROS projections → data/projections/ros_mtl_2026.csv
```

The training scripts handle feature engineering, train/val/test splitting, model training,
and evaluation automatically. Each prints a per-target metrics table and comparison against
the naive persistence baseline (Y+1 = Y).

## Data Sources

- **FanGraphs**: batting stats, park factors (via pybaseball)
- **Statcast raw BBE**: pitch-level batted-ball events cached as `statcast_raw_{year}.parquet` (via pybaseball)
- **Statcast aggregated**: per-batter metrics aggregated from local raw files (API fallback if raw missing)
- **Baseball Savant**: sprint speed (2016+), bat speed (2024+)
- **Baseball Reference**: per-(batter, ISO-week) batting logs for in-season ROS snapshots (via pybaseball)
- **Team stats**: team offensive environment for lineup context
- **Public projections**: Steamer, ZiPS, The Bat, The Bat X (comparison only, not used in training). Current-year projections can be fetched via the FanGraphs API. Historical projections must be downloaded manually from [FanGraphs Projections](https://www.fangraphs.com/projections) and require a paid FanGraphs membership.

## Feature Pipeline (162 registered; ~121 active for preseason after exclusions)

| Group            | Features                                                                                                                                                                                                                | Count        |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| **Batting**      | pa, bb_rate, k_rate, iso, babip, avg, obp, slg, hr, sb, cs, sb_rate, woba, wrc_plus, hbp_rate, contact_rate, per-PA rates (hr, r, rbi, sb, sb_attempt), ibb_rate, ubb_rate, singles/doubles/triples/extra_base/cs rates | 28           |
| **Statcast**     | avg*exit_velocity, ev_p95, max_exit_velocity, avg_launch_angle, barrel_rate, hard_hit_rate, sweet_spot_rate, xwOBA/xBA/xSLG, bbe_count + 11 has*\* indicators                                                           | 22           |
| **Non-Contact**  | regressed k_rate/bb_rate/hbp_rate/babip/iso/hr_per_bbe                                                                                                                                                                  | 6            |
| **Sprint Speed** | sprint_speed, has_sprint_speed                                                                                                                                                                                          | 2            |
| **Bat Speed**    | avg*bat_speed, avg_swing_speed, squared_up/blast/fast_swing rates, tracking counts + 10 has*\* indicators                                                                                                               | 20           |
| **Age**          | age, age_squared, age_delta_speed/power/patience (computed but excluded by default)                                                                                                                                     | 5 (2 active) |
| **Park Factors** | park_factor_runs, park_factor_hr                                                                                                                                                                                        | 2            |
| **Team Stats**   | team_runs_per_game, team_ops, team_sb, sb_rule_era, sb_era_x_speed, speed_age_interaction, team_sb_per_game, sb_era_x_attempt_rate                                                                                      | 8            |
| **Temporal**     | prev_year/weighted_avg/trend for 6 targets + 3 xStats + 6 per-PA rates                                                                                                                                                  | 45           |
| **In-Season** (Phase 2, opt-in) | 10 ytd passthroughs (pa/obp/slg/hr_per_pa/r_per_pa/rbi_per_pa/sb_per_pa/iso/bb_rate/k_rate) + 10 trail4w rates + week_index + pa_fraction + 2 IL stubs                                                | 24           |

Feature groups and exclusions are config-driven via `configs/data.yaml` (preseason) and `configs/mtl_ros.yaml` (ROS). Redundant features (hard_hit_rate, sweet_spot_rate, bat speed indicators pre-2024) and aging delta features (hurt benchmark RMSE per ablation testing) are excluded by default. The **In-Season** group is opt-in — it's only enabled by the Phase 2 ROS training path.

## Data Directory Layout

All generated data lives under `data/` and is gitignored:

```
data/
├── raw/                           # Per-year cached parquets (from fetch_all)
│   ├── batting_{year}.parquet
│   ├── statcast_raw_{year}.parquet      # Raw BBE (pitch-level, retains game_date)
│   ├── statcast_agg_{year}.parquet      # Aggregated per-batter metrics
│   ├── statcast_agg_week_{year}.parquet # Per-(batter, ISO-week) BBE metrics
│   ├── batting_week_{year}.parquet      # BRef per-(batter, ISO-week) batting logs
│   ├── weekly_snapshots_{year}.parquet  # Weekly snapshots with ytd + ros targets
│   ├── sprint_speed_{year}.parquet
│   ├── bat_speed_{year}.parquet         # 2024+ only
│   ├── park_factors_{year}.parquet
│   └── team_batting_{year}.parquet
├── external_projections/          # Public projections (comparison only)
│   ├── steamer_{year}.csv
│   ├── zips_{year}.csv
│   ├── thebat_{year}.csv
│   └── thebatx_{year}.csv
├── merged_batter_data.parquet     # Merged preseason modeling dataset (from merge)
├── models/
│   ├── mtl/
│   │   └── mtl_forecaster.pt            # Preseason MTL checkpoint
│   └── mtl_ros_quantile/                # Phase 2 ROS MTL checkpoints
├── reports/
│   ├── mtl_report.json
│   ├── mtl_backtest_report.json
│   ├── benchmark/                 # Multi-year benchmark vs public projections
│   │   ├── benchmark_report.json
│   │   ├── benchmark_table.csv
│   │   └── benchmark_rmse.png
│   └── benchmark_ros/             # Rolling ROS benchmark at PA checkpoints
│       ├── benchmark_ros_report.json
│       ├── benchmark_ros_table.csv
│       ├── preseason/             # Per-year preseason MTL prediction cache
│       └── phase2/                # Per-year Phase 2 ensemble checkpoints
└── projections/                   # Preseason + ROS projections
    ├── projections_mtl_2026.csv
    ├── projections_vs_external_2026.csv  # Preseason vs public projections
    └── ros_mtl_2026.csv                   # Phase 2 ROS projections (median + p05/p95)
```

## Evaluation

Implemented in `src/eval/`:

- Per-target RMSE, MAE, R², MAPE on chronological test set
- Comparison against naive persistence baseline (Y+1 = Y)
- JSON report export
- Calibration scatter plots (predicted vs actual)
- Residual distribution histograms
- MTL training curves (loss + validation RMSE)
- Marcel PA projection + rate_to_count helper (`src/eval/pa_projection.py`)
- **ROS metrics** (`src/eval/ros_metrics.py`): pinball quantile loss, PIT coverage for calibration, per-player PA-checkpoint row selection — used by the ROS benchmark script
- **Shrinkage baseline** (`src/models/baselines/shrinkage.py`): closed-form Beta-Binomial posterior per stat (OBP, SLG, HR/PA, R/PA, RBI/PA, SB/PA) with per-stat pseudocount τ₀ (stabilisation-based defaults or fit via leave-one-year-out cross-fitting); emits both posterior means and Beta-CDF quantiles for pinball/PIT
- **Phase 2 MTL ROS** (`src/models/mtl_ros/`): quantile-regressing MTL with a 7th PA-remaining head, pinball + Kendall uncertainty weighting, multi-seed ensemble with monotonic quantile sort

## Project Structure

See [AGENTS.md](AGENTS.md) (also available as [CLAUDE.md](CLAUDE.md) via symlink) for the full engineering plan, data contract, and feature specification.

## Running Tests

```bash
uv run pytest tests/ -v
```
