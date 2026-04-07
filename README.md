# Baseball Hydra

**An MLB projection system built using a multi-task learning (MTL) neural network**

Baseball Hydra predicts next-season batter stats — OBP, SLG, HR, R, RBI, and SB — by training a multi-task learning (MTL) neural network on a decade of historical performance, Statcast batted-ball data, sprint and bat speed metrics, ballpark factors, and team context. The codebase was developed using Claude Code.

### Why multi-task learning?

Traditional projection systems model each stat independently. MTL trains a single neural network with a shared backbone that feeds into stat-specific prediction heads, so the model learns cross-stat relationships automatically: exit velocity informs HR projections, sprint speed shapes SB predictions, and OBP context flows into R and RBI estimates. The result is a model where improving one prediction can improve them all.

### How it stacks up

Benchmarked over 1,063 player-seasons (2022-2025, rolling retrain):

| System      | OBP    | SLG    | HR   | R     | RBI   | SB   | Mean RMSE |
| ----------- | ------ | ------ | ---- | ----- | ----- | ---- | --------- |
| **MTL**     | 0.0306 | 0.0621 | 7.49 | 19.01 | 19.83 | 6.87 | **8.882** |
| ZiPS        | 0.0289 | 0.0594 | 7.45 | 19.71 | 21.24 | 6.41 | 9.151     |
| Steamer     | 0.0294 | 0.0602 | 7.96 | 20.94 | 21.55 | 6.53 | 9.512     |
| Last season | 0.0352 | 0.0726 | 8.65 | 22.37 | 23.23 | 7.35 | 10.284    |

**2.9% ahead of ZiPS** and **6.6% ahead of Steamer** on aggregate mean RMSE, with the largest gains on counting stats (R and RBI).

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
```

The training scripts handle feature engineering, train/val/test splitting, model training,
and evaluation automatically. Each prints a per-target metrics table and comparison against
the naive persistence baseline (Y+1 = Y).

## Data Sources

- **FanGraphs**: batting stats, park factors (via pybaseball)
- **Statcast raw BBE**: pitch-level batted-ball events cached as `statcast_raw_{year}.parquet` (via pybaseball)
- **Statcast aggregated**: per-batter metrics aggregated from local raw files (API fallback if raw missing)
- **Baseball Savant**: sprint speed (2016+), bat speed (2024+)
- **Team stats**: team offensive environment for lineup context
- **Public projections**: Steamer, ZiPS, The Bat, The Bat X (comparison only, not used in training). Current-year projections can be fetched via the FanGraphs API. Historical projections must be downloaded manually from [FanGraphs Projections](https://www.fangraphs.com/projections) and require a paid FanGraphs membership.

## Feature Pipeline (138 registered, ~121 active after exclusions)

| Group            | Features                                                                                                                                                                                                                | Count        |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| **Batting**      | pa, bb_rate, k_rate, iso, babip, avg, obp, slg, hr, sb, cs, sb_rate, woba, wrc_plus, hbp_rate, contact_rate, per-PA rates (hr, r, rbi, sb, sb_attempt), ibb_rate, ubb_rate, singles/doubles/triples/extra_base/cs rates | 28           |
| **Statcast**     | avg_exit_velocity, ev_p95, max_exit_velocity, avg_launch_angle, barrel_rate, hard_hit_rate, sweet_spot_rate, xwOBA/xBA/xSLG, bbe_count + 11 has_\* indicators                                                          | 22           |
| **Non-Contact**  | regressed k_rate/bb_rate/hbp_rate/babip/iso/hr_per_bbe                                                                                                                                                                  | 6            |
| **Sprint Speed** | sprint_speed, has_sprint_speed                                                                                                                                                                                          | 2            |
| **Bat Speed**    | avg_bat_speed, avg_swing_speed, squared_up/blast/fast_swing rates, tracking counts + 10 has_\* indicators                                                                                                              | 20           |
| **Age**          | age, age_squared, age_delta_speed/power/patience (computed but excluded by default)                                                                                                                                     | 5 (2 active) |
| **Park Factors** | park_factor_runs, park_factor_hr                                                                                                                                                                                        | 2            |
| **Team Stats**   | team_runs_per_game, team_ops, team_sb, sb_rule_era, sb_era_x_speed, speed_age_interaction, team_sb_per_game, sb_era_x_attempt_rate                                                                                      | 8            |
| **Temporal**     | prev_year/weighted_avg/trend for 6 targets + 3 xStats + 6 per-PA rates                                                                                                                                                  | 45           |

Feature groups and exclusions are config-driven via `configs/data.yaml`. Redundant features (hard_hit_rate, sweet_spot_rate, bat speed indicators pre-2024) and aging delta features (hurt benchmark RMSE per ablation testing) are excluded by default.

## Data Directory Layout

All generated data lives under `data/` and is gitignored:

```
data/
├── raw/                           # Per-year cached parquets (from fetch_all)
│   ├── batting_{year}.parquet
│   ├── statcast_raw_{year}.parquet   # Raw BBE (pitch-level)
│   ├── statcast_agg_{year}.parquet   # Aggregated per-batter metrics
│   ├── sprint_speed_{year}.parquet
│   ├── bat_speed_{year}.parquet      # 2024+ only
│   ├── park_factors_{year}.parquet
│   └── team_batting_{year}.parquet
├── external_projections/          # Public projections (comparison only)
│   ├── steamer_{year}.csv
│   ├── zips_{year}.csv
│   ├── thebat_{year}.csv
│   └── thebatx_{year}.csv
├── merged_batter_data.parquet     # Merged modeling dataset (from merge)
├── models/
│   └── mtl/
│       └── mtl_forecaster.pt        # MTL checkpoint
├── reports/
│   ├── mtl_report.json
│   ├── mtl_backtest_report.json
│   └── benchmark/                 # Multi-year benchmark vs public projections
│       ├── benchmark_report.json
│       ├── benchmark_table.csv
│       └── benchmark_rmse.png
└── projections/                   # Our model projections
    ├── projections_mtl_2026.csv
    └── projections_vs_external_2026.csv  # Our model vs public projections
```

## Evaluation

Implemented in `src/eval/`:

- Per-target RMSE, MAE, R², MAPE on chronological test set
- Comparison against naive persistence baseline (Y+1 = Y)
- JSON report export
- Calibration scatter plots (predicted vs actual)
- Residual distribution histograms
- MTL training curves (loss + validation RMSE)

## Project Structure

See [CLAUDE.md](CLAUDE.md) for the full engineering plan, data contract, and feature specification.

## Running Tests

```bash
uv run pytest tests/ -v   # 251 tests
```
