# 2026 preseason evaluation

[Read the analysis](evaluation_report.md), including the snapshot-timing caveat.
This compares the saved Hydra forecast with eight public systems and a repeat-2025
baseline. It does not retrain or change a production model.

## Verify from a clean checkout

After installing the project's dependencies (`uv sync`), run from the repo root:

```bash
uv run python data/reports/preseason_2026/verify_results.py
```

This command needs no network, model checkpoints, or untracked source caches.
It independently recomputes every per-target and aggregate metric, playing-time
metric, and paired bootstrap interval from the committed player-level snapshots.
It also reconstructs the cohorts and checks the FanGraphs/MLB actuals comparison.
It does not modify the report or other files.

## Evidence

- [Player comparison](player_comparison.csv): all 243 Hydra players, matched
  actuals, each system's saved forecasts and projected PA, and cohort membership.
- [Earlier Hydra export](earlier_hydra.csv): the March 29 forecasts used only for
  the timing sensitivity check, keyed to the same player IDs.
- [FanGraphs actuals](actuals_2026_fangraphs.csv) and
  [MLB actuals](actuals_2026_mlb.csv): compact extracts from the October 29 download.
- [Per-target metrics](metrics_by_target.csv), [summaries](summary_metrics.csv),
  [playing time](playing_time_metrics.csv), and
  [paired bootstrap intervals](paired_bootstrap.csv).
- [Manifest](manifest.json): original source hashes, snapshot timestamps, code
  revision, seed and configuration. Original file modification dates do not prove
  an immutable pre-Opening-Day freeze.

![Accuracy against naive persistence](accuracy_heatmap.png)

## Rebuild from the original local caches

`evaluate.py` is the original analysis driver. Unlike the portable verification
command, it requires every original input listed in `evaluation.yaml` and
`manifest.json`: the saved public projection CSVs, 2016–2025 merged parquet,
2025 batting parquet, earlier Hydra export, and both raw API JSON responses.
Those bulk caches remain untracked. Do not replace historical forecasts with
fresh downloads; current public endpoints can return different forecasts.

With the original caches restored, run from the repository root:

```bash
uv run python data/reports/preseason_2026/evaluate.py
```

That command rewrites the derived CSVs, plots, report and manifest in this folder.
The report's narrative is specific to this frozen 2026 analysis, not a template
for other seasons. The normalized MLB and earlier-Hydra verification snapshots
are also rebuilt from their respective original sources.
