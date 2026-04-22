"""Feature name registry and metadata.

Central registry of all feature names, their groups, types, and descriptions.
Used by the pipeline to select features based on config and to document the
feature matrix schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FeatureGroup(str, Enum):
    """Feature group identifiers matching config keys."""

    BATTING = "batting"
    STATCAST = "statcast"
    NON_CONTACT = "non_contact"
    SPRINT_SPEED = "sprint_speed"
    BAT_SPEED = "bat_speed"
    PARK_FACTORS = "park_factors"
    TEAM_STATS = "team_stats"
    AGE = "age"
    TEMPORAL = "temporal"
    IN_SEASON = "in_season"


class FeatureType(str, Enum):
    """Semantic type of the feature value."""

    RATE = "rate"
    COUNT = "count"
    CONTINUOUS = "continuous"
    INDICATOR = "indicator"


@dataclass(frozen=True)
class FeatureMeta:
    """Metadata for a single feature column."""

    name: str
    group: FeatureGroup
    ftype: FeatureType
    description: str


# ---------------------------------------------------------------------------
# Batting features (Section 4.1)
# ---------------------------------------------------------------------------
BATTING_FEATURES: list[FeatureMeta] = [
    FeatureMeta("pa", FeatureGroup.BATTING, FeatureType.COUNT, "Plate appearances"),
    FeatureMeta(
        "bb_rate", FeatureGroup.BATTING, FeatureType.RATE, "Walk rate (BB / PA)"
    ),
    FeatureMeta(
        "k_rate", FeatureGroup.BATTING, FeatureType.RATE, "Strikeout rate (SO / PA)"
    ),
    FeatureMeta(
        "iso", FeatureGroup.BATTING, FeatureType.RATE, "Isolated power (SLG - AVG)"
    ),
    FeatureMeta(
        "babip",
        FeatureGroup.BATTING,
        FeatureType.RATE,
        "Batting average on balls in play",
    ),
    FeatureMeta("avg", FeatureGroup.BATTING, FeatureType.RATE, "Batting average"),
    FeatureMeta("obp", FeatureGroup.BATTING, FeatureType.RATE, "On-base percentage"),
    FeatureMeta("slg", FeatureGroup.BATTING, FeatureType.RATE, "Slugging percentage"),
    FeatureMeta("hr", FeatureGroup.BATTING, FeatureType.COUNT, "Home runs"),
    FeatureMeta("sb", FeatureGroup.BATTING, FeatureType.COUNT, "Stolen bases"),
    FeatureMeta("cs", FeatureGroup.BATTING, FeatureType.COUNT, "Caught stealing"),
    FeatureMeta(
        "sb_rate", FeatureGroup.BATTING, FeatureType.RATE, "Stolen base success rate"
    ),
    FeatureMeta(
        "woba", FeatureGroup.BATTING, FeatureType.RATE, "Weighted on-base average"
    ),
    FeatureMeta(
        "wrc_plus",
        FeatureGroup.BATTING,
        FeatureType.CONTINUOUS,
        "Weighted runs created plus",
    ),
    FeatureMeta(
        "hbp_rate",
        FeatureGroup.BATTING,
        FeatureType.RATE,
        "Hit-by-pitch rate (HBP / PA)",
    ),
    FeatureMeta(
        "contact_rate",
        FeatureGroup.BATTING,
        FeatureType.RATE,
        "Contact rate (1 - K% - BB% - HBP%)",
    ),
    FeatureMeta(
        "hr_per_pa",
        FeatureGroup.BATTING,
        FeatureType.RATE,
        "Home runs per plate appearance",
    ),
    FeatureMeta(
        "r_per_pa", FeatureGroup.BATTING, FeatureType.RATE, "Runs per plate appearance"
    ),
    FeatureMeta(
        "rbi_per_pa", FeatureGroup.BATTING, FeatureType.RATE, "RBI per plate appearance"
    ),
    FeatureMeta(
        "sb_per_pa",
        FeatureGroup.BATTING,
        FeatureType.RATE,
        "Stolen bases per plate appearance",
    ),
    FeatureMeta(
        "sb_attempt_rate",
        FeatureGroup.BATTING,
        FeatureType.RATE,
        "Stolen base attempt rate ((SB+CS) / PA)",
    ),
    FeatureMeta(
        "ibb_rate",
        FeatureGroup.BATTING,
        FeatureType.RATE,
        "Intentional walk rate (IBB / PA)",
    ),
    FeatureMeta(
        "ubb_rate",
        FeatureGroup.BATTING,
        FeatureType.RATE,
        "Unintentional walk rate ((BB - IBB) / PA)",
    ),
    FeatureMeta(
        "singles_rate",
        FeatureGroup.BATTING,
        FeatureType.RATE,
        "Singles rate ((H - 2B - 3B - HR) / PA)",
    ),
    FeatureMeta(
        "doubles_rate", FeatureGroup.BATTING, FeatureType.RATE, "Doubles rate (2B / PA)"
    ),
    FeatureMeta(
        "triples_rate", FeatureGroup.BATTING, FeatureType.RATE, "Triples rate (3B / PA)"
    ),
    FeatureMeta(
        "extra_base_rate",
        FeatureGroup.BATTING,
        FeatureType.RATE,
        "Extra-base hit rate ((2B + 3B + HR) / PA)",
    ),
    FeatureMeta(
        "cs_rate",
        FeatureGroup.BATTING,
        FeatureType.RATE,
        "Caught stealing rate (CS / PA)",
    ),
]

# ---------------------------------------------------------------------------
# Statcast features (Section 4.2)
# ---------------------------------------------------------------------------
STATCAST_FEATURES: list[FeatureMeta] = [
    FeatureMeta(
        "bbe_count",
        FeatureGroup.STATCAST,
        FeatureType.COUNT,
        "Batted-ball events count",
    ),
    FeatureMeta(
        "avg_exit_velocity",
        FeatureGroup.STATCAST,
        FeatureType.CONTINUOUS,
        "Average exit velocity (mph)",
    ),
    FeatureMeta(
        "ev_p95",
        FeatureGroup.STATCAST,
        FeatureType.CONTINUOUS,
        "95th percentile exit velocity (mph)",
    ),
    FeatureMeta(
        "max_exit_velocity",
        FeatureGroup.STATCAST,
        FeatureType.CONTINUOUS,
        "Maximum exit velocity (mph)",
    ),
    FeatureMeta(
        "avg_launch_angle",
        FeatureGroup.STATCAST,
        FeatureType.CONTINUOUS,
        "Average launch angle (degrees)",
    ),
    FeatureMeta(
        "barrel_rate", FeatureGroup.STATCAST, FeatureType.RATE, "Barrel rate (% of BBE)"
    ),
    FeatureMeta(
        "hard_hit_rate",
        FeatureGroup.STATCAST,
        FeatureType.RATE,
        "Hard hit rate (EV >= 95 mph)",
    ),
    FeatureMeta(
        "sweet_spot_rate",
        FeatureGroup.STATCAST,
        FeatureType.RATE,
        "Sweet spot rate (LA 8-32 deg)",
    ),
    FeatureMeta(
        "estimated_woba_using_speedangle",
        FeatureGroup.STATCAST,
        FeatureType.RATE,
        "Expected wOBA from speed-angle model",
    ),
    FeatureMeta(
        "estimated_ba_using_speedangle",
        FeatureGroup.STATCAST,
        FeatureType.RATE,
        "Expected batting average from speed-angle model",
    ),
    FeatureMeta(
        "estimated_slg_using_speedangle",
        FeatureGroup.STATCAST,
        FeatureType.RATE,
        "Expected slugging from speed-angle model",
    ),
    FeatureMeta(
        "has_bbe_count",
        FeatureGroup.STATCAST,
        FeatureType.INDICATOR,
        "BBE count observed (1) or imputed (0)",
    ),
    FeatureMeta(
        "has_avg_exit_velocity",
        FeatureGroup.STATCAST,
        FeatureType.INDICATOR,
        "Avg EV observed (1) or imputed (0)",
    ),
    FeatureMeta(
        "has_ev_p95",
        FeatureGroup.STATCAST,
        FeatureType.INDICATOR,
        "EV p95 observed (1) or imputed (0)",
    ),
    FeatureMeta(
        "has_max_exit_velocity",
        FeatureGroup.STATCAST,
        FeatureType.INDICATOR,
        "Max EV observed (1) or imputed (0)",
    ),
    FeatureMeta(
        "has_avg_launch_angle",
        FeatureGroup.STATCAST,
        FeatureType.INDICATOR,
        "Avg LA observed (1) or imputed (0)",
    ),
    FeatureMeta(
        "has_barrel_rate",
        FeatureGroup.STATCAST,
        FeatureType.INDICATOR,
        "Barrel rate observed (1) or imputed (0)",
    ),
    FeatureMeta(
        "has_hard_hit_rate",
        FeatureGroup.STATCAST,
        FeatureType.INDICATOR,
        "Hard-hit rate observed (1) or imputed (0)",
    ),
    FeatureMeta(
        "has_sweet_spot_rate",
        FeatureGroup.STATCAST,
        FeatureType.INDICATOR,
        "Sweet-spot rate observed (1) or imputed (0)",
    ),
    FeatureMeta(
        "has_estimated_woba_using_speedangle",
        FeatureGroup.STATCAST,
        FeatureType.INDICATOR,
        "Expected wOBA observed (1) or imputed (0)",
    ),
    FeatureMeta(
        "has_estimated_ba_using_speedangle",
        FeatureGroup.STATCAST,
        FeatureType.INDICATOR,
        "Expected BA observed (1) or imputed (0)",
    ),
    FeatureMeta(
        "has_estimated_slg_using_speedangle",
        FeatureGroup.STATCAST,
        FeatureType.INDICATOR,
        "Expected SLG observed (1) or imputed (0)",
    ),
]

# ---------------------------------------------------------------------------
# Non-contact features (stabilisation-regressed rates)
# ---------------------------------------------------------------------------
NON_CONTACT_FEATURES: list[FeatureMeta] = [
    FeatureMeta(
        "regressed_k_rate",
        FeatureGroup.NON_CONTACT,
        FeatureType.RATE,
        "Stabilisation-regressed strikeout rate",
    ),
    FeatureMeta(
        "regressed_bb_rate",
        FeatureGroup.NON_CONTACT,
        FeatureType.RATE,
        "Stabilisation-regressed walk rate",
    ),
    FeatureMeta(
        "regressed_hbp_rate",
        FeatureGroup.NON_CONTACT,
        FeatureType.RATE,
        "Stabilisation-regressed HBP rate",
    ),
    FeatureMeta(
        "regressed_babip",
        FeatureGroup.NON_CONTACT,
        FeatureType.RATE,
        "Stabilisation-regressed BABIP (stab=1200 PA)",
    ),
    FeatureMeta(
        "regressed_iso",
        FeatureGroup.NON_CONTACT,
        FeatureType.RATE,
        "Stabilisation-regressed ISO (stab=160 PA)",
    ),
    FeatureMeta(
        "regressed_hr_per_bbe",
        FeatureGroup.NON_CONTACT,
        FeatureType.RATE,
        "Stabilisation-regressed HR per BBE (stab=60 BBE)",
    ),
]

# ---------------------------------------------------------------------------
# Speed features (Section 4.3)
# ---------------------------------------------------------------------------
SPRINT_SPEED_FEATURES: list[FeatureMeta] = [
    FeatureMeta(
        "sprint_speed",
        FeatureGroup.SPRINT_SPEED,
        FeatureType.CONTINUOUS,
        "Sprint speed (ft/s)",
    ),
    FeatureMeta(
        "has_sprint_speed",
        FeatureGroup.SPRINT_SPEED,
        FeatureType.INDICATOR,
        "Sprint speed observed (1) or imputed (0)",
    ),
]

BAT_SPEED_FEATURES: list[FeatureMeta] = [
    FeatureMeta(
        "avg_bat_speed",
        FeatureGroup.BAT_SPEED,
        FeatureType.CONTINUOUS,
        "Average bat speed (mph)",
    ),
    FeatureMeta(
        "avg_swing_speed",
        FeatureGroup.BAT_SPEED,
        FeatureType.CONTINUOUS,
        "Average swing speed (mph)",
    ),
    FeatureMeta(
        "squared_up_rate",
        FeatureGroup.BAT_SPEED,
        FeatureType.RATE,
        "Squared-up contact rate",
    ),
    FeatureMeta("blast_rate", FeatureGroup.BAT_SPEED, FeatureType.RATE, "Blast rate"),
    FeatureMeta(
        "fast_swing_rate", FeatureGroup.BAT_SPEED, FeatureType.RATE, "Fast swing rate"
    ),
    FeatureMeta(
        "bat_tracking_swings",
        FeatureGroup.BAT_SPEED,
        FeatureType.COUNT,
        "Tracked swing count",
    ),
    FeatureMeta(
        "bat_tracking_bbe",
        FeatureGroup.BAT_SPEED,
        FeatureType.COUNT,
        "Tracked batted-ball-event count",
    ),
    FeatureMeta(
        "bat_tracking_blasts", FeatureGroup.BAT_SPEED, FeatureType.COUNT, "Blast count"
    ),
    FeatureMeta(
        "bat_tracking_squared_up",
        FeatureGroup.BAT_SPEED,
        FeatureType.COUNT,
        "Squared-up count",
    ),
    FeatureMeta(
        "bat_tracking_fast_swings",
        FeatureGroup.BAT_SPEED,
        FeatureType.COUNT,
        "Fast swing count",
    ),
    FeatureMeta(
        "has_avg_bat_speed",
        FeatureGroup.BAT_SPEED,
        FeatureType.INDICATOR,
        "Bat speed observed (1) or imputed (0)",
    ),
    FeatureMeta(
        "has_avg_swing_speed",
        FeatureGroup.BAT_SPEED,
        FeatureType.INDICATOR,
        "Swing speed observed (1) or imputed (0)",
    ),
    FeatureMeta(
        "has_squared_up_rate",
        FeatureGroup.BAT_SPEED,
        FeatureType.INDICATOR,
        "Squared-up rate observed (1) or imputed (0)",
    ),
    FeatureMeta(
        "has_blast_rate",
        FeatureGroup.BAT_SPEED,
        FeatureType.INDICATOR,
        "Blast rate observed (1) or imputed (0)",
    ),
    FeatureMeta(
        "has_fast_swing_rate",
        FeatureGroup.BAT_SPEED,
        FeatureType.INDICATOR,
        "Fast swing rate observed (1) or imputed (0)",
    ),
    FeatureMeta(
        "has_bat_tracking_swings",
        FeatureGroup.BAT_SPEED,
        FeatureType.INDICATOR,
        "Tracked swings observed (1) or imputed (0)",
    ),
    FeatureMeta(
        "has_bat_tracking_bbe",
        FeatureGroup.BAT_SPEED,
        FeatureType.INDICATOR,
        "Tracked BBE observed (1) or imputed (0)",
    ),
    FeatureMeta(
        "has_bat_tracking_blasts",
        FeatureGroup.BAT_SPEED,
        FeatureType.INDICATOR,
        "Blast count observed (1) or imputed (0)",
    ),
    FeatureMeta(
        "has_bat_tracking_squared_up",
        FeatureGroup.BAT_SPEED,
        FeatureType.INDICATOR,
        "Squared-up count observed (1) or imputed (0)",
    ),
    FeatureMeta(
        "has_bat_tracking_fast_swings",
        FeatureGroup.BAT_SPEED,
        FeatureType.INDICATOR,
        "Fast swing count observed (1) or imputed (0)",
    ),
]

# ---------------------------------------------------------------------------
# Context features (Section 4.4)
# ---------------------------------------------------------------------------
AGE_FEATURES: list[FeatureMeta] = [
    FeatureMeta(
        "age", FeatureGroup.AGE, FeatureType.CONTINUOUS, "Player age at season midpoint"
    ),
    FeatureMeta(
        "age_squared",
        FeatureGroup.AGE,
        FeatureType.CONTINUOUS,
        "Age squared (aging curve)",
    ),
    FeatureMeta(
        "age_delta_speed",
        FeatureGroup.AGE,
        FeatureType.CONTINUOUS,
        "Expected speed change from peak (piecewise, peak ~23)",
    ),
    FeatureMeta(
        "age_delta_power",
        FeatureGroup.AGE,
        FeatureType.CONTINUOUS,
        "Expected power change from peak (piecewise, peak ~27)",
    ),
    FeatureMeta(
        "age_delta_patience",
        FeatureGroup.AGE,
        FeatureType.CONTINUOUS,
        "Expected plate discipline change from peak (piecewise, peak ~30)",
    ),
]

PARK_FACTOR_FEATURES: list[FeatureMeta] = [
    FeatureMeta(
        "park_factor_runs",
        FeatureGroup.PARK_FACTORS,
        FeatureType.CONTINUOUS,
        "Ballpark factor for runs",
    ),
    FeatureMeta(
        "park_factor_hr",
        FeatureGroup.PARK_FACTORS,
        FeatureType.CONTINUOUS,
        "Ballpark factor for home runs",
    ),
]

TEAM_STAT_FEATURES: list[FeatureMeta] = [
    FeatureMeta(
        "team_runs_per_game",
        FeatureGroup.TEAM_STATS,
        FeatureType.CONTINUOUS,
        "Team runs per game",
    ),
    FeatureMeta(
        "team_ops",
        FeatureGroup.TEAM_STATS,
        FeatureType.CONTINUOUS,
        "Team OPS (lineup protection)",
    ),
    FeatureMeta(
        "team_sb", FeatureGroup.TEAM_STATS, FeatureType.COUNT, "Team stolen bases"
    ),
    FeatureMeta(
        "sb_rule_era",
        FeatureGroup.TEAM_STATS,
        FeatureType.INDICATOR,
        "SB rule era (1 if season >= 2023)",
    ),
    FeatureMeta(
        "sb_era_x_speed",
        FeatureGroup.TEAM_STATS,
        FeatureType.CONTINUOUS,
        "SB rule era x sprint speed interaction",
    ),
    FeatureMeta(
        "speed_age_interaction",
        FeatureGroup.TEAM_STATS,
        FeatureType.CONTINUOUS,
        "Sprint speed x (age - 27) interaction",
    ),
    FeatureMeta(
        "team_sb_per_game",
        FeatureGroup.TEAM_STATS,
        FeatureType.CONTINUOUS,
        "Team stolen bases per game",
    ),
    FeatureMeta(
        "sb_era_x_attempt_rate",
        FeatureGroup.TEAM_STATS,
        FeatureType.CONTINUOUS,
        "SB rule era x SB attempt rate interaction",
    ),
]

# ---------------------------------------------------------------------------
# Temporal features (Section 4.5)
# Generated dynamically for each target stat: OBP, SLG, HR, R, RBI, SB
# ---------------------------------------------------------------------------
TARGET_STATS: list[str] = ["obp", "slg", "hr", "r", "rbi", "sb"]
RATE_DECOMP_STATS: list[str] = [
    "hr_per_pa",
    "r_per_pa",
    "rbi_per_pa",
    "sb_per_pa",
    "sb_attempt_rate",
    "iso",
]
EXPECTED_STAT_TEMPORAL_STATS: list[str] = [
    "estimated_woba_using_speedangle",
    "estimated_ba_using_speedangle",
    "estimated_slg_using_speedangle",
]


def _build_temporal_features() -> list[FeatureMeta]:
    """Generate temporal feature metadata for target stats and contact stats."""
    features: list[FeatureMeta] = []

    # Temporal features for 6 target stats
    for stat in TARGET_STATS:
        ftype = FeatureType.RATE if stat in ("obp", "slg") else FeatureType.COUNT
        features.append(
            FeatureMeta(
                f"prev_year_{stat}",
                FeatureGroup.TEMPORAL,
                ftype,
                f"Prior year {stat.upper()} value",
            )
        )
        features.append(
            FeatureMeta(
                f"weighted_avg_{stat}",
                FeatureGroup.TEMPORAL,
                ftype,
                f"Weighted average of {stat.upper()} over Y-1, Y-2, Y-3",
            )
        )
        features.append(
            FeatureMeta(
                f"trend_{stat}",
                FeatureGroup.TEMPORAL,
                FeatureType.CONTINUOUS,
                f"Trend in {stat.upper()} (Y-1 minus Y-2)",
            )
        )

    # Temporal features for expected Statcast stats (xwOBA, xBA, xSLG)
    for stat in EXPECTED_STAT_TEMPORAL_STATS:
        features.append(
            FeatureMeta(
                f"prev_year_{stat}",
                FeatureGroup.TEMPORAL,
                FeatureType.RATE,
                f"Prior year {stat} value",
            )
        )
        features.append(
            FeatureMeta(
                f"weighted_avg_{stat}",
                FeatureGroup.TEMPORAL,
                FeatureType.RATE,
                f"Weighted average of {stat} over Y-1, Y-2, Y-3",
            )
        )
        features.append(
            FeatureMeta(
                f"trend_{stat}",
                FeatureGroup.TEMPORAL,
                FeatureType.CONTINUOUS,
                f"Trend in {stat} (Y-1 minus Y-2)",
            )
        )

    # Temporal features for per-PA rate decomposition stats
    for stat in RATE_DECOMP_STATS:
        features.append(
            FeatureMeta(
                f"prev_year_{stat}",
                FeatureGroup.TEMPORAL,
                FeatureType.RATE,
                f"Prior year {stat} value",
            )
        )
        features.append(
            FeatureMeta(
                f"weighted_avg_{stat}",
                FeatureGroup.TEMPORAL,
                FeatureType.RATE,
                f"Weighted average of {stat} over Y-1, Y-2, Y-3",
            )
        )
        features.append(
            FeatureMeta(
                f"trend_{stat}",
                FeatureGroup.TEMPORAL,
                FeatureType.CONTINUOUS,
                f"Trend in {stat} (Y-1 minus Y-2)",
            )
        )

    return features


TEMPORAL_FEATURES: list[FeatureMeta] = _build_temporal_features()

# ---------------------------------------------------------------------------
# In-season features (Phase 2: weekly-snapshot inputs for ROS MTL)
# ---------------------------------------------------------------------------
# Consumed by ``src.features.in_season.compute_in_season_features`` which
# reads weekly snapshot rows (see ``src.data.build_snapshots``) and returns
# a DataFrame with these 24 columns alongside the preseason feature matrix.
# The preseason MTL in ``src/models/mtl/`` never sees these features.
IN_SEASON_FEATURES: list[FeatureMeta] = [
    # YTD passthroughs (10) — already present on the weekly snapshot row.
    FeatureMeta(
        "pa_ytd",
        FeatureGroup.IN_SEASON,
        FeatureType.COUNT,
        "Plate appearances to date this season",
    ),
    FeatureMeta(
        "obp_ytd",
        FeatureGroup.IN_SEASON,
        FeatureType.RATE,
        "On-base percentage to date this season",
    ),
    FeatureMeta(
        "slg_ytd",
        FeatureGroup.IN_SEASON,
        FeatureType.RATE,
        "Slugging percentage to date this season",
    ),
    FeatureMeta(
        "hr_per_pa_ytd",
        FeatureGroup.IN_SEASON,
        FeatureType.RATE,
        "Home runs per PA to date this season",
    ),
    FeatureMeta(
        "r_per_pa_ytd",
        FeatureGroup.IN_SEASON,
        FeatureType.RATE,
        "Runs per PA to date this season",
    ),
    FeatureMeta(
        "rbi_per_pa_ytd",
        FeatureGroup.IN_SEASON,
        FeatureType.RATE,
        "RBI per PA to date this season",
    ),
    FeatureMeta(
        "sb_per_pa_ytd",
        FeatureGroup.IN_SEASON,
        FeatureType.RATE,
        "Stolen bases per PA to date this season",
    ),
    FeatureMeta(
        "iso_ytd",
        FeatureGroup.IN_SEASON,
        FeatureType.RATE,
        "Isolated power to date this season",
    ),
    FeatureMeta(
        "bb_rate_ytd",
        FeatureGroup.IN_SEASON,
        FeatureType.RATE,
        "Walk rate to date this season",
    ),
    FeatureMeta(
        "k_rate_ytd",
        FeatureGroup.IN_SEASON,
        FeatureType.RATE,
        "Strikeout rate to date this season",
    ),
    # Trailing-4-week rates (10) — derived from trail4w_* count columns when
    # rate columns aren't pre-computed on the snapshot.
    FeatureMeta(
        "trail4w_pa",
        FeatureGroup.IN_SEASON,
        FeatureType.COUNT,
        "Plate appearances in trailing 4 ISO weeks",
    ),
    FeatureMeta(
        "trail4w_obp",
        FeatureGroup.IN_SEASON,
        FeatureType.RATE,
        "On-base percentage over trailing 4 ISO weeks",
    ),
    FeatureMeta(
        "trail4w_slg",
        FeatureGroup.IN_SEASON,
        FeatureType.RATE,
        "Slugging percentage over trailing 4 ISO weeks",
    ),
    FeatureMeta(
        "trail4w_hr_per_pa",
        FeatureGroup.IN_SEASON,
        FeatureType.RATE,
        "Home runs per PA over trailing 4 ISO weeks",
    ),
    FeatureMeta(
        "trail4w_r_per_pa",
        FeatureGroup.IN_SEASON,
        FeatureType.RATE,
        "Runs per PA over trailing 4 ISO weeks",
    ),
    FeatureMeta(
        "trail4w_rbi_per_pa",
        FeatureGroup.IN_SEASON,
        FeatureType.RATE,
        "RBI per PA over trailing 4 ISO weeks",
    ),
    FeatureMeta(
        "trail4w_sb_per_pa",
        FeatureGroup.IN_SEASON,
        FeatureType.RATE,
        "Stolen bases per PA over trailing 4 ISO weeks",
    ),
    FeatureMeta(
        "trail4w_iso",
        FeatureGroup.IN_SEASON,
        FeatureType.RATE,
        "Isolated power over trailing 4 ISO weeks",
    ),
    FeatureMeta(
        "trail4w_bb_rate",
        FeatureGroup.IN_SEASON,
        FeatureType.RATE,
        "Walk rate over trailing 4 ISO weeks",
    ),
    FeatureMeta(
        "trail4w_k_rate",
        FeatureGroup.IN_SEASON,
        FeatureType.RATE,
        "Strikeout rate over trailing 4 ISO weeks",
    ),
    # Derived timing (2) — computed from (mlbam_id, season, iso_week) and pa_ytd.
    FeatureMeta(
        "week_index",
        FeatureGroup.IN_SEASON,
        FeatureType.CONTINUOUS,
        "0-indexed ISO week within the (player, season)",
    ),
    FeatureMeta(
        "pa_fraction",
        FeatureGroup.IN_SEASON,
        FeatureType.CONTINUOUS,
        "pa_ytd divided by a constant expected full-season PA (650)",
    ),
    # IL stubs (2) — constants until an external IL feed is wired in.
    FeatureMeta(
        "days_on_il_ytd",
        FeatureGroup.IN_SEASON,
        FeatureType.COUNT,
        "Days on the injured list to date (stub: always 0 until IL source wired)",
    ),
    FeatureMeta(
        "has_il_data",
        FeatureGroup.IN_SEASON,
        FeatureType.INDICATOR,
        "Whether real IL data is available (stub: always 0 until IL source wired)",
    ),
]

# ---------------------------------------------------------------------------
# Combined registry
# ---------------------------------------------------------------------------
ALL_FEATURES: list[FeatureMeta] = (
    BATTING_FEATURES
    + STATCAST_FEATURES
    + NON_CONTACT_FEATURES
    + SPRINT_SPEED_FEATURES
    + BAT_SPEED_FEATURES
    + AGE_FEATURES
    + PARK_FACTOR_FEATURES
    + TEAM_STAT_FEATURES
    + TEMPORAL_FEATURES
    + IN_SEASON_FEATURES
)

# Target columns (fixed order — do not reorder)
TARGET_COLUMNS: list[str] = [f"target_{s}" for s in TARGET_STATS]
TARGET_DISPLAY: list[str] = [s.upper() for s in TARGET_STATS]
RATE_STATS: frozenset[str] = frozenset({"obp", "slg"})
COUNT_STATS: frozenset[str] = frozenset({"hr", "r", "rbi", "sb"})

# Identity / metadata columns (not features, not targets)
ID_COLUMNS: list[str] = ["mlbam_id", "idfg", "name", "team", "season"]

# Group → feature-list mapping for config-driven selection
_GROUP_MAP: dict[FeatureGroup, list[FeatureMeta]] = {
    FeatureGroup.BATTING: BATTING_FEATURES,
    FeatureGroup.STATCAST: STATCAST_FEATURES,
    FeatureGroup.NON_CONTACT: NON_CONTACT_FEATURES,
    FeatureGroup.SPRINT_SPEED: SPRINT_SPEED_FEATURES,
    FeatureGroup.BAT_SPEED: BAT_SPEED_FEATURES,
    FeatureGroup.AGE: AGE_FEATURES,
    FeatureGroup.PARK_FACTORS: PARK_FACTOR_FEATURES,
    FeatureGroup.TEAM_STATS: TEAM_STAT_FEATURES,
    FeatureGroup.TEMPORAL: TEMPORAL_FEATURES,
    FeatureGroup.IN_SEASON: IN_SEASON_FEATURES,
}

# Groups that must be opted into explicitly. Leaving these off by default
# ensures the preseason MTL (whose ``configs/data.yaml`` never mentions
# ``in_season``) keeps its pre-Phase-2 feature set unchanged.
_OPT_IN_GROUPS: frozenset[FeatureGroup] = frozenset({FeatureGroup.IN_SEASON})


def get_feature_names(
    enabled_groups: dict[str, bool] | None = None,
) -> list[str]:
    """Return ordered list of feature column names for enabled groups.

    Parameters
    ----------
    enabled_groups:
        Mapping of group name → enabled flag (from ``configs/data.yaml``
        ``feature_groups`` section). If None, all default-on groups are
        enabled; opt-in groups like ``IN_SEASON`` remain off until set
        explicitly.

    Returns
    -------
    list[str]
        Feature column names in registry order.
    """
    names: list[str] = []
    for group, features in _GROUP_MAP.items():
        default = group not in _OPT_IN_GROUPS
        enabled = (
            default
            if enabled_groups is None
            else enabled_groups.get(group.value, default)
        )
        if enabled:
            names.extend(f.name for f in features)
    return names


def get_feature_metadata(name: str) -> FeatureMeta | None:
    """Look up metadata for a single feature by name."""
    for f in ALL_FEATURES:
        if f.name == name:
            return f
    return None
