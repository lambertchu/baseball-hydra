"""Compute derived batting features from FanGraphs data.

Takes the merged dataset and computes rate-based features that are not
directly available in the raw batting stats: walk rate, strikeout rate,
isolated power, stolen base success rate, hit-by-pitch rate, and contact rate.

Features that already exist in the raw data (avg, obp, slg, babip, woba,
wrc_plus, pa, hr, sb, cs) are passed through as-is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_batting_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived batting features from the merged dataset.

    Parameters
    ----------
    df:
        Merged dataset with raw batting columns (pa, bb, so, slg, avg,
        sb, cs, etc.).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with derived columns added:
        ``bb_rate``, ``k_rate``, ``iso``, ``sb_rate``.
    """
    df = df.copy()

    # Walk rate: BB / PA
    if "bb" in df.columns and "pa" in df.columns:
        df["bb_rate"] = np.where(df["pa"] > 0, df["bb"] / df["pa"], 0.0)
    else:
        df["bb_rate"] = 0.0

    # Strikeout rate: SO / PA
    if "so" in df.columns and "pa" in df.columns:
        df["k_rate"] = np.where(df["pa"] > 0, df["so"] / df["pa"], 0.0)
    else:
        df["k_rate"] = 0.0

    # Isolated power: SLG - AVG
    if "slg" in df.columns and "avg" in df.columns:
        df["iso"] = df["slg"] - df["avg"]
    else:
        df["iso"] = 0.0

    # Stolen base success rate: SB / (SB + CS), guarded against division by zero
    if "sb" in df.columns and "cs" in df.columns:
        total_attempts = df["sb"] + df["cs"]
        df["sb_rate"] = np.where(total_attempts > 0, df["sb"] / total_attempts, 0.0)
    else:
        df["sb_rate"] = 0.0

    # Stolen base attempt rate: (SB + CS) / PA — willingness to run
    if "sb" in df.columns and "cs" in df.columns and "pa" in df.columns:
        df["sb_attempt_rate"] = np.where(df["pa"] > 0, (df["sb"] + df["cs"]) / df["pa"], 0.0)
    else:
        df["sb_attempt_rate"] = 0.0

    # Hit-by-pitch rate: HBP / PA
    if "hbp" in df.columns and "pa" in df.columns:
        df["hbp_rate"] = np.where(df["pa"] > 0, df["hbp"] / df["pa"], 0.0)
    else:
        df["hbp_rate"] = 0.0

    # Contact rate: fraction of PA resulting in a ball in play
    # contact_rate = 1 - k_rate - bb_rate - hbp_rate
    df["contact_rate"] = 1.0 - df["k_rate"] - df["bb_rate"] - df["hbp_rate"]
    # Clamp to [0, 1] for edge cases
    df["contact_rate"] = df["contact_rate"].clip(lower=0.0, upper=1.0)

    # Per-PA rate features for count stats (separates talent from playing time)
    if "pa" in df.columns:
        pa = df["pa"].clip(lower=1)  # avoid division by zero
        for stat in ("hr", "r", "rbi", "sb"):
            if stat in df.columns:
                df[f"{stat}_per_pa"] = df[stat] / pa
            else:
                df[f"{stat}_per_pa"] = 0.0

    # Intentional walk rate: IBB / PA — pitcher-perceived threat level
    if "ibb" in df.columns and "pa" in df.columns:
        df["ibb_rate"] = np.where(df["pa"] > 0, df["ibb"] / df["pa"], 0.0)
    else:
        df["ibb_rate"] = 0.0

    # Unintentional walk rate: (BB - IBB) / PA — true plate discipline
    if "bb" in df.columns and "ibb" in df.columns and "pa" in df.columns:
        df["ubb_rate"] = np.where(df["pa"] > 0, (df["bb"] - df["ibb"]).clip(lower=0) / df["pa"], 0.0)
    else:
        df["ubb_rate"] = 0.0

    # Singles rate: (H - 2B - 3B - HR) / PA — purest OBP signal
    if all(c in df.columns for c in ("h", "doubles", "triples", "hr", "pa")):
        singles = (df["h"] - df["doubles"] - df["triples"] - df["hr"]).clip(lower=0)
        df["singles_rate"] = np.where(df["pa"] > 0, singles / df["pa"], 0.0)
    else:
        df["singles_rate"] = 0.0

    # Doubles rate: 2B / PA — gap power signal
    if "doubles" in df.columns and "pa" in df.columns:
        df["doubles_rate"] = np.where(df["pa"] > 0, df["doubles"] / df["pa"], 0.0)
    else:
        df["doubles_rate"] = 0.0

    # Triples rate: 3B / PA — speed-power combination
    if "triples" in df.columns and "pa" in df.columns:
        df["triples_rate"] = np.where(df["pa"] > 0, df["triples"] / df["pa"], 0.0)
    else:
        df["triples_rate"] = 0.0

    # Extra-base hit rate: (2B + 3B + HR) / PA
    if all(c in df.columns for c in ("doubles", "triples", "hr", "pa")):
        df["extra_base_rate"] = np.where(
            df["pa"] > 0, (df["doubles"] + df["triples"] + df["hr"]) / df["pa"], 0.0
        )
    else:
        df["extra_base_rate"] = 0.0

    # Caught stealing rate: CS / PA — risk tolerance signal
    if "cs" in df.columns and "pa" in df.columns:
        df["cs_rate"] = np.where(df["pa"] > 0, df["cs"] / df["pa"], 0.0)
    else:
        df["cs_rate"] = 0.0

    return df
