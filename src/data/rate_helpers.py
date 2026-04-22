"""Shared rate-calculation helpers for baseball count stats.

Keeps the OBP/SLG formula and ``num / den`` guard in one place so the
preseason snapshot builder (``src.data.build_snapshots``) and the
in-season feature module (``src.features.in_season``) cannot drift.

Both call sites previously had their own verbatim copies; this module is
the canonical source. Callers that need an extra "all components missing"
guard (e.g. the in-season feature layer) apply it post-call rather than
having the helper branch on it — keeps the formulas readable and honours
the fact that the snapshot builder intentionally returns a rate when
counts are present but partially NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Elementwise division, returning NaN where the denominator is 0, NaN, or negative.

    Parameters
    ----------
    num, den:
        Same-length numeric Series (index must be aligned; caller's
        responsibility).

    Returns
    -------
    pd.Series
        Float Series ``num / den`` with NaN wherever ``den <= 0`` or
        either operand is NaN. Never raises on division by zero.
    """
    n = num.to_numpy(dtype=float, copy=False)
    d = den.to_numpy(dtype=float, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(d > 0, n / d, np.nan)
    return pd.Series(result, index=num.index, dtype=float)


def obp_slg(
    h: pd.Series,
    bb: pd.Series,
    hbp: pd.Series,
    sf: pd.Series,
    singles: pd.Series,
    doubles: pd.Series,
    triples: pd.Series,
    hr: pd.Series,
    ab: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Return (OBP, SLG) from count components. Components may contain NaN.

    Formula:
        OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
        SLG = (1B + 2*2B + 3*3B + 4*HR) / AB

    NaN values in the count components are treated as zero for the sum,
    but :func:`safe_div` still returns NaN where the denominator collapses
    to zero (or is NaN). Callers that need "all-NaN inputs stay NaN" can
    mask the result post-call (see ``src.features.in_season``).
    """
    obp_den = ab.fillna(0) + bb.fillna(0) + hbp.fillna(0) + sf.fillna(0)
    obp = safe_div(h.fillna(0) + bb.fillna(0) + hbp.fillna(0), obp_den)
    tb = (
        singles.fillna(0)
        + 2 * doubles.fillna(0)
        + 3 * triples.fillna(0)
        + 4 * hr.fillna(0)
    )
    slg = safe_div(tb, ab)
    return obp, slg
