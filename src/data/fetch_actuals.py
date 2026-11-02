"""Fetch season batting totals used to score preseason forecasts.

Saves the raw FanGraphs leaderboard JSON (primary actuals, keyed by FanGraphs
ID with ``xMLBAMID``) and the raw MLB Stats API JSON (independent cross-check).

Usage::

    python -m src.data.fetch_actuals --year 2027
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

from src.data.fetch_projections import _HEADERS

logger = logging.getLogger(__name__)

FANGRAPHS_URL = (
    "https://www.fangraphs.com/api/leaders/major-league/data"
    "?pos=all&stats=bat&lg=all&qual=0&type=8&season={year}&season1={year}"
    "&month=0&team=0&ind=0&pageitems=2000000&pagenum=1"
)
MLB_URL = (
    "https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting"
    "&season={year}&sportIds=1&gameType=R&playerPool=ALL&limit=2000"
)


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def fetch_actuals(
    year: int, out_dir: str | Path = "data/raw/actuals", force: bool = False
) -> dict[str, Path]:
    """Download both raw season-total responses for ``year``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for source, url in [("mlb", MLB_URL), ("fangraphs", FANGRAPHS_URL)]:
        path = out_dir / f"{source}_actuals_{year}.json"
        if path.exists() and not force:
            logger.info("Skipping %s — %s exists", source, path)
        else:
            try:
                path.write_text(json.dumps(_get_json(url.format(year=year))))
            except urllib.error.HTTPError as err:
                # FanGraphs can block scripted requests; save the JSON by hand.
                raise RuntimeError(
                    f"{source} returned HTTP {err.code}. Open"
                    f" {url.format(year=year)} in a browser and save the response"
                    f" to {path}, then rerun."
                ) from err
            logger.info("Saved %s actuals → %s", source, path)
        paths[source] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--out-dir", default="data/raw/actuals")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fetch_actuals(args.year, args.out_dir, args.force)


if __name__ == "__main__":
    main()
