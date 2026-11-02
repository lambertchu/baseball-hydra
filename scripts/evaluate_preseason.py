"""Score saved preseason forecasts against actual season results.

Usage::

    python scripts/evaluate_preseason.py --config configs/preseason_eval_2026.yaml
    python scripts/evaluate_preseason.py --config ... --verify       # offline check
    python scripts/evaluate_preseason.py --config ... --report-only  # re-render text
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.preseason import ROOT, load_config, render_report, run  # noqa: E402
from src.eval.preseason_verify import verify  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", help="Override the config's output_dir")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify", action="store_true", help="Recheck saved outputs offline"
    )
    mode.add_argument(
        "--report-only", action="store_true", help="Rebuild the report from outputs"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(args.output_dir) if args.output_dir else ROOT / cfg["output_dir"]
    if args.verify:
        verify(cfg, out)
    elif args.report_only:
        (out / "evaluation_report.md").write_text(render_report(cfg, out))
    else:
        run(cfg, out)
        verify(cfg, out)


if __name__ == "__main__":
    main()
