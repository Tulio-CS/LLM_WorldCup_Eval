"""Command-line interface.

    python -m fifa_forecast run            # collect data, then export + manifest
    python -m fifa_forecast run --dry-run  # offline smoke test (mock provider)
    python -m fifa_forecast export         # rebuild the Excel workbook from the DB
    python -m fifa_forecast manifest       # (re)write experiment_manifest.json
    python -m fifa_forecast info           # print the planned execution count
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import config as cfg
from . import manifest as manifest_mod
from .export import export_workbook
from .matches import filter_matches, load_matches
from .runner import ExperimentRunner


def _progress(msg: str) -> None:
    print(msg, flush=True)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a config.json overrides file (defaults to ./config.json).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fifa_forecast",
        description="FIFA World Cup 2026 AI forecast data-collection framework.",
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Execute the forecast experiment.")
    _add_common(p_run)
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Use the deterministic mock provider (no network, no API keys).",
    )
    p_run.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run and overwrite combinations already present in the DB.",
    )
    p_run.add_argument(
        "--limit-matches",
        type=int,
        default=None,
        help="Only process the first N matches (useful for testing).",
    )
    p_run.add_argument(
        "--date",
        action="append",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only matches kicking off on this local (UTC-3) date. Repeatable.",
    )
    p_run.add_argument(
        "--match-id",
        action="append",
        default=None,
        help="Only these match ids. Repeatable.",
    )
    p_run.add_argument(
        "--no-export",
        action="store_true",
        help="Skip building the Excel workbook after the run.",
    )

    p_export = sub.add_parser("export", help="Rebuild the Excel workbook from the DB.")
    _add_common(p_export)

    p_manifest = sub.add_parser("manifest", help="Write experiment_manifest.json.")
    _add_common(p_manifest)
    p_manifest.add_argument("--dry-run", action="store_true")

    p_info = sub.add_parser("info", help="Print the planned execution count.")
    _add_common(p_info)
    p_info.add_argument(
        "--date",
        action="append",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only matches kicking off on this local (UTC-3) date. Repeatable.",
    )
    p_info.add_argument(
        "--match-id",
        action="append",
        default=None,
        help="Only these match ids. Repeatable.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "run"

    config = cfg.load_config(getattr(args, "config", None))

    if command == "info":
        matches = load_matches(cfg.ROOT / config.matches_csv)
        matches = filter_matches(
            matches, dates=args.date, match_ids=args.match_id
        )
        total = (
            len(matches)
            * len(config.team_order_types)
            * len(config.match_moments)
            * len(config.enabled_models())
            * len(config.prompt_ids)
            * config.runs_per_combination
        )
        if args.date or args.match_id:
            print(
                "Selected matches:      "
                + ", ".join(
                    f"{m.match_id}:{m.team_1} vs {m.team_2} ({m.local_date})"
                    for m in matches
                )
                or "(none)"
            )
        print(f"Matches loaded:        {len(matches)}")
        print(f"Enabled models:        {len(config.enabled_models())}")
        print(f"Prompt strategies:     {len(config.prompt_ids)}")
        print(f"Team orderings:        {len(config.team_order_types)}")
        print(f"Match moments:         {len(config.match_moments)} ({', '.join(config.match_moments)})")
        print(f"Repetitions/combo:     {config.runs_per_combination}")
        print(f"Planned executions:    {total}")
        return 0

    if command == "export":
        path = export_workbook(config)
        print(f"Workbook written: {path}")
        return 0

    if command == "manifest":
        matches = load_matches(cfg.ROOT / config.matches_csv)
        m = manifest_mod.build_manifest(
            config, dry_run=getattr(args, "dry_run", False), matches_count=len(matches)
        )
        path = manifest_mod.write_manifest(m)
        print(f"Manifest written: {path}")
        return 0

    # command == "run"
    matches = load_matches(cfg.ROOT / config.matches_csv)
    matches = filter_matches(matches, dates=args.date, match_ids=args.match_id)
    manifest = manifest_mod.build_manifest(
        config, dry_run=args.dry_run, matches_count=len(matches)
    )
    manifest_path = manifest_mod.write_manifest(manifest)
    print(f"Manifest written: {manifest_path}")

    runner = ExperimentRunner(
        config,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        progress=_progress,
    )
    try:
        stats = runner.run(
            limit_matches=args.limit_matches,
            dates=args.date,
            match_ids=args.match_id,
        )
    finally:
        runner.close()

    print(
        "Run complete: "
        f"total={stats['total']} success={stats['success']} "
        f"error={stats['error']} skipped={stats['skipped']}"
    )

    if not args.no_export:
        path = export_workbook(config)
        print(f"Workbook written: {path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
