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
from pathlib import Path
from typing import Sequence

from . import config as cfg
from . import manifest as manifest_mod
from .analysis import ReportOptions, build_report, print_report, write_report_excel
from .cost import estimate_cost, print_estimate
from .evaluation import build_evaluation, print_evaluation, write_evaluation_excel
from .export import export_workbook
from .fetch import (
    build_new_fixtures,
    build_results,
    append_fixtures_to_csv,
    fetch_world_cup,
    resolve_api_key,
)
from .matches import filter_matches, load_matches
from .results import load_results_csv, write_results_template
from .runner import ExperimentRunner
from .storage import Database


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
        "--moment",
        action="append",
        default=None,
        choices=["pre_match", "halftime", "post_match"],
        help="Forecast moment(s) to run. Repeatable. Default: config match_moments.",
    )
    p_run.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model key(s) to run, e.g. claude-opus. Repeatable. Default: all enabled.",
    )
    p_run.add_argument(
        "--reps",
        type=int,
        default=None,
        help="Override repetitions for every model (default: each model's config reps).",
    )
    p_run.add_argument(
        "--retry-errors",
        action="store_true",
        help="Re-run combinations that previously errored (plus any missing ones); "
        "successful ones are still skipped.",
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

    p_report = sub.add_parser(
        "report",
        help="Summarize counts, errors, prompt quality and repetition variability "
        "from the collected data (read-only; works on a partial run).",
    )
    _add_common(p_report)
    p_report.add_argument(
        "--out", default=None, help="Output .xlsx path (default under data/exports/)."
    )
    p_report.add_argument(
        "--prob-margin",
        type=float,
        default=5.0,
        help="Acceptable +/- on win probability, in points (default 5).",
    )
    p_report.add_argument(
        "--gd-margin",
        type=float,
        default=0.3,
        help="Acceptable +/- on goal difference, in goals (default 0.3).",
    )
    p_report.add_argument(
        "--shuffles",
        type=int,
        default=40,
        help="Permutations per combination for the convergence curve (default 40).",
    )
    p_report.add_argument(
        "--no-excel", action="store_true", help="Print to console only; skip the .xlsx."
    )

    p_eval = sub.add_parser(
        "evaluate",
        help="Compare AI forecasts to actual results for finished matches "
        "(accuracy, exact score, Brier).",
    )
    _add_common(p_eval)
    p_eval.add_argument(
        "--out", default=None, help="Output .xlsx path (default under data/exports/)."
    )
    p_eval.add_argument(
        "--no-excel", action="store_true", help="Print to console only; skip the .xlsx."
    )

    p_est = sub.add_parser(
        "estimate",
        help="Project USD cost from observed token usage for a planned run.",
    )
    _add_common(p_est)
    p_est.add_argument(
        "--matches",
        type=int,
        default=None,
        help="Number of matches (default: count in the matches CSV).",
    )
    p_est.add_argument(
        "--occasions",
        type=int,
        default=1,
        help="Forecasts per match, e.g. 3 = 1 pre-match + 2 in-play (default 1).",
    )
    p_est.add_argument(
        "--reps",
        type=int,
        default=None,
        help="Repetitions per combination (default: config runs_per_combination).",
    )
    p_est.add_argument(
        "--orders",
        type=int,
        default=None,
        help="Team orderings per match (default: config, normally 2).",
    )

    p_tmpl = sub.add_parser(
        "results-template",
        help="Write a blank results CSV (one row per match) for manual entry.",
    )
    _add_common(p_tmpl)
    p_tmpl.add_argument("--out", default="results.csv", help="Output CSV path.")

    p_addres = sub.add_parser(
        "add-results",
        help="Ingest actual match results from a CSV into the match_results table.",
    )
    _add_common(p_addres)
    p_addres.add_argument("--csv", required=True, help="Results CSV to ingest.")
    p_addres.add_argument(
        "--no-export", action="store_true", help="Skip rebuilding the Excel workbook."
    )

    p_fetch = sub.add_parser(
        "fetch",
        help="Fetch fixtures/results from a sports API (default: football-data.org).",
    )
    _add_common(p_fetch)
    p_fetch.add_argument(
        "--results", action="store_true", help="Ingest finished match scores."
    )
    p_fetch.add_argument(
        "--fixtures",
        action="store_true",
        help="Append newly-scheduled games (not yet in the CSV) to the matches CSV.",
    )
    p_fetch.add_argument("--competition", default="WC", help="Competition code (default WC).")
    p_fetch.add_argument(
        "--api-key", default=None, help="Override FOOTBALL_DATA_API_KEY from .env."
    )
    p_fetch.add_argument("--timeout", type=int, default=30)
    p_fetch.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested/appended without writing anything.",
    )
    p_fetch.add_argument(
        "--no-export", action="store_true", help="Skip rebuilding the Excel workbook."
    )

    p_merge = sub.add_parser(
        "merge",
        help="Merge forecast rows from one or more other SQLite DBs into this one "
        "(e.g. combine a machine that ran local models with the main DB).",
    )
    _add_common(p_merge)
    p_merge.add_argument(
        "--from",
        dest="from_db",
        action="append",
        required=True,
        metavar="PATH.db",
        help="Source .db file to merge in. Repeatable.",
    )
    p_merge.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite rows sharing a run_id (default: keep the existing row).",
    )
    p_merge.add_argument(
        "--no-export", action="store_true", help="Skip rebuilding the Excel workbook."
    )

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

    if command == "results-template":
        out, n = write_results_template(cfg.ROOT / config.matches_csv, args.out)
        print(f"Wrote template with {n} matches: {out}")
        print("Fill in actual_score_team_1 / actual_score_team_2, then run: "
              "python -m fifa_forecast add-results --csv " + str(args.out))
        return 0

    if command == "fetch":
        do_results = args.results or not args.fixtures  # default to results
        do_fixtures = args.fixtures
        key = resolve_api_key(args.api_key)
        matches_path = cfg.ROOT / config.matches_csv
        print(f"Fetching competition {args.competition!r} from football-data.org ...")
        api_matches = fetch_world_cup(
            key, competition=args.competition, timeout=args.timeout
        )
        print(f"  {len(api_matches)} matches returned by the API.")
        local = load_matches(matches_path)

        if do_fixtures:
            rows = build_new_fixtures(api_matches, local)
            if not rows:
                print("Fixtures: nothing new to add.")
            elif args.dry_run:
                print(f"Fixtures (dry-run) — {len(rows)} new game(s) would be appended:")
                for r in rows[:50]:
                    print(f"    {r['datetime_utc_minus_03']}  {r['team_1']} vs {r['team_2']}  [{r['stage']}]")
            else:
                n = append_fixtures_to_csv(matches_path, rows)
                print(f"Fixtures: appended {n} new game(s) to {config.matches_csv}.")
                local = load_matches(matches_path)  # reload so results can map them

        if do_results:
            results, unmatched = build_results(api_matches, local)
            print(f"Results: {len(results)} finished game(s) matched to local match_ids.")
            if unmatched:
                print(f"  {len(unmatched)} finished API game(s) could NOT be matched "
                      "(name mismatch or not in your CSV):")
                for a in unmatched[:25]:
                    print(f"    {a['home']} {a['home_score']}-{a['away_score']} {a['away']} [{a['stage']}]")
                print("  -> add an alias in fetch.py _ALIASES or the game to the matches CSV.")
            if args.dry_run:
                print("Results (dry-run): nothing written.")
            else:
                db = Database(cfg.ROOT / config.database_path)
                try:
                    for r in results:
                        db.upsert_result(r)
                    total = db.count_results()
                finally:
                    db.close()
                print(f"  ingested; match_results now has {total} rows.")

        if not args.dry_run and not args.no_export:
            path = export_workbook(config)
            print(f"Workbook written: {path}")
        return 0

    if command == "add-results":
        results, warnings = load_results_csv(args.csv, cfg.ROOT / config.matches_csv)
        db = Database(cfg.ROOT / config.database_path)
        try:
            ingested = 0
            for r in results:
                db.upsert_result(r)
                ingested += 1
            total = db.count_results()
        finally:
            db.close()
        finished = sum(1 for r in results if r.status == "finished")
        for w in warnings:
            print(f"  warning: {w}")
        print(f"Ingested {ingested} result rows ({finished} finished); "
              f"match_results now has {total} rows.")
        if not args.no_export:
            path = export_workbook(config)
            print(f"Workbook written: {path}")
        return 0

    if command == "estimate":
        matches = args.matches
        if matches is None:
            matches = len(load_matches(cfg.ROOT / config.matches_csv))
        est = estimate_cost(
            config,
            matches=matches,
            occasions=args.occasions,
            reps=args.reps,
            orders=args.orders,
        )
        print_estimate(est)
        return 0

    if command == "report":
        opts = ReportOptions(
            prob_margin=args.prob_margin,
            gd_margin=args.gd_margin,
            shuffles=args.shuffles,
        )
        report = build_report(config, opts)
        print_report(report)
        if not args.no_excel:
            path = write_report_excel(report, args.out)
            print(f"\nReport workbook written: {path}")
        return 0

    if command == "evaluate":
        ev = build_evaluation(config)
        print_evaluation(ev)
        if not args.no_excel and ev.tables:
            path = write_evaluation_excel(ev, args.out)
            print(f"\nEvaluation workbook written: {path}")
        return 0

    if command == "merge":
        db = Database(cfg.ROOT / config.database_path)
        try:
            for src in args.from_db:
                src_path = Path(src)
                if not src_path.is_absolute():
                    src_path = cfg.ROOT / src_path
                stats = db.merge_from(src_path, replace=args.replace)
                print(
                    f"Merged {src_path.name}: +{stats['runs_added']} run row(s), "
                    f"+{stats['results_added']} result row(s)."
                )
            print(
                f"Totals now: {db.count()} forecast rows, "
                f"{db.count_results()} result rows."
            )
        finally:
            db.close()
        if not args.no_export:
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
            moments=args.moment,
            model_keys=args.model,
            reps=args.reps,
            retry_errors=args.retry_errors,
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
