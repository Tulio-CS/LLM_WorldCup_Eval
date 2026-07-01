"""Persistence: SQLite dataset + on-disk JSON archive.

* :class:`Database` owns ``fifa_forecasts.db`` and the ``forecast_runs`` table.
* :class:`FileArchive` writes the four per-execution artifacts
  (request / response / trace / metadata) into the ``data/`` folder tree.

Raw outputs are never overwritten: each execution has a unique ``run_id`` and
its own files, and inserts use that id as the primary key.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from . import config as cfg
from .results import RESULT_COLUMN_NAMES, RESULT_COLUMNS, MatchResult


# Column name -> SQLite type. The dataclass below mirrors these names exactly.
FORECAST_COLUMNS: list[tuple[str, str]] = [
    ("run_id", "TEXT PRIMARY KEY"),
    ("match_id", "TEXT"),
    ("phase", "TEXT"),
    ("team_1", "TEXT"),
    ("team_2", "TEXT"),
    ("kickoff_datetime", "TEXT"),
    ("team_order_type", "TEXT"),
    ("match_moment", "TEXT"),  # pre_match | halftime | post_match
    ("prompt_team_1", "TEXT"),  # team presented first in the prompt
    ("prompt_team_2", "TEXT"),  # team presented second in the prompt
    ("provider", "TEXT"),
    ("model", "TEXT"),  # model key from the config catalog
    ("model_id", "TEXT"),  # exact API model string
    ("prompt_id", "TEXT"),
    ("repetition_number", "INTEGER"),
    ("request_timestamp_utc", "TEXT"),
    ("response_timestamp_utc", "TEXT"),
    ("request_timestamp_local", "TEXT"),
    ("response_timestamp_local", "TEXT"),
    ("latency_ms", "REAL"),
    ("temperature", "REAL"),
    ("top_p", "REAL"),
    ("seed", "INTEGER"),
    ("max_tokens", "INTEGER"),
    ("reasoning_effort", "TEXT"),
    ("response_format", "TEXT"),
    ("api_params_json", "TEXT"),  # full params dict actually used
    ("system_prompt", "TEXT"),
    ("prompt_text", "TEXT"),
    ("raw_response", "TEXT"),
    ("parsed_json", "TEXT"),
    ("parsed_score_team_1", "INTEGER"),
    ("parsed_score_team_2", "INTEGER"),
    ("parsed_team1_win_probability", "INTEGER"),
    ("parsed_draw_probability", "INTEGER"),
    ("parsed_team2_win_probability", "INTEGER"),
    ("white_hat", "TEXT"),
    ("red_hat", "TEXT"),
    ("black_hat", "TEXT"),
    ("yellow_hat", "TEXT"),
    ("green_hat", "TEXT"),
    ("blue_hat", "TEXT"),
    ("response_id", "TEXT"),
    ("request_id", "TEXT"),
    ("trace_id", "TEXT"),
    ("prompt_tokens", "INTEGER"),
    ("completion_tokens", "INTEGER"),
    ("reasoning_tokens", "INTEGER"),
    ("total_tokens", "INTEGER"),
    ("api_cost", "REAL"),
    ("json_valid", "INTEGER"),
    ("validation_notes", "TEXT"),
    ("execution_status", "TEXT"),
    ("error_message", "TEXT"),
    ("attempt_count", "INTEGER"),
    ("created_at", "TEXT"),
]

_COLUMN_NAMES = [name for name, _ in FORECAST_COLUMNS]


@dataclass
class RunRecord:
    """One row of ``forecast_runs``. Every field defaults to None so error rows
    can be written with whatever information is available."""

    run_id: str
    match_id: str | None = None
    phase: str | None = None
    team_1: str | None = None
    team_2: str | None = None
    kickoff_datetime: str | None = None
    team_order_type: str | None = None
    match_moment: str | None = None
    prompt_team_1: str | None = None
    prompt_team_2: str | None = None
    provider: str | None = None
    model: str | None = None
    model_id: str | None = None
    prompt_id: str | None = None
    repetition_number: int | None = None
    request_timestamp_utc: str | None = None
    response_timestamp_utc: str | None = None
    request_timestamp_local: str | None = None
    response_timestamp_local: str | None = None
    latency_ms: float | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    response_format: str | None = None
    api_params_json: str | None = None
    system_prompt: str | None = None
    prompt_text: str | None = None
    raw_response: str | None = None
    parsed_json: str | None = None
    parsed_score_team_1: int | None = None
    parsed_score_team_2: int | None = None
    parsed_team1_win_probability: int | None = None
    parsed_draw_probability: int | None = None
    parsed_team2_win_probability: int | None = None
    white_hat: str | None = None
    red_hat: str | None = None
    black_hat: str | None = None
    yellow_hat: str | None = None
    green_hat: str | None = None
    blue_hat: str | None = None
    response_id: str | None = None
    request_id: str | None = None
    trace_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    api_cost: float | None = None
    json_valid: int | None = None
    validation_notes: str | None = None
    execution_status: str | None = None
    error_message: str | None = None
    attempt_count: int | None = None
    created_at: str | None = None


# Fail fast if the dataclass and column list ever drift apart.
assert [f.name for f in fields(RunRecord)] == _COLUMN_NAMES, (
    "RunRecord fields and FORECAST_COLUMNS are out of sync"
)


class Database:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        # Ensure the parent directory exists (e.g. a fresh volume / custom path).
        if self.db_path.parent and not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cols_sql = ",\n  ".join(f"{name} {ctype}" for name, ctype in FORECAST_COLUMNS)
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS forecast_runs (\n  {cols_sql}\n)")
        self._migrate()
        for col in ("match_id", "model", "prompt_id", "team_order_type", "execution_status"):
            self.conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_forecast_{col} "
                f"ON forecast_runs ({col})"
            )
        results_cols = ",\n  ".join(f"{name} {ctype}" for name, ctype in RESULT_COLUMNS)
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS match_results (\n  {results_cols}\n)"
        )
        self.conn.commit()

    def _migrate(self) -> None:
        """Add any columns missing from a pre-existing database.

        SQLite only supports appending columns, so new fields are added at the
        end physically; inserts use explicit column names, so position is
        irrelevant. Safe to run on every startup.
        """
        existing = {row[1] for row in self.conn.execute("PRAGMA table_info(forecast_runs)")}
        for name, ctype in FORECAST_COLUMNS:
            if name not in existing:
                # Strip "PRIMARY KEY" — cannot be added via ALTER TABLE.
                col_type = ctype.replace("PRIMARY KEY", "").strip() or "TEXT"
                self.conn.execute(
                    f"ALTER TABLE forecast_runs ADD COLUMN {name} {col_type}"
                )
        self.conn.commit()

    def insert_run(self, record: RunRecord) -> None:
        values = [getattr(record, name) for name in _COLUMN_NAMES]
        placeholders = ", ".join("?" for _ in _COLUMN_NAMES)
        columns = ", ".join(_COLUMN_NAMES)
        self.conn.execute(
            f"INSERT OR REPLACE INTO forecast_runs ({columns}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()

    def count(self, where: str = "", params: tuple = ()) -> int:
        sql = "SELECT COUNT(*) FROM forecast_runs"
        if where:
            sql += f" WHERE {where}"
        return int(self.conn.execute(sql, params).fetchone()[0])

    def status_of(self, run_id: str) -> str | None:
        """Return 'success' / 'error' for an existing run, or None if missing."""
        row = self.conn.execute(
            "SELECT execution_status FROM forecast_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return row[0] if row else None

    def fetch_all(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM forecast_runs").fetchall()
        return [dict(row) for row in rows]

    def merge_from(self, other_path: str | Path, *, replace: bool = False) -> dict[str, int]:
        """Copy rows from another forecast DB into this one.

        run_id / result primary keys are deterministic, so merging is safe and
        idempotent: by default existing rows are kept (INSERT OR IGNORE); pass
        ``replace=True`` to overwrite same-id rows (INSERT OR REPLACE). Only the
        columns present in *both* schemas are copied, so an older source DB
        missing newer columns still merges cleanly. Returns how many rows were
        added to each table.
        """
        other = Path(other_path)
        if not other.exists():
            raise FileNotFoundError(f"Source database not found: {other}")
        verb = "REPLACE" if replace else "IGNORE"
        before_runs, before_results = self.count(), self.count_results()

        self.conn.execute("ATTACH DATABASE ? AS src", (str(other),))
        try:
            def _copy(table: str, canonical: list[str]) -> None:
                exists = self.conn.execute(
                    "SELECT 1 FROM src.sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                if not exists:
                    return
                src_cols = {row[1] for row in self.conn.execute(f"PRAGMA src.table_info({table})")}
                cols = [c for c in canonical if c in src_cols]
                col_sql = ", ".join(cols)
                self.conn.execute(
                    f"INSERT OR {verb} INTO {table} ({col_sql}) "
                    f"SELECT {col_sql} FROM src.{table}"
                )

            _copy("forecast_runs", _COLUMN_NAMES)
            _copy("match_results", RESULT_COLUMN_NAMES)
            self.conn.commit()
        finally:
            self.conn.execute("DETACH DATABASE src")

        return {
            "runs_added": self.count() - before_runs,
            "results_added": self.count_results() - before_results,
            "runs_total": self.count(),
            "results_total": self.count_results(),
        }

    # -- results ------------------------------------------------------------
    def upsert_result(self, result: MatchResult) -> None:
        values = [getattr(result, name) for name in RESULT_COLUMN_NAMES]
        placeholders = ", ".join("?" for _ in RESULT_COLUMN_NAMES)
        columns = ", ".join(RESULT_COLUMN_NAMES)
        self.conn.execute(
            f"INSERT OR REPLACE INTO match_results ({columns}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()

    def count_results(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM match_results").fetchone()[0])

    def fetch_results(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM match_results").fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.conn.close()


def _json_default(obj: Any) -> Any:
    for attr in ("model_dump", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # pragma: no cover
                pass
    return str(obj)


class FileArchive:
    """Writes the four per-execution JSON artifacts to the data/ tree."""

    def __init__(self) -> None:
        for d in cfg.ALL_DATA_DIRS:
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write(path: Path, payload: Any) -> str:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        return str(path)

    def write_request(self, run_id: str, payload: Any) -> str:
        return self._write(cfg.RAW_REQUESTS_DIR / f"{run_id}.json", payload)

    def write_response(self, run_id: str, payload: Any) -> str:
        return self._write(cfg.RAW_RESPONSES_DIR / f"{run_id}.json", payload)

    def write_trace(self, run_id: str, payload: Any) -> str:
        return self._write(cfg.TRACES_DIR / f"{run_id}.json", payload)

    def write_metadata(self, run_id: str, payload: Any) -> str:
        return self._write(cfg.METADATA_DIR / f"{run_id}.json", payload)
