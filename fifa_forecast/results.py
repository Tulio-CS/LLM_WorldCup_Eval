"""Actual match results (ground truth) ingestion.

Predictions live in ``forecast_runs``; the *actual* outcome of each match lives
in a separate ``match_results`` table so it can be joined back by ``match_id``
for evaluation later (this module does no scoring — collection only).

``actual_winner`` is stored as a canonical side label — ``team_1`` / ``team_2``
/ ``draw`` — relative to the match's canonical ``team_1``/``team_2`` (the CSV
order), independent of the prompt's presentation order. For knockouts decided on
penalties, ``went_to_penalties`` and ``penalty_winner`` carry the shoot-out.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .matches import load_matches

# Column name -> SQLite type; the dataclass below mirrors these names.
RESULT_COLUMNS: list[tuple[str, str]] = [
    ("match_id", "TEXT PRIMARY KEY"),
    ("phase", "TEXT"),
    ("team_1", "TEXT"),
    ("team_2", "TEXT"),
    ("kickoff_datetime", "TEXT"),
    ("status", "TEXT"),  # scheduled | in_progress | finished
    ("actual_score_team_1", "INTEGER"),
    ("actual_score_team_2", "INTEGER"),
    ("actual_winner", "TEXT"),  # team_1 | team_2 | draw (canonical side)
    ("went_to_penalties", "INTEGER"),
    ("penalty_winner", "TEXT"),  # team_1 | team_2 | null
    ("source", "TEXT"),
    ("fetched_at", "TEXT"),
    ("notes", "TEXT"),
]

RESULT_COLUMN_NAMES = [name for name, _ in RESULT_COLUMNS]


@dataclass
class MatchResult:
    match_id: str
    phase: str | None = None
    team_1: str | None = None
    team_2: str | None = None
    kickoff_datetime: str | None = None
    status: str | None = None
    actual_score_team_1: int | None = None
    actual_score_team_2: int | None = None
    actual_winner: str | None = None
    went_to_penalties: int | None = None
    penalty_winner: str | None = None
    source: str | None = None
    fetched_at: str | None = None
    notes: str | None = None

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


assert [f.name for f in fields(MatchResult)] == RESULT_COLUMN_NAMES, (
    "MatchResult fields and RESULT_COLUMNS are out of sync"
)


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def derive_winner(s1: int | None, s2: int | None) -> str | None:
    if s1 is None or s2 is None:
        return None
    if s1 > s2:
        return "team_1"
    if s2 > s1:
        return "team_2"
    return "draw"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_results_csv(
    csv_path: str | Path, matches_csv: str | Path | None = None
) -> tuple[list[MatchResult], list[str]]:
    """Parse a results CSV. Returns (results, warnings).

    Required column: ``match_id``. Optional: ``actual_score_team_1``,
    ``actual_score_team_2``, ``status``, ``went_to_penalties``,
    ``penalty_winner``, ``source``, ``notes``, and team names (used only for a
    sanity check). Phase / team names / kickoff are filled from the matches CSV
    when available.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")

    match_index: dict[str, Any] = {}
    if matches_csv is not None and Path(matches_csv).exists():
        match_index = {m.match_id: m for m in load_matches(matches_csv)}

    warnings: list[str] = []
    results: list[MatchResult] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "match_id" not in reader.fieldnames:
            raise ValueError("Results CSV must have a 'match_id' column.")
        for row in reader:
            match_id = (row.get("match_id") or "").strip()
            if not match_id:
                continue
            s1 = _to_int(row.get("actual_score_team_1"))
            s2 = _to_int(row.get("actual_score_team_2"))
            status = (row.get("status") or "").strip().lower()
            if not status:
                status = "finished" if (s1 is not None and s2 is not None) else "scheduled"

            m = match_index.get(match_id)
            phase = (row.get("phase") or (m.phase if m else "") or "").strip() or None
            t1 = (row.get("team_1") or (m.team_1 if m else "") or "").strip() or None
            t2 = (row.get("team_2") or (m.team_2 if m else "") or "").strip() or None
            kickoff = (
                row.get("kickoff_datetime")
                or (m.kickoff_datetime if m else "")
                or ""
            ).strip() or None

            if m is None:
                warnings.append(
                    f"match_id {match_id!r} not found in matches CSV (kept as-is)."
                )
            elif row.get("team_1") and m and row["team_1"].strip() and row["team_1"].strip() != m.team_1:
                warnings.append(
                    f"match_id {match_id}: team_1 {row['team_1']!r} != "
                    f"matches CSV {m.team_1!r}."
                )

            pen = _to_int(row.get("went_to_penalties"))
            results.append(
                MatchResult(
                    match_id=match_id,
                    phase=phase,
                    team_1=t1,
                    team_2=t2,
                    kickoff_datetime=kickoff,
                    status=status,
                    actual_score_team_1=s1,
                    actual_score_team_2=s2,
                    actual_winner=derive_winner(s1, s2),
                    went_to_penalties=pen,
                    penalty_winner=(row.get("penalty_winner") or "").strip() or None,
                    source=(row.get("source") or "manual").strip() or "manual",
                    fetched_at=_now_utc(),
                    notes=(row.get("notes") or "").strip() or None,
                )
            )
    return results, warnings


def write_results_template(
    matches_csv: str | Path, out_path: str | Path
) -> tuple[Path, int]:
    """Write a blank results CSV (one row per match) for manual entry."""
    matches = load_matches(matches_csv)
    out = Path(out_path)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "match_id",
                "team_1",
                "team_2",
                "kickoff_datetime",
                "status",
                "actual_score_team_1",
                "actual_score_team_2",
                "went_to_penalties",
                "penalty_winner",
                "notes",
            ]
        )
        for m in matches:
            writer.writerow(
                [m.match_id, m.team_1, m.team_2, m.kickoff_datetime, "scheduled", "", "", "", "", ""]
            )
    return out, len(matches)
