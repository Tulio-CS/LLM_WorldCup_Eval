"""Match ingestion.

The benchmark's internal match schema is::

    match_id, phase, team_1, team_2, kickoff_datetime (UTC, ISO-8601)

The project ships a real fixtures file with a different shape::

    datetime_utc_minus_03, team_1, team_2, stage

so the loader auto-detects and adapts it: ``stage`` -> ``phase``, the UTC-3
timestamp -> UTC ISO-8601, and a stable ``match_id`` is generated from row
order. A file that already matches the internal schema is used as-is.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Match:
    match_id: str
    phase: str
    team_1: str
    team_2: str
    kickoff_datetime: str  # ISO-8601 UTC, or "" if unknown
    kickoff_local: str = ""  # original local (UTC-3) timestamp from the CSV

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def local_date(self) -> str:
        """Calendar date (YYYY-MM-DD) as shown in the source CSV (UTC-3),
        falling back to the UTC date when no local timestamp is available."""
        src = self.kickoff_local or self.kickoff_datetime
        return src[:10] if src else ""


def _parse_utc_minus_3(value: str) -> str:
    """Convert a 'YYYY-MM-DD HH:MM' UTC-3 timestamp to ISO-8601 UTC."""
    value = (value or "").strip()
    if not value:
        return ""
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            naive = datetime.strptime(value, fmt)
            break
        except ValueError:
            continue
    else:  # could not parse — keep the raw value rather than discard it
        return value
    # The source is UTC-3 (Brazil time); shift +3h to reach UTC.
    utc_dt = (naive + timedelta(hours=3)).replace(tzinfo=timezone.utc)
    return utc_dt.isoformat().replace("+00:00", "Z")


def load_matches(csv_path: str | Path) -> list[Match]:
    """Load matches from ``csv_path``, adapting either supported schema."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Matches CSV not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        rows = list(reader)

    matches: list[Match] = []
    has_internal = {"team_1", "team_2"} <= fields and (
        "phase" in fields or "stage" in fields
    )
    if not has_internal:
        raise ValueError(
            f"Unrecognised matches CSV columns: {sorted(fields)}. "
            "Expected at least team_1, team_2 and phase/stage."
        )

    for idx, row in enumerate(rows, start=1):
        team_1 = (row.get("team_1") or "").strip()
        team_2 = (row.get("team_2") or "").strip()
        if not team_1 or not team_2:
            continue  # skip blank lines

        phase = (row.get("phase") or row.get("stage") or "").strip()

        match_id = (row.get("match_id") or "").strip() or str(idx)

        local_raw = (row.get("datetime_utc_minus_03") or "").strip()
        kickoff = (row.get("kickoff_datetime") or "").strip()
        if not kickoff and local_raw:
            kickoff = _parse_utc_minus_3(local_raw)

        matches.append(
            Match(
                match_id=match_id,
                phase=phase,
                team_1=team_1,
                team_2=team_2,
                kickoff_datetime=kickoff,
                kickoff_local=local_raw or kickoff,
            )
        )
    return matches


def filter_matches(
    matches: list[Match],
    *,
    dates: list[str] | None = None,
    match_ids: list[str] | None = None,
) -> list[Match]:
    """Restrict to matches on the given local (UTC-3) dates and/or match ids.

    ``dates`` are compared against :pyattr:`Match.local_date` (the CSV date),
    so "2026-06-22" selects exactly the fixtures shown for that day.
    """
    result = matches
    if match_ids:
        wanted = {str(m) for m in match_ids}
        result = [m for m in result if m.match_id in wanted]
    if dates:
        wanted_dates = {str(d).strip() for d in dates}
        result = [m for m in result if m.local_date in wanted_dates]
    return result


def ordered_pair(match: Match, team_order_type: str) -> tuple[str, str]:
    """Return (first_team, second_team) for the given ordering.

    ``original`` keeps the CSV order; ``reversed`` swaps them. The returned pair
    is what is substituted into the prompt, so the model's ``score_team_1``
    always refers to ``first_team``.
    """
    if team_order_type == "reversed":
        return match.team_2, match.team_1
    return match.team_1, match.team_2
