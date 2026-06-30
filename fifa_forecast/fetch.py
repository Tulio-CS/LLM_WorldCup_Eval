"""Automatic fixtures & results fetching from a sports data API.

Default source: **football-data.org** (free tier includes the FIFA World Cup,
competition code ``WC``). Set ``FOOTBALL_DATA_API_KEY`` in ``.env`` (free signup
at https://www.football-data.org/), then::

    python -m fifa_forecast fetch --results            # ingest finished scores
    python -m fifa_forecast fetch --fixtures           # append new games to the CSV
    python -m fifa_forecast fetch --results --fixtures --dry-run   # preview only

The hard part is identifying which API match is which local match: the API uses
its own team-name spellings (e.g. "Korea Republic", "Côte d'Ivoire", "USA"), so
names are accent-/punctuation-normalized and run through an alias table, then
matched as an unordered team pair (disambiguated by date if a pair repeats).
Unmatched games are reported, never guessed.
"""

from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .matches import Match, load_matches
from .results import MatchResult, derive_winner


class FetchError(RuntimeError):
    pass


# Canonical-normalized aliases: variant spellings -> the spelling used locally.
# Keys and values are compared after _norm() (lowercase, de-accented, alnum-only).
_ALIASES: dict[str, str] = {
    "korearepublic": "southkorea",
    "republicofkorea": "southkorea",
    "korearep": "southkorea",
    "czechrepublic": "czechia",
    "usa": "unitedstates",
    "unitedstatesofamerica": "unitedstates",
    "cotedivoire": "ivorycoast",
    "turkiye": "turkey",
    "congodr": "drcongo",
    "democraticrepublicofcongo": "drcongo",
    "drcongo": "drcongo",
    "caboverde": "capeverde",
    "bosniaherzegovina": "bosniaandherzegovina",
    "iranislamicrepublic": "iran",
    "iririran": "iran",
}

_STAGE_LABELS = {
    "GROUP_STAGE": "Group",
    "LAST_32": "Round of 32",
    "LAST_16": "Round of 16",
    "QUARTER_FINALS": "Quarter-final",
    "QUARTER_FINAL": "Quarter-final",
    "SEMI_FINALS": "Semi-final",
    "SEMI_FINAL": "Semi-final",
    "THIRD_PLACE": "Third place",
    "FINAL": "Final",
}


def _norm(name: str | None) -> str:
    if not name:
        return ""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_name = decomposed.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", ascii_name.lower())


def _canon(name: str | None) -> str:
    n = _norm(name)
    return _ALIASES.get(n, n)


def _pair_key(a: str | None, b: str | None) -> frozenset[str]:
    return frozenset({_canon(a), _canon(b)})


def _utc_to_local_minus3(utc_iso: str | None) -> tuple[str, str]:
    """Return (utc_iso_Z, 'YYYY-MM-DD HH:MM' in UTC-3) for the CSV."""
    if not utc_iso:
        return "", ""
    s = utc_iso.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return utc_iso, ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    utc = dt.astimezone(timezone.utc)
    local = utc - timedelta(hours=3)
    return (
        utc.isoformat().replace("+00:00", "Z"),
        local.strftime("%Y-%m-%d %H:%M"),
    )


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #
class FootballDataClient:
    BASE = "https://api.football-data.org/v4"

    def __init__(self, api_key: str, timeout: int = 30) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def competition_matches(self, code: str = "WC") -> list[dict[str, Any]]:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise FetchError("The 'requests' package is required for fetch.") from exc
        url = f"{self.BASE}/competitions/{code}/matches"
        resp = requests.get(
            url, headers={"X-Auth-Token": self.api_key}, timeout=self.timeout
        )
        if resp.status_code == 403:
            raise FetchError(
                "403 from football-data.org — the API key may be invalid or the "
                f"free plan may not include competition {code!r}."
            )
        if resp.status_code == 429:
            raise FetchError("429 rate-limited by football-data.org — wait and retry.")
        if resp.status_code != 200:
            raise FetchError(f"football-data.org returned HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json().get("matches", [])


def normalize_api_matches(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten football-data match objects into a simple, source-agnostic shape."""
    out: list[dict[str, Any]] = []
    for m in raw:
        home = (m.get("homeTeam") or {}).get("name")
        away = (m.get("awayTeam") or {}).get("name")
        score = m.get("score") or {}
        full = score.get("fullTime") or {}
        utc_iso, local = _utc_to_local_minus3(m.get("utcDate"))
        out.append(
            {
                "home": home,
                "away": away,
                "status": (m.get("status") or "").upper(),
                "stage": m.get("stage"),
                "stage_label": _STAGE_LABELS.get(m.get("stage") or "", (m.get("stage") or "").title()),
                "group": m.get("group"),
                "home_score": full.get("home"),
                "away_score": full.get("away"),
                "winner_side": score.get("winner"),  # HOME_TEAM | AWAY_TEAM | DRAW
                "utc_date": utc_iso,
                "local_date": local,
                "date": (utc_iso or "")[:10],
            }
        )
    return out


def _local_index(matches: list[Match]) -> dict[frozenset[str], list[Match]]:
    idx: dict[frozenset[str], list[Match]] = {}
    for m in matches:
        idx.setdefault(_pair_key(m.team_1, m.team_2), []).append(m)
    return idx


def build_results(
    api_matches: list[dict[str, Any]],
    local_matches: list[Match],
    *,
    source: str = "football-data.org",
) -> tuple[list[MatchResult], list[dict[str, Any]]]:
    """Map FINISHED API games to local match_ids. Returns (results, unmatched)."""
    idx = _local_index(local_matches)
    fetched_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    results: list[MatchResult] = []
    unmatched: list[dict[str, Any]] = []

    for a in api_matches:
        if a["status"] != "FINISHED" or a["home_score"] is None or a["away_score"] is None:
            continue
        candidates = idx.get(_pair_key(a["home"], a["away"]), [])
        if not candidates:
            unmatched.append(a)
            continue
        if len(candidates) > 1:  # disambiguate a repeated pair by date
            same_day = [c for c in candidates if c.local_date == a["date"] or (c.kickoff_datetime or "")[:10] == a["date"]]
            local = same_day[0] if same_day else candidates[0]
        else:
            local = candidates[0]

        # Orient API home/away onto the local team_1/team_2.
        if _canon(a["home"]) == _canon(local.team_1):
            s1, s2 = a["home_score"], a["away_score"]
        else:
            s1, s2 = a["away_score"], a["home_score"]

        results.append(
            MatchResult(
                match_id=local.match_id,
                phase=local.phase,
                team_1=local.team_1,
                team_2=local.team_2,
                kickoff_datetime=local.kickoff_datetime,
                status="finished",
                actual_score_team_1=int(s1),
                actual_score_team_2=int(s2),
                actual_winner=derive_winner(int(s1), int(s2)),
                went_to_penalties=None,
                penalty_winner=None,
                source=source,
                fetched_at=fetched_at,
                notes=f"group={a.get('group')}" if a.get("group") else None,
            )
        )
    return results, unmatched


def build_new_fixtures(
    api_matches: list[dict[str, Any]], local_matches: list[Match]
) -> list[dict[str, Any]]:
    """API games whose team pair is not already in the local matches, as CSV rows
    (internal-source schema: datetime_utc_minus_03, team_1, team_2, stage)."""
    have = set(_local_index(local_matches).keys())
    rows: list[dict[str, Any]] = []
    seen: set[frozenset[str]] = set()
    for a in api_matches:
        if not a["home"] or not a["away"]:
            continue  # undetermined knockout slot
        key = _pair_key(a["home"], a["away"])
        if key in have or key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "datetime_utc_minus_03": a["local_date"],
                "team_1": a["home"],
                "team_2": a["away"],
                "stage": a["stage_label"] or "",
            }
        )
    return rows


def append_fixtures_to_csv(matches_csv: str | Path, rows: list[dict[str, Any]]) -> int:
    """Append new fixture rows to the matches CSV (its own schema). Returns count."""
    import csv

    if not rows:
        return 0
    path = Path(matches_csv)
    header = ["datetime_utc_minus_03", "team_1", "team_2", "stage"]
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in header})
    return len(rows)


def fetch_world_cup(
    api_key: str, *, competition: str = "WC", timeout: int = 30
) -> list[dict[str, Any]]:
    client = FootballDataClient(api_key, timeout=timeout)
    return normalize_api_matches(client.competition_matches(competition))


def resolve_api_key(explicit: str | None = None) -> str:
    key = explicit or os.environ.get("FOOTBALL_DATA_API_KEY")
    if not key:
        raise FetchError(
            "No API key. Set FOOTBALL_DATA_API_KEY in .env (free key at "
            "https://www.football-data.org/) or pass --api-key."
        )
    return key
