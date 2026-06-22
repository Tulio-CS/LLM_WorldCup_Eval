"""Response parsing and validation.

LLMs frequently wrap JSON in prose or code fences. ``parse_forecast`` extracts
the JSON object, pulls out the prediction fields, and reports validity — but it
never raises: an unparseable response yields ``json_valid=False`` with the raw
text preserved upstream, so no data is lost.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .prompts import HAT_FIELDS, PROMPTS_WITH_HATS, PROMPTS_WITH_PROBABILITIES


@dataclass
class ParsedForecast:
    json_valid: bool = False
    parsed_json: dict[str, Any] | None = None

    score_team_1: int | None = None
    score_team_2: int | None = None

    team1_win_probability: int | None = None
    draw_probability: int | None = None
    team2_win_probability: int | None = None

    hats: dict[str, str] = field(default_factory=dict)

    validation_notes: list[str] = field(default_factory=list)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a single JSON object from arbitrary text."""
    if not text:
        return None

    candidates: list[str] = []

    # 1) fenced code block
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1))

    # 2) the whole string
    candidates.append(text.strip())

    # 3) first '{' .. last '}' span
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):  # bool is a subclass of int; reject it
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        try:
            return int(s)
        except ValueError:
            try:
                f = float(s)
                return int(f) if f.is_integer() else None
            except ValueError:
                return None
    return None


def parse_forecast(raw_text: str, prompt_id: str) -> ParsedForecast:
    """Parse a model response for the fields the given prompt requested."""
    result = ParsedForecast()
    obj = _extract_json_object(raw_text or "")

    if obj is None:
        result.validation_notes.append("no JSON object found")
        return result

    result.parsed_json = obj
    result.json_valid = True

    # --- scores (all prompts) ---------------------------------------------
    result.score_team_1 = _coerce_int(obj.get("score_team_1"))
    result.score_team_2 = _coerce_int(obj.get("score_team_2"))
    if result.score_team_1 is None or result.score_team_2 is None:
        result.json_valid = False
        result.validation_notes.append("missing/invalid score field(s)")
    else:
        if result.score_team_1 < 0 or result.score_team_2 < 0:
            result.json_valid = False
            result.validation_notes.append("negative score")

    # --- probabilities -----------------------------------------------------
    if prompt_id in PROMPTS_WITH_PROBABILITIES:
        p1 = _coerce_int(obj.get("team1_win_probability"))
        pd = _coerce_int(obj.get("draw_probability"))
        p2 = _coerce_int(obj.get("team2_win_probability"))
        result.team1_win_probability = p1
        result.draw_probability = pd
        result.team2_win_probability = p2
        if None in (p1, pd, p2):
            result.json_valid = False
            result.validation_notes.append("missing/invalid probability field(s)")
        else:
            if any(not (0 <= p <= 100) for p in (p1, pd, p2)):
                result.json_valid = False
                result.validation_notes.append("probability out of [0,100]")
            if (p1 + pd + p2) != 100:
                # Recorded but not fatal — kept for calibration analysis.
                result.validation_notes.append(
                    f"probabilities sum to {p1 + pd + p2}, not 100"
                )

    # --- six hats ----------------------------------------------------------
    if prompt_id in PROMPTS_WITH_HATS:
        for hat in HAT_FIELDS:
            value = obj.get(hat)
            if isinstance(value, str) and value.strip():
                result.hats[hat] = value
            else:
                result.validation_notes.append(f"missing/empty {hat}")
        if len(result.hats) < len(HAT_FIELDS):
            result.json_valid = False

    return result
