"""FastAPI web server for the FIFA WC 2026 LLM Eval dashboard."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import sqlite3
import threading
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from fifa_forecast import config as cfg
from fifa_forecast.config import load_config
from fifa_forecast.matches import filter_matches, load_matches
from fifa_forecast.parsing import parse_forecast
from fifa_forecast.prompts import (
    MATCH_MOMENTS,
    MOMENT_OFFSET_MINUTES,
    PROMPT_TEMPLATES,
)
from fifa_forecast.runner import ExperimentRunner, make_run_id
from fifa_forecast.storage import Database, FileArchive, RunRecord

app = FastAPI(title="FIFA WC 2026 LLM Eval")

_active_tasks: dict[str, queue.Queue] = {}


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(
        (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8")
    )


# ── Config ────────────────────────────────────────────────────────────────────

@app.get("/api/models")
def get_models():
    config = load_config()
    return [
        {
            "key": m["key"],
            "model_id": m["model_id"],
            "provider": m["provider"],
            "enabled": m.get("enabled", True),
        }
        for m in config.models
    ]


@app.get("/api/matches")
def get_matches():
    config = load_config()
    matches = load_matches(cfg.ROOT / config.matches_csv)
    return [
        {
            "match_id": m.match_id,
            "team_1": m.team_1,
            "team_2": m.team_2,
            "phase": m.phase,
            "kickoff_local": m.kickoff_local,
            "date": m.local_date,
        }
        for m in matches
    ]


@app.get("/api/prompts")
def get_prompts():
    return list(PROMPT_TEMPLATES.keys())


@app.get("/api/moments")
def get_moments():
    return list(MATCH_MOMENTS)


# ── Results ───────────────────────────────────────────────────────────────────

@app.get("/api/runs")
def list_runs(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    model: Optional[str] = None,
    status: Optional[str] = None,
    match_id: Optional[str] = None,
    prompt_id: Optional[str] = None,
    moment: Optional[str] = None,
):
    db_path = cfg.ROOT / load_config().database_path
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        where, params = [], []
        if model:
            where.append("model = ?"); params.append(model)
        if status:
            where.append("execution_status = ?"); params.append(status)
        if match_id:
            where.append("match_id = ?"); params.append(match_id)
        if prompt_id:
            where.append("prompt_id = ?"); params.append(prompt_id)
        if moment:
            where.append("match_moment = ?"); params.append(moment)

        wc = f"WHERE {' AND '.join(where)}" if where else ""
        rows = conn.execute(
            f"""SELECT run_id, match_id, team_1, team_2, phase,
                       model, model_id, prompt_id, team_order_type, match_moment,
                       repetition_number,
                       execution_status, error_message,
                       parsed_score_team_1, parsed_score_team_2,
                       parsed_team1_win_probability, parsed_draw_probability,
                       parsed_team2_win_probability,
                       json_valid, latency_ms, total_tokens, api_cost,
                       request_timestamp_utc, created_at
                FROM forecast_runs {wc}
                ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()
        total = conn.execute(
            f"SELECT COUNT(*) FROM forecast_runs {wc}", params
        ).fetchone()[0]
        return {"total": total, "rows": [_canonicalize_run(dict(r)) for r in rows]}
    finally:
        conn.close()


@app.get("/api/stats")
def get_stats():
    db_path = cfg.ROOT / load_config().database_path
    conn = sqlite3.connect(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM forecast_runs").fetchone()[0]
        success = conn.execute(
            "SELECT COUNT(*) FROM forecast_runs WHERE execution_status='success'"
        ).fetchone()[0]
        errors = conn.execute(
            "SELECT COUNT(*) FROM forecast_runs WHERE execution_status='error'"
        ).fetchone()[0]
        by_model = conn.execute(
            """SELECT model, COUNT(*) as cnt,
                      SUM(CASE WHEN execution_status='success' THEN 1 ELSE 0 END) as ok
               FROM forecast_runs GROUP BY model ORDER BY cnt DESC"""
        ).fetchall()
        return {
            "total": total,
            "success": success,
            "errors": errors,
            "by_model": [{"model": r[0], "total": r[1], "success": r[2]} for r in by_model],
        }
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Canonical (match-order) orientation.
#
# The stored ``parsed_*`` scores/probabilities are in the order the teams were
# *presented* to the model. For ``team_order_type='reversed'`` runs that order is
# swapped relative to the canonical ``team_1`` vs ``team_2`` of the fixture, so a
# reversed "1-2" actually means "2-1" for the match as listed. These SQL snippets
# undo the swap so every aggregation/comparison is in canonical team order.
# (Total goals, draw probability and max-confidence are order-independent.)
# --------------------------------------------------------------------------- #
_C1 = ("CASE WHEN team_order_type='reversed' "
       "THEN parsed_score_team_2 ELSE parsed_score_team_1 END")
_C2 = ("CASE WHEN team_order_type='reversed' "
       "THEN parsed_score_team_1 ELSE parsed_score_team_2 END")
_P1 = ("CASE WHEN team_order_type='reversed' "
       "THEN parsed_team2_win_probability ELSE parsed_team1_win_probability END")
_P2 = ("CASE WHEN team_order_type='reversed' "
       "THEN parsed_team1_win_probability ELSE parsed_team2_win_probability END")


def _canonicalize_run(d: dict) -> dict:
    """Add canonical-order score/probability fields to a run row dict."""
    reversed_ = d.get("team_order_type") == "reversed"
    s1, s2 = d.get("parsed_score_team_1"), d.get("parsed_score_team_2")
    p1, p2 = d.get("parsed_team1_win_probability"), d.get("parsed_team2_win_probability")
    d["canon_score_team_1"] = s2 if reversed_ else s1
    d["canon_score_team_2"] = s1 if reversed_ else s2
    d["canon_team1_win_probability"] = p2 if reversed_ else p1
    d["canon_team2_win_probability"] = p1 if reversed_ else p2
    d["canon_draw_probability"] = d.get("parsed_draw_probability")
    return d


@app.get("/api/dashboard")
def get_dashboard():
    """Aggregated statistics — execution health + model prediction behaviour."""
    db_path = cfg.ROOT / load_config().database_path
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        def one(sql, params=()):
            return conn.execute(sql, params).fetchone()[0]

        total = one("SELECT COUNT(*) FROM forecast_runs")
        success = one("SELECT COUNT(*) FROM forecast_runs WHERE execution_status='success'")
        errors = one("SELECT COUNT(*) FROM forecast_runs WHERE execution_status='error'")
        manual = one("SELECT COUNT(*) FROM forecast_runs WHERE execution_status='manual'")
        valid_json = one("SELECT COUNT(*) FROM forecast_runs WHERE json_valid=1")
        total_tokens = one("SELECT COALESCE(SUM(total_tokens),0) FROM forecast_runs") or 0
        total_cost = one("SELECT COALESCE(SUM(api_cost),0) FROM forecast_runs") or 0
        avg_latency = one(
            "SELECT AVG(latency_ms) FROM forecast_runs WHERE execution_status='success'"
        )

        # Per-model health + cost/latency + confidence (avg of the top probability)
        by_model = [dict(r) for r in conn.execute(
            """SELECT model,
                      COUNT(*) AS total,
                      SUM(CASE WHEN execution_status='success' THEN 1 ELSE 0 END) AS success,
                      SUM(CASE WHEN execution_status='error' THEN 1 ELSE 0 END) AS errors,
                      SUM(CASE WHEN json_valid=1 THEN 1 ELSE 0 END) AS valid_json,
                      AVG(latency_ms) AS avg_latency,
                      AVG(total_tokens) AS avg_tokens,
                      SUM(COALESCE(api_cost,0)) AS total_cost,
                      AVG(MAX(parsed_team1_win_probability,
                              parsed_draw_probability,
                              parsed_team2_win_probability)) AS avg_confidence
               FROM forecast_runs
               GROUP BY model ORDER BY total DESC"""
        ).fetchall()]

        by_prompt = [dict(r) for r in conn.execute(
            """SELECT prompt_id,
                      COUNT(*) AS total,
                      SUM(CASE WHEN execution_status='success' THEN 1 ELSE 0 END) AS success
               FROM forecast_runs GROUP BY prompt_id ORDER BY total DESC"""
        ).fetchall()]

        by_moment = [dict(r) for r in conn.execute(
            """SELECT match_moment,
                      COUNT(*) AS total,
                      SUM(CASE WHEN execution_status='success' THEN 1 ELSE 0 END) AS success
               FROM forecast_runs GROUP BY match_moment ORDER BY total DESC"""
        ).fetchall()]

        # Prediction behaviour (only rows with a parsed scoreline) — canonical order
        pred = conn.execute(
            f"""SELECT
                  COUNT(*) AS n,
                  AVG({_C1}) AS avg_t1,
                  AVG({_C2}) AS avg_t2,
                  AVG(parsed_score_team_1 + parsed_score_team_2) AS avg_goals,
                  SUM(CASE WHEN {_C1} > {_C2} THEN 1 ELSE 0 END) AS t1_win,
                  SUM(CASE WHEN {_C1} = {_C2} THEN 1 ELSE 0 END) AS draw,
                  SUM(CASE WHEN {_C1} < {_C2} THEN 1 ELSE 0 END) AS t2_win
               FROM forecast_runs
               WHERE parsed_score_team_1 IS NOT NULL
                 AND parsed_score_team_2 IS NOT NULL"""
        ).fetchone()

        # Most-predicted scorelines — canonical order
        top_scores = [dict(r) for r in conn.execute(
            f"""SELECT ({_C1} || '-' || {_C2}) AS scoreline,
                      COUNT(*) AS cnt
               FROM forecast_runs
               WHERE parsed_score_team_1 IS NOT NULL AND parsed_score_team_2 IS NOT NULL
               GROUP BY scoreline ORDER BY cnt DESC LIMIT 8"""
        ).fetchall()]

        # Average win/draw/loss probabilities by model — canonical order
        prob_by_model = [dict(r) for r in conn.execute(
            f"""SELECT model,
                      AVG({_P1}) AS avg_t1,
                      AVG(parsed_draw_probability) AS avg_draw,
                      AVG({_P2}) AS avg_t2,
                      COUNT(*) AS n
               FROM forecast_runs
               WHERE parsed_team1_win_probability IS NOT NULL
               GROUP BY model ORDER BY n DESC"""
        ).fetchall()]

        # Distribution of predicted total goals
        goals_hist = [dict(r) for r in conn.execute(
            """SELECT (parsed_score_team_1 + parsed_score_team_2) AS goals,
                      COUNT(*) AS cnt
               FROM forecast_runs
               WHERE parsed_score_team_1 IS NOT NULL AND parsed_score_team_2 IS NOT NULL
               GROUP BY goals ORDER BY goals"""
        ).fetchall()]

        # Per-match consensus: how the models collectively lean — canonical order
        by_match = [dict(r) for r in conn.execute(
            f"""SELECT match_id,
                      MAX(team_1) AS team_1, MAX(team_2) AS team_2,
                      MAX(phase) AS phase,
                      COUNT(*) AS n,
                      SUM(CASE WHEN {_C1} > {_C2} THEN 1 ELSE 0 END) AS t1_win,
                      SUM(CASE WHEN {_C1} = {_C2} THEN 1 ELSE 0 END) AS draw,
                      SUM(CASE WHEN {_C1} < {_C2} THEN 1 ELSE 0 END) AS t2_win,
                      AVG({_C1}) AS avg_t1,
                      AVG({_C2}) AS avg_t2
               FROM forecast_runs
               WHERE parsed_score_team_1 IS NOT NULL AND parsed_score_team_2 IS NOT NULL
               GROUP BY match_id
               ORDER BY n DESC"""
        ).fetchall()]
        # Attach the single most-predicted scoreline per match (canonical order).
        for m in by_match:
            top = conn.execute(
                f"""SELECT ({_C1} || '-' || {_C2}) AS scoreline,
                          COUNT(*) AS cnt
                   FROM forecast_runs
                   WHERE match_id = ? AND parsed_score_team_1 IS NOT NULL
                     AND parsed_score_team_2 IS NOT NULL
                   GROUP BY scoreline ORDER BY cnt DESC LIMIT 1""",
                (m["match_id"],),
            ).fetchone()
            m["top_scoreline"] = top["scoreline"] if top else None

        return {
            "totals": {
                "total": total, "success": success, "errors": errors,
                "manual": manual, "valid_json": valid_json,
                "total_tokens": total_tokens, "total_cost": total_cost,
                "avg_latency": avg_latency,
            },
            "by_model": by_model,
            "by_prompt": by_prompt,
            "by_moment": by_moment,
            "predictions": {
                "n": pred["n"] or 0,
                "avg_t1": pred["avg_t1"], "avg_t2": pred["avg_t2"],
                "avg_goals": pred["avg_goals"],
                "t1_win": pred["t1_win"] or 0, "draw": pred["draw"] or 0,
                "t2_win": pred["t2_win"] or 0,
            },
            "top_scores": top_scores,
            "prob_by_model": prob_by_model,
            "goals_hist": goals_hist,
            "by_match": by_match,
        }
    finally:
        conn.close()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    db_path = cfg.ROOT / load_config().database_path
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM forecast_runs WHERE run_id = ?", [run_id]
        ).fetchone()
        if not row:
            raise HTTPException(404, "Run not found")
        return dict(row)
    finally:
        conn.close()


# ── Execute experiment ────────────────────────────────────────────────────────

def _parse_local(value: str | None) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime((value or "").strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


# Match times in the dataset are local Brazil time (UTC-3).
LOCAL_TZ = timezone(timedelta(hours=-3))


def _scheduled_dt_local(kickoff_local: str | None, moment: str) -> datetime | None:
    """tz-aware local datetime when this run should fire (kickoff + offset)."""
    dt = _parse_local(kickoff_local)
    if dt is None:
        return None
    dt += timedelta(minutes=MOMENT_OFFSET_MINUTES.get(moment, 0))
    return dt.replace(tzinfo=LOCAL_TZ)


def _scheduled_local(kickoff_local: str | None, moment: str) -> str | None:
    """When this run is intended to fire = kickoff (local) + moment offset."""
    dt = _scheduled_dt_local(kickoff_local, moment)
    return dt.strftime("%Y-%m-%d %H:%M") if dt else None


def _scheduled_utc(kickoff_local: str | None, moment: str) -> str | None:
    """Same instant as :func:`_scheduled_local` but as a UTC ISO string."""
    dt = _scheduled_dt_local(kickoff_local, moment)
    return dt.astimezone(timezone.utc).isoformat() if dt else None


class PlanRequest(BaseModel):
    model_keys: list[str]
    match_ids: Optional[list[str]] = None
    dates: Optional[list[str]] = None
    prompt_ids: list[str]
    team_order_types: list[str] = ["original", "reversed"]
    match_moments: list[str] = ["pre_match"]
    runs_per_combination: int = 1


def _iter_plan(config, req: PlanRequest):
    """Yield one dict per planned execution (the full combinatorial grid)."""
    matches = load_matches(cfg.ROOT / config.matches_csv)
    matches = filter_matches(matches, dates=req.dates, match_ids=req.match_ids)
    models = [
        m for m in config.models
        if m["key"] in req.model_keys and m.get("enabled", True)
    ]
    for match in matches:
        for moment in req.match_moments:
            for order in req.team_order_types:
                for model in models:
                    for prompt_id in req.prompt_ids:
                        for rep in range(1, req.runs_per_combination + 1):
                            run_id = make_run_id(
                                match.match_id, model["key"], prompt_id,
                                order, rep, moment,
                            )
                            yield {
                                "run_id": run_id,
                                "match_id": match.match_id,
                                "teams": f"{match.team_1} vs {match.team_2}",
                                "kickoff_local": match.kickoff_local,
                                "scheduled_local": _scheduled_local(match.kickoff_local, moment),
                                "scheduled_utc": _scheduled_utc(match.kickoff_local, moment),
                                "model": model["key"],
                                "prompt_id": prompt_id,
                                "team_order_type": order,
                                "match_moment": moment,
                                "repetition_number": rep,
                            }


@app.post("/api/plan")
def plan(req: PlanRequest):
    """Preview the full grid of planned executions — what will run, and when."""
    config = load_config()
    db_path = cfg.ROOT / config.database_path
    conn = sqlite3.connect(db_path)
    try:
        existing = {row[0] for row in conn.execute("SELECT run_id FROM forecast_runs")}
    finally:
        conn.close()

    CAP = 3000
    rows: list[dict] = []
    total = done = 0
    truncated = False

    for row in _iter_plan(config, req):
        total += 1
        already = row["run_id"] in existing
        if already:
            done += 1
        if len(rows) < CAP:
            row = dict(row)
            row["already_done"] = already
            rows.append(row)
        else:
            truncated = True

    rows.sort(key=lambda r: (r["scheduled_local"] or "9999", str(r["match_id"]), r["model"]))
    return {
        "total": total,
        "already_done": done,
        "to_run": total - done,
        "truncated": truncated,
        "rows": rows,
    }


# ── Persisted plan (survives page refresh) ─────────────────────────────────────

def _ensure_planned_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS planned_runs (
            run_id TEXT PRIMARY KEY,
            match_id TEXT,
            teams TEXT,
            kickoff_local TEXT,
            scheduled_local TEXT,
            scheduled_utc TEXT,
            model TEXT,
            prompt_id TEXT,
            team_order_type TEXT,
            match_moment TEXT,
            repetition_number INTEGER,
            status TEXT DEFAULT 'scheduled',
            created_at TEXT
        )"""
    )
    # Migrate older tables that pre-date the scheduler columns.
    existing = {row[1] for row in conn.execute("PRAGMA table_info(planned_runs)")}
    if "scheduled_utc" not in existing:
        conn.execute("ALTER TABLE planned_runs ADD COLUMN scheduled_utc TEXT")
    if "status" not in existing:
        conn.execute("ALTER TABLE planned_runs ADD COLUMN status TEXT DEFAULT 'scheduled'")
    conn.commit()


@app.post("/api/plan/save")
def save_plan(req: PlanRequest):
    """Persist the planned grid so it survives a page refresh."""
    config = load_config()
    db_path = cfg.ROOT / config.database_path
    conn = sqlite3.connect(db_path)
    try:
        _ensure_planned_table(conn)
        now = datetime.now(timezone.utc).isoformat()
        saved = 0
        for row in _iter_plan(config, req):
            conn.execute(
                """INSERT OR IGNORE INTO planned_runs
                   (run_id, match_id, teams, kickoff_local, scheduled_local,
                    scheduled_utc, model, prompt_id, team_order_type, match_moment,
                    repetition_number, status, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (row["run_id"], row["match_id"], row["teams"],
                 row["kickoff_local"], row["scheduled_local"], row["scheduled_utc"],
                 row["model"], row["prompt_id"], row["team_order_type"],
                 row["match_moment"], row["repetition_number"], "scheduled", now),
            )
            saved += 1
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM planned_runs").fetchone()[0]
        return {"added": saved, "total_planned": total}
    finally:
        conn.close()


@app.get("/api/planned")
def list_planned(only: str = Query("all", pattern="^(all|pending|done|cancelled)$")):
    """List persisted planned runs with their effective scheduler state."""
    config = load_config()
    db_path = cfg.ROOT / config.database_path
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        _ensure_planned_table(conn)
        rows = conn.execute(
            """SELECT p.*,
                      f.execution_status AS execution_status,
                      f.parsed_score_team_1 AS parsed_score_team_1,
                      f.parsed_score_team_2 AS parsed_score_team_2,
                      f.error_message AS error_message
               FROM planned_runs p
               LEFT JOIN forecast_runs f ON f.run_id = p.run_id
               ORDER BY (p.scheduled_local IS NULL), p.scheduled_local,
                        p.match_id, p.model"""
        ).fetchall()
        now_iso = datetime.now(timezone.utc).isoformat()
        out = []
        counts = {"pending": 0, "done": 0, "cancelled": 0, "running": 0}
        for r in rows:
            d = dict(r)
            executed = d.get("execution_status") is not None
            raw_status = d.get("status") or "scheduled"
            if executed:
                effective = "done"
            elif raw_status == "cancelled":
                effective = "cancelled"
            elif raw_status == "running":
                effective = "running"
            elif raw_status in ("done", "error"):
                effective = raw_status
            else:
                # scheduled: overdue if its time has passed but it hasn't fired
                su = d.get("scheduled_utc")
                effective = "overdue" if (su and su <= now_iso) else "scheduled"
            d["effective_status"] = effective
            d["done"] = effective in ("done",)
            if effective in ("done", "error"):
                counts["done"] += 1
            elif effective == "cancelled":
                counts["cancelled"] += 1
            elif effective == "running":
                counts["running"] += 1
            else:
                counts["pending"] += 1
            out.append(d)
        if only == "pending":
            out = [r for r in out if r["effective_status"] in ("scheduled", "overdue", "running")]
        elif only == "done":
            out = [r for r in out if r["effective_status"] in ("done", "error")]
        elif only == "cancelled":
            out = [r for r in out if r["effective_status"] == "cancelled"]
        return {
            "total": len(rows),
            "pending": counts["pending"] + counts["running"],
            "done": counts["done"],
            "cancelled": counts["cancelled"],
            "rows": out,
        }
    finally:
        conn.close()


@app.post("/api/planned/clear")
def clear_planned(scope: str = Query("all", pattern="^(all|done)$")):
    """Clear the persisted plan — everything, or only the already-executed rows."""
    config = load_config()
    db_path = cfg.ROOT / config.database_path
    conn = sqlite3.connect(db_path)
    try:
        _ensure_planned_table(conn)
        if scope == "done":
            conn.execute(
                """DELETE FROM planned_runs
                   WHERE run_id IN (SELECT run_id FROM forecast_runs)"""
            )
        else:
            conn.execute("DELETE FROM planned_runs")
        conn.commit()
        remaining = conn.execute("SELECT COUNT(*) FROM planned_runs").fetchone()[0]
        return {"remaining": remaining}
    finally:
        conn.close()


class CancelRequest(BaseModel):
    run_ids: Optional[list[str]] = None  # None => all future scheduled runs


@app.post("/api/planned/cancel")
def cancel_planned(req: CancelRequest):
    """Cancel future runs so the scheduler skips them.

    With ``run_ids`` cancels those specific rows; without it cancels every run
    still waiting to fire. Already-executed runs are left untouched.
    """
    config = load_config()
    db_path = cfg.ROOT / config.database_path
    conn = sqlite3.connect(db_path)
    try:
        _ensure_planned_table(conn)
        executed = "run_id NOT IN (SELECT run_id FROM forecast_runs)"
        if req.run_ids:
            ph = ",".join("?" * len(req.run_ids))
            cur = conn.execute(
                f"""UPDATE planned_runs SET status='cancelled'
                    WHERE status IN ('scheduled','running') AND {executed}
                      AND run_id IN ({ph})""",
                req.run_ids,
            )
        else:
            cur = conn.execute(
                f"""UPDATE planned_runs SET status='cancelled'
                    WHERE status IN ('scheduled','running') AND {executed}"""
            )
        conn.commit()
        return {"cancelled": cur.rowcount}
    finally:
        conn.close()


@app.post("/api/planned/reactivate")
def reactivate_planned(req: CancelRequest):
    """Re-enable previously cancelled runs (back to 'scheduled')."""
    config = load_config()
    db_path = cfg.ROOT / config.database_path
    conn = sqlite3.connect(db_path)
    try:
        _ensure_planned_table(conn)
        if req.run_ids:
            ph = ",".join("?" * len(req.run_ids))
            cur = conn.execute(
                f"UPDATE planned_runs SET status='scheduled' "
                f"WHERE status='cancelled' AND run_id IN ({ph})",
                req.run_ids,
            )
        else:
            cur = conn.execute(
                "UPDATE planned_runs SET status='scheduled' WHERE status='cancelled'"
            )
        conn.commit()
        return {"reactivated": cur.rowcount}
    finally:
        conn.close()


# ── Background scheduler ───────────────────────────────────────────────────────
# A lightweight daemon thread fires planned runs once their scheduled time
# (kickoff + moment offset) arrives. So "rodar no intervalo" actually executes
# at half-time without anyone clicking a button.

SCHEDULER_INTERVAL = float(os.environ.get("FIFA_SCHEDULER_INTERVAL", "30"))
SCHEDULER_ENABLED = os.environ.get("FIFA_SCHEDULER", "1") != "0"

_scheduler_stop = threading.Event()
_scheduler_lock = threading.Lock()


def _due_planned(conn: sqlite3.Connection, now_iso: str) -> list[sqlite3.Row]:
    """Planned rows whose time has arrived and that still need to run."""
    return conn.execute(
        """SELECT p.* FROM planned_runs p
           WHERE p.status = 'scheduled'
             AND p.scheduled_utc IS NOT NULL
             AND p.scheduled_utc <= ?
             AND p.run_id NOT IN (SELECT run_id FROM forecast_runs)
           ORDER BY p.scheduled_utc
           LIMIT 25""",
        (now_iso,),
    ).fetchall()


def _execute_planned_row(row: dict) -> str:
    """Fire a single planned execution via the experiment runner."""
    config = load_config()
    model_config = next((m for m in config.models if m["key"] == row["model"]), None)
    if model_config is None:
        return f"ERR model {row['model']} not in config"
    matches = load_matches(cfg.ROOT / config.matches_csv)
    match = next((m for m in matches if str(m.match_id) == str(row["match_id"])), None)
    if match is None:
        return f"ERR match {row['match_id']} not found"

    runner = ExperimentRunner(config, overwrite=False, progress=lambda m: None)
    try:
        return runner._execute_one(
            match,
            row["team_order_type"],
            row["match_moment"],
            model_config,
            row["prompt_id"],
            int(row["repetition_number"] or 1),
        )
    finally:
        runner.close()


def _scheduler_tick() -> None:
    db_path = cfg.ROOT / load_config().database_path
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        _ensure_planned_table(conn)
        now_iso = datetime.now(timezone.utc).isoformat()
        # Reconcile rows already executed (e.g. via the Rodar tab).
        conn.execute(
            """UPDATE planned_runs SET status='done'
               WHERE status IN ('scheduled','running')
                 AND run_id IN (SELECT run_id FROM forecast_runs)"""
        )
        conn.commit()
        due = [dict(r) for r in _due_planned(conn, now_iso)]
    finally:
        conn.close()

    for row in due:
        rid = row["run_id"]
        # Claim the row so a concurrent tick won't double-fire it.
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.execute(
                "UPDATE planned_runs SET status='running' "
                "WHERE run_id=? AND status='scheduled'",
                (rid,),
            )
            conn.commit()
            claimed = cur.rowcount == 1
        finally:
            conn.close()
        if not claimed:
            continue

        try:
            detail = _execute_planned_row(row)
            new_status = "error" if detail.strip().startswith("ERR") else "done"
        except Exception as exc:  # noqa: BLE001 - keep the scheduler alive
            traceback.print_exc()
            detail, new_status = f"ERR {exc!r}", "error"

        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "UPDATE planned_runs SET status=? WHERE run_id=?", (new_status, rid)
            )
            conn.commit()
        finally:
            conn.close()
        print(f"[scheduler] {new_status}: {detail}", flush=True)


def _scheduler_loop() -> None:
    # First tick after a short delay so startup finishes first.
    while not _scheduler_stop.wait(SCHEDULER_INTERVAL):
        if not _scheduler_lock.acquire(blocking=False):
            continue
        try:
            _scheduler_tick()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        finally:
            _scheduler_lock.release()


@app.on_event("startup")
def _start_scheduler() -> None:
    if not SCHEDULER_ENABLED:
        print("[scheduler] disabled (FIFA_SCHEDULER=0)", flush=True)
        return
    threading.Thread(target=_scheduler_loop, daemon=True, name="fifa-scheduler").start()
    print(
        f"[scheduler] started — checking every {SCHEDULER_INTERVAL:.0f}s", flush=True
    )


@app.on_event("shutdown")
def _stop_scheduler() -> None:
    _scheduler_stop.set()


class RunRequest(BaseModel):
    model_keys: list[str]
    match_ids: Optional[list[str]] = None
    dates: Optional[list[str]] = None
    prompt_ids: list[str]
    team_order_types: list[str] = ["original", "reversed"]
    match_moments: list[str] = ["pre_match"]
    runs_per_combination: int = 1
    overwrite: bool = False


@app.post("/api/runs/start")
def start_run(req: RunRequest):
    task_id = str(uuid.uuid4())
    log_q: queue.Queue = queue.Queue()
    _active_tasks[task_id] = log_q

    def _run():
        try:
            config = load_config()
            config.models = [m for m in config.models if m["key"] in req.model_keys]
            config.prompt_ids = req.prompt_ids
            config.team_order_types = req.team_order_types
            config.match_moments = req.match_moments
            config.runs_per_combination = req.runs_per_combination

            def _log(msg: str):
                log_q.put({"type": "log", "message": msg})

            runner = ExperimentRunner(config, progress=_log, overwrite=req.overwrite)
            stats = runner.run(match_ids=req.match_ids, dates=req.dates)
            log_q.put({"type": "done", "stats": stats})
        except Exception:
            log_q.put({"type": "error", "message": traceback.format_exc()})
        finally:
            log_q.put(None)

    threading.Thread(target=_run, daemon=True).start()
    return {"task_id": task_id}


@app.get("/api/stream/{task_id}")
async def stream_logs(task_id: str):
    q = _active_tasks.get(task_id)
    if q is None:
        raise HTTPException(404, "Task not found")

    async def generator():
        loop = asyncio.get_event_loop()
        while True:
            msg = await loop.run_in_executor(None, q.get)
            if msg is None:
                break
            yield {"data": json.dumps(msg)}
        _active_tasks.pop(task_id, None)

    return EventSourceResponse(generator())


# ── Manual run ────────────────────────────────────────────────────────────────

class ManualRunRequest(BaseModel):
    match_id: str
    model_key: str
    prompt_id: str
    team_order_type: str = "original"
    match_moment: str = "pre_match"
    repetition_number: int = 1
    raw_response: str
    notes: Optional[str] = None


@app.post("/api/manual")
def save_manual_run(data: ManualRunRequest):
    config = load_config()
    matches = load_matches(cfg.ROOT / config.matches_csv)
    match = next((m for m in matches if str(m.match_id) == str(data.match_id)), None)
    if not match:
        raise HTTPException(404, "Match not found")

    model_cfg = next((m for m in config.models if m["key"] == data.model_key), None)
    if not model_cfg:
        raise HTTPException(404, "Model not found")

    run_id = (
        make_run_id(
            data.match_id, data.model_key, data.prompt_id,
            data.team_order_type, data.repetition_number, data.match_moment,
        )
        + "__manual"
    )

    parsed = parse_forecast(data.raw_response, data.prompt_id)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    is_rev = data.team_order_type == "reversed"

    record = RunRecord(
        run_id=run_id,
        match_id=data.match_id,
        phase=match.phase,
        team_1=match.team_1,
        team_2=match.team_2,
        kickoff_datetime=match.kickoff_datetime,
        team_order_type=data.team_order_type,
        match_moment=data.match_moment,
        prompt_team_1=match.team_2 if is_rev else match.team_1,
        prompt_team_2=match.team_1 if is_rev else match.team_2,
        provider=model_cfg.get("provider"),
        model=data.model_key,
        model_id=model_cfg.get("model_id"),
        prompt_id=data.prompt_id,
        repetition_number=data.repetition_number,
        request_timestamp_utc=now,
        response_timestamp_utc=now,
        request_timestamp_local=now,
        response_timestamp_local=now,
        raw_response=data.raw_response,
        parsed_json=json.dumps(parsed.parsed_json) if parsed.parsed_json else None,
        parsed_score_team_1=parsed.score_team_1,
        parsed_score_team_2=parsed.score_team_2,
        parsed_team1_win_probability=parsed.team1_win_probability,
        parsed_draw_probability=parsed.draw_probability,
        parsed_team2_win_probability=parsed.team2_win_probability,
        white_hat=parsed.hats.get("white_hat"),
        red_hat=parsed.hats.get("red_hat"),
        black_hat=parsed.hats.get("black_hat"),
        yellow_hat=parsed.hats.get("yellow_hat"),
        green_hat=parsed.hats.get("green_hat"),
        blue_hat=parsed.hats.get("blue_hat"),
        json_valid=1 if parsed.json_valid else 0,
        validation_notes="; ".join(parsed.validation_notes) if parsed.validation_notes else (data.notes or None),
        execution_status="manual",
        attempt_count=1,
        created_at=now,
    )

    db = Database(cfg.ROOT / config.database_path)
    db.insert_run(record)
    db.close()

    # Archive the same four JSON artifacts as an automated run, so manual entries
    # are inspectable in data/raw_requests, raw_responses, traces and metadata.
    archive = FileArchive()
    archive.write_request(
        run_id,
        {
            "manual": True,
            "provider": model_cfg.get("provider"),
            "model_id": model_cfg.get("model_id"),
            "prompt_id": data.prompt_id,
            "team_order_type": data.team_order_type,
            "match_moment": data.match_moment,
        },
    )
    archive.write_response(
        run_id,
        {"raw_text": data.raw_response, "parsed": parsed.parsed_json},
    )
    archive.write_trace(run_id, {"manual": True, "notes": data.notes})
    archive.write_metadata(
        run_id,
        {
            "run_id": run_id,
            "match_id": data.match_id,
            "model": data.model_key,
            "prompt_id": data.prompt_id,
            "match_moment": data.match_moment,
            "json_valid": bool(parsed.json_valid),
            "execution_status": "manual",
            "created_at": now,
        },
    )

    return {
        "run_id": run_id,
        "json_valid": parsed.json_valid,
        "parsed": {
            "score_team_1": parsed.score_team_1,
            "score_team_2": parsed.score_team_2,
            "team1_win_probability": parsed.team1_win_probability,
            "draw_probability": parsed.draw_probability,
            "team2_win_probability": parsed.team2_win_probability,
        },
        "validation_notes": parsed.validation_notes,
    }
