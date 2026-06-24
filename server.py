"""FastAPI web server for the FIFA WC 2026 LLM Eval dashboard."""

from __future__ import annotations

import asyncio
import json
import queue
import sqlite3
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from fifa_forecast import config as cfg
from fifa_forecast.config import load_config
from fifa_forecast.matches import load_matches
from fifa_forecast.parsing import parse_forecast
from fifa_forecast.prompts import MATCH_MOMENTS, PROMPT_TEMPLATES
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
        return {"total": total, "rows": [dict(r) for r in rows]}
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

class RunRequest(BaseModel):
    model_keys: list[str]
    match_ids: Optional[list[str]] = None
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
            stats = runner.run(match_ids=req.match_ids)
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
