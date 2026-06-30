"""Web dashboard for the FIFA forecast collector (FastAPI).

A small operations UI intended to run as a Coolify web service:

* view dataset stats, the quality/variability report, and ingested results;
* download the forecasts and report Excel workbooks;
* trigger ``run`` / ``fetch`` / ``report`` as background jobs (one at a time).

Serve with::

    uvicorn fifa_forecast.web:app --host 0.0.0.0 --port 8000

Security: set ``DASHBOARD_PASSWORD`` (and optionally ``DASHBOARD_USER``, default
``admin``) to require HTTP Basic auth. If no password is set the dashboard is
read-only — the money-spending action endpoints are disabled — so a public URL
can never trigger paid API calls by accident. ``/health`` is always open.
"""

from __future__ import annotations

import html
import os
import secrets
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from . import __version__
from . import config as cfg
from .export import export_workbook

app = FastAPI(title="FIFA WC2026 Forecast Dashboard", version=__version__)

_DASHBOARD_USER = os.environ.get("DASHBOARD_USER", "admin")
_DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD")
_security = HTTPBasic(auto_error=False)


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def require_view(creds: HTTPBasicCredentials | None = Depends(_security)) -> None:
    """Enforce Basic auth on every page when a password is configured."""
    if not _DASHBOARD_PASSWORD:
        return  # open, read-only mode
    ok = (
        creds is not None
        and secrets.compare_digest(creds.username, _DASHBOARD_USER)
        and secrets.compare_digest(creds.password, _DASHBOARD_PASSWORD)
    )
    if not ok:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )


def require_action(creds: HTTPBasicCredentials | None = Depends(_security)) -> None:
    """Gate state-changing endpoints: a password MUST be configured + valid."""
    if not _DASHBOARD_PASSWORD:
        raise HTTPException(
            status_code=403,
            detail="Actions are disabled. Set DASHBOARD_PASSWORD to enable run/fetch.",
        )
    require_view(creds)


# --------------------------------------------------------------------------- #
# Background job runner (one at a time)
# --------------------------------------------------------------------------- #
_job_lock = threading.Lock()
_JOB: dict[str, Any] = {
    "command": None,
    "status": "idle",  # idle | running | finished | failed
    "started_at": None,
    "finished_at": None,
    "returncode": None,
    "log_path": None,
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def start_job(args: list[str]) -> tuple[bool, str]:
    with _job_lock:
        if _JOB["status"] == "running":
            return False, "A job is already running."
        log_dir = cfg.DATA_DIR / "jobs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = log_dir / f"{stamp}_{args[0]}.log"
        cmd = [sys.executable, "-m", "fifa_forecast", *args]
        _JOB.update(
            command=" ".join(args),
            status="running",
            started_at=_now(),
            finished_at=None,
            returncode=None,
            log_path=str(log_path),
        )

    def _run() -> None:
        try:
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(f"$ python -m fifa_forecast {' '.join(args)}\n\n")
                fh.flush()
                proc = subprocess.Popen(
                    cmd,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    cwd=str(cfg.ROOT),
                    env=os.environ.copy(),
                )
                rc = proc.wait()
        except Exception as exc:  # noqa: BLE001
            rc = -1
            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"\n[runner error] {exc!r}\n")
            except Exception:
                pass
        with _job_lock:
            _JOB.update(
                status="finished" if rc == 0 else "failed",
                finished_at=_now(),
                returncode=rc,
            )

    threading.Thread(target=_run, daemon=True).start()
    return True, "started"


def _job_log_tail(max_lines: int = 60) -> str:
    path = _JOB.get("log_path")
    if not path or not Path(path).exists():
        return ""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-max_lines:])


# --------------------------------------------------------------------------- #
# Dataset stats
# --------------------------------------------------------------------------- #
def _db_path() -> Path:
    config = cfg.load_config()
    return cfg.ROOT / config.database_path


def _summary() -> dict[str, Any]:
    path = _db_path()
    out: dict[str, Any] = {"db_exists": path.exists(), "db_path": str(path)}
    if not path.exists():
        return out
    con = sqlite3.connect(path)
    try:
        con.row_factory = sqlite3.Row

        def scalar(sql: str) -> int:
            row = con.execute(sql).fetchone()
            return int(row[0]) if row and row[0] is not None else 0

        out["total"] = scalar("SELECT COUNT(*) FROM forecast_runs")
        out["success"] = scalar("SELECT COUNT(*) FROM forecast_runs WHERE execution_status='success'")
        out["error"] = scalar("SELECT COUNT(*) FROM forecast_runs WHERE execution_status='error'")
        out["valid"] = scalar("SELECT COUNT(*) FROM forecast_runs WHERE json_valid=1")
        out["matches"] = scalar("SELECT COUNT(DISTINCT match_id) FROM forecast_runs")
        out["models"] = scalar("SELECT COUNT(DISTINCT model) FROM forecast_runs")
        out["prompts"] = scalar("SELECT COUNT(DISTINCT prompt_id) FROM forecast_runs")
        try:
            out["results"] = scalar("SELECT COUNT(*) FROM match_results")
            out["results_finished"] = scalar(
                "SELECT COUNT(*) FROM match_results WHERE status='finished'"
            )
        except sqlite3.OperationalError:
            out["results"] = 0
            out["results_finished"] = 0
    finally:
        con.close()
    return out


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #
_STYLE = """
<style>
:root{color-scheme:dark}
body{font-family:system-ui,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1420;color:#e6e9ef}
.wrap{max-width:980px;margin:0 auto;padding:24px}
h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;margin:24px 0 8px;color:#9fb0c8}
.muted{color:#7c8aa0;font-size:13px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0}
.card{background:#18203250;border:1px solid #263350;border-radius:10px;padding:12px 16px;min-width:120px}
.card .n{font-size:22px;font-weight:600} .card .l{font-size:12px;color:#8aa}
.btnrow{display:flex;flex-wrap:wrap;gap:10px;align-items:end;margin:8px 0}
form.inline{display:inline-flex;gap:8px;align-items:end;background:#161d2e;border:1px solid #263350;padding:10px 12px;border-radius:10px}
label{font-size:12px;color:#9fb0c8;display:block;margin-bottom:3px}
input,select{background:#0f1420;border:1px solid #2c3a5a;color:#e6e9ef;border-radius:6px;padding:6px 8px}
button{background:#2d6cdf;border:0;color:#fff;border-radius:6px;padding:7px 14px;font-weight:600;cursor:pointer}
button.warn{background:#c2410c} button.ghost{background:#243150}
a{color:#6ea8fe} pre{background:#0b0f18;border:1px solid #1e2840;border-radius:8px;padding:12px;overflow:auto;font-size:12px;max-height:340px}
table{border-collapse:collapse;font-size:12px;width:100%} th,td{border:1px solid #243150;padding:4px 8px;text-align:left}
th{background:#18213450} .banner{background:#3a2a0a;border:1px solid #6b4e16;padding:8px 12px;border-radius:8px;font-size:13px;margin:10px 0}
.status-running{color:#f5c542}.status-finished{color:#3ddc84}.status-failed{color:#ff6b6b}.status-idle{color:#7c8aa0}
</style>
"""


def _page(body: str, *, refresh: bool = False) -> str:
    meta = '<meta http-equiv="refresh" content="4">' if refresh else ""
    return (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>{meta}"
        f"<title>FIFA WC2026 Forecast Dashboard</title>{_STYLE}</head>"
        f"<body><div class='wrap'>{body}</div></body></html>"
    )


def _card(n: Any, label: str) -> str:
    return f"<div class='card'><div class='n'>{html.escape(str(n))}</div><div class='l'>{html.escape(label)}</div></div>"


def _render_home() -> str:
    s = _summary()
    job = dict(_JOB)
    running = job["status"] == "running"

    if not s.get("db_exists"):
        cards = "<div class='banner'>No database yet. Run a collection to create it.</div>"
    else:
        cards = "<div class='cards'>" + "".join(
            [
                _card(s.get("total", 0), "executions"),
                _card(s.get("success", 0), "success"),
                _card(s.get("error", 0), "errors"),
                _card(s.get("valid", 0), "valid JSON"),
                _card(s.get("matches", 0), "matches"),
                _card(s.get("models", 0), "models"),
                _card(s.get("results", 0), "results"),
                _card(s.get("results_finished", 0), "finished"),
            ]
        ) + "</div>"

    actions_enabled = bool(_DASHBOARD_PASSWORD)
    banner = ""
    if not actions_enabled:
        banner = (
            "<div class='banner'>Read-only mode — set <code>DASHBOARD_PASSWORD</code> "
            "in Coolify to enable Run / Fetch / Report buttons.</div>"
        )

    disabled = "" if (actions_enabled and not running) else "disabled"
    run_confirm = (
        "onsubmit=\"return confirm('This calls the paid LLM APIs and may cost money. Continue?')\""
    )
    actions = f"""
    <h2>Actions</h2>
    <div class='btnrow'>
      <form class='inline' method='post' action='./actions/fetch'>
        <div><label>Fetch</label>
        <select name='mode'><option value='results'>results</option>
        <option value='fixtures'>fixtures</option><option value='both'>both</option></select></div>
        <label><input type='checkbox' name='dry_run' value='1'> dry-run</label>
        <button class='ghost' {disabled}>Fetch</button>
      </form>
      <form class='inline' method='post' action='./actions/report'>
        <div><label>Analysis</label><span class='muted'>rebuild report</span></div>
        <button class='ghost' {disabled}>Report</button>
      </form>
      <form class='inline' method='post' action='./actions/run' {run_confirm}>
        <div><label>Run date (optional)</label><input name='date' placeholder='2026-06-27'></div>
        <label><input type='checkbox' name='dry_run' value='1'> dry-run</label>
        <button class='warn' {disabled}>Run forecasts 💸</button>
      </form>
    </div>
    """

    status_cls = f"status-{job['status']}"
    log = html.escape(_job_log_tail())
    job_panel = f"""
    <h2>Current job</h2>
    <p>command: <code>{html.escape(str(job['command']))}</code> ·
       status: <b class='{status_cls}'>{job['status']}</b> ·
       started: {html.escape(str(job['started_at']))} ·
       rc: {html.escape(str(job['returncode']))}</p>
    <pre>{log or '(no job run yet)'}</pre>
    """

    downloads = """
    <h2>Downloads & views</h2>
    <p>
      <a href='./evaluate'>🎯 Forecast vs results (web)</a> &nbsp;·&nbsp;
      <a href='./report'>📊 Report (web)</a> &nbsp;·&nbsp;
      <a href='./results'>⚽ Results (web)</a>
    </p>
    <p>
      <a href='./download/forecasts'>⬇ Forecasts (.xlsx)</a> &nbsp;·&nbsp;
      <a href='./download/report'>⬇ Report (.xlsx)</a> &nbsp;·&nbsp;
      <a href='./download/evaluation'>⬇ Evaluation (.xlsx)</a>
    </p>
    """

    return _page(
        f"<h1>FIFA World Cup 2026 — Forecast Dashboard</h1>"
        f"<p class='muted'>v{__version__} · DB: {html.escape(s.get('db_path',''))}</p>"
        f"{banner}{cards}{actions}{job_panel}{downloads}",
        refresh=running,
    )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": __version__})


@app.get("/", response_class=HTMLResponse)
def home(_: None = Depends(require_view)) -> HTMLResponse:
    return HTMLResponse(_render_home())


@app.get("/api/summary")
def api_summary(_: None = Depends(require_view)) -> JSONResponse:
    return JSONResponse({"summary": _summary(), "job": _JOB})


@app.get("/report", response_class=HTMLResponse)
def report_page(_: None = Depends(require_view)) -> HTMLResponse:
    from .analysis import ReportOptions, build_report

    config = cfg.load_config()
    try:
        rep = build_report(config, ReportOptions())
    except FileNotFoundError:
        return HTMLResponse(_page("<h1>Report</h1><p>No database yet.</p>"))
    pick = ["Overview", "By Model", "Prompt Quality", "Variability by Model", "Recommended Reps", "Errors"]
    blocks = []
    for name in pick:
        df = rep.tables.get(name)
        if df is None or df.empty:
            continue
        blocks.append(f"<h2>{html.escape(name)}</h2>" + df.to_html(index=False, border=0))
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in rep.notes)
    return HTMLResponse(
        _page(
            "<h1>Quality & variability report</h1><p><a href='./'>← back</a></p>"
            + "".join(blocks)
            + (f"<h2>Notes</h2><ul>{notes}</ul>" if notes else "")
        )
    )


@app.get("/results", response_class=HTMLResponse)
def results_page(_: None = Depends(require_view)) -> HTMLResponse:
    import pandas as pd

    path = _db_path()
    if not path.exists():
        return HTMLResponse(_page("<h1>Results</h1><p>No database yet.</p>"))
    con = sqlite3.connect(path)
    try:
        try:
            df = pd.read_sql_query(
                "SELECT match_id, team_1, team_2, status, actual_score_team_1, "
                "actual_score_team_2, actual_winner, source FROM match_results "
                "ORDER BY CAST(match_id AS INTEGER)",
                con,
            )
        except Exception:
            df = pd.DataFrame()
    finally:
        con.close()
    table = df.to_html(index=False, border=0) if not df.empty else "<p>No results ingested yet.</p>"
    return HTMLResponse(_page("<h1>Match results</h1><p><a href='./'>← back</a></p>" + table))


@app.get("/evaluate", response_class=HTMLResponse)
def evaluate_page(_: None = Depends(require_view)) -> HTMLResponse:
    from .evaluation import build_evaluation

    config = cfg.load_config()
    try:
        ev = build_evaluation(config)
    except FileNotFoundError:
        return HTMLResponse(_page("<h1>Forecast vs results</h1><p>No database yet.</p>"))
    if not ev.tables:
        notes = "".join(f"<p>{html.escape(n)}</p>" for n in ev.notes)
        return HTMLResponse(
            _page("<h1>Forecast vs results</h1><p><a href='./'>← back</a></p>" + notes)
        )
    blocks = []
    for name in ["Leaderboard (by model)", "By Model & Prompt", "By Moment", "By Match"]:
        df = ev.tables.get(name)
        if df is None or df.empty:
            continue
        blocks.append(f"<h2>{html.escape(name)}</h2>" + df.to_html(index=False, border=0))
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in ev.notes)
    return HTMLResponse(
        _page(
            "<h1>AI forecast vs actual results</h1><p><a href='./'>← back</a></p>"
            + "".join(blocks)
            + (f"<h2>Notes</h2><ul>{notes}</ul>" if notes else "")
        )
    )


@app.get("/download/evaluation")
def download_evaluation(_: None = Depends(require_view)) -> FileResponse:
    from .evaluation import build_evaluation, write_evaluation_excel

    config = cfg.load_config()
    ev = build_evaluation(config)
    path = write_evaluation_excel(ev)
    return FileResponse(path, filename=Path(path).name)


@app.get("/download/forecasts")
def download_forecasts(_: None = Depends(require_view)) -> FileResponse:
    config = cfg.load_config()
    path = export_workbook(config)
    return FileResponse(path, filename=Path(path).name)


@app.get("/download/report")
def download_report(_: None = Depends(require_view)) -> FileResponse:
    from .analysis import ReportOptions, build_report, write_report_excel

    config = cfg.load_config()
    rep = build_report(config, ReportOptions())
    path = write_report_excel(rep)
    return FileResponse(path, filename=Path(path).name)


@app.post("/actions/run")
def action_run(
    date: str = Form(default=""),
    dry_run: str = Form(default=""),
    _: None = Depends(require_action),
) -> RedirectResponse:
    args = ["run"]
    if date.strip():
        args += ["--date", date.strip()]
    if dry_run:
        args.append("--dry-run")
    start_job(args)
    return RedirectResponse(url="./", status_code=303)


@app.post("/actions/fetch")
def action_fetch(
    mode: str = Form(default="results"),
    dry_run: str = Form(default=""),
    _: None = Depends(require_action),
) -> RedirectResponse:
    args = ["fetch"]
    if mode in ("results", "both"):
        args.append("--results")
    if mode in ("fixtures", "both"):
        args.append("--fixtures")
    if dry_run:
        args.append("--dry-run")
    start_job(args)
    return RedirectResponse(url="./", status_code=303)


@app.post("/actions/report")
def action_report(_: None = Depends(require_action)) -> RedirectResponse:
    start_job(["report"])
    return RedirectResponse(url="./", status_code=303)
