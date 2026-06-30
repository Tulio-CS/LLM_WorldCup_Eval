"""Web dashboard for the FIFA forecast collector (FastAPI single-page app).

An operations UI intended to run as the Coolify web service:

* **Overview** — dataset KPIs, the accuracy leaderboard, and the live job log.
* **Run** — launch a collection with full control: pick a single match (or all),
  the moment(s), the models, and the repetition count; dry-run toggle. Plus
  one-click fetch-results / fetch-fixtures / rebuild-report.
* **Results** — the ingested actual scores.
* **Compare** — for any match, what each model predicted vs the real result.

Serve with::

    uvicorn fifa_forecast.web:app --host 0.0.0.0 --port 8000

Security: set ``DASHBOARD_PASSWORD`` (and optionally ``DASHBOARD_USER``, default
``admin``) for HTTP Basic auth. Without a password the UI is read-only — the
money-spending action endpoints are disabled. ``/health`` is always open.
"""

from __future__ import annotations

import math
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from . import __version__
from . import config as cfg
from . import prompts as prompt_lib
from .evaluation import build_evaluation, per_match_comparison, write_evaluation_excel
from .export import export_workbook
from .matches import load_matches

app = FastAPI(title="FIFA WC2026 Forecast Dashboard", version=__version__)

_DASHBOARD_USER = os.environ.get("DASHBOARD_USER", "admin")
_DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD")
_security = HTTPBasic(auto_error=False)


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def require_view(creds: HTTPBasicCredentials | None = Depends(_security)) -> None:
    if not _DASHBOARD_PASSWORD:
        return
    ok = (
        creds is not None
        and secrets.compare_digest(creds.username, _DASHBOARD_USER)
        and secrets.compare_digest(creds.password, _DASHBOARD_PASSWORD)
    )
    if not ok:
        raise HTTPException(401, "Unauthorized", {"WWW-Authenticate": "Basic"})


def require_action(creds: HTTPBasicCredentials | None = Depends(_security)) -> None:
    if not _DASHBOARD_PASSWORD:
        raise HTTPException(403, "Actions disabled. Set DASHBOARD_PASSWORD to enable.")
    require_view(creds)


# --------------------------------------------------------------------------- #
# Background job runner (one at a time)
# --------------------------------------------------------------------------- #
_job_lock = threading.Lock()
_JOB: dict[str, Any] = {
    "command": None, "status": "idle", "started_at": None,
    "finished_at": None, "returncode": None, "log_path": None,
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
        _JOB.update(command=" ".join(args), status="running", started_at=_now(),
                    finished_at=None, returncode=None, log_path=str(log_path))

    def _run() -> None:
        rc = -1
        try:
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(f"$ python -m fifa_forecast {' '.join(args)}\n\n")
                fh.flush()
                proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                        cwd=str(cfg.ROOT), env=os.environ.copy())
                rc = proc.wait()
        except Exception as exc:  # noqa: BLE001
            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"\n[runner error] {exc!r}\n")
            except Exception:
                pass
        with _job_lock:
            _JOB.update(status="finished" if rc == 0 else "failed",
                        finished_at=_now(), returncode=rc)

    threading.Thread(target=_run, daemon=True).start()
    return True, "started"


def _job_payload() -> dict[str, Any]:
    job = dict(_JOB)
    log = ""
    path = job.get("log_path")
    if path and Path(path).exists():
        try:
            log = "\n".join(
                Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
            )
        except Exception:
            log = ""
    job["log"] = log
    progress = None
    found = re.findall(r"\[(\d+)/(\d+)\]", log)
    if found:
        done, total = found[-1]
        progress = {"done": int(done), "total": int(total)}
    job["progress"] = progress
    return job


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def _db_path() -> Path:
    return cfg.ROOT / cfg.load_config().database_path


def _summary() -> dict[str, Any]:
    path = _db_path()
    out: dict[str, Any] = {"db_exists": path.exists()}
    if not path.exists():
        return out
    con = sqlite3.connect(path)
    try:
        def n(sql: str) -> int:
            r = con.execute(sql).fetchone()
            return int(r[0]) if r and r[0] is not None else 0
        out["total"] = n("SELECT COUNT(*) FROM forecast_runs")
        out["success"] = n("SELECT COUNT(*) FROM forecast_runs WHERE execution_status='success'")
        out["error"] = n("SELECT COUNT(*) FROM forecast_runs WHERE execution_status='error'")
        out["valid"] = n("SELECT COUNT(*) FROM forecast_runs WHERE json_valid=1")
        out["matches"] = n("SELECT COUNT(DISTINCT match_id) FROM forecast_runs")
        out["models"] = n("SELECT COUNT(DISTINCT model) FROM forecast_runs")
        try:
            out["results"] = n("SELECT COUNT(*) FROM match_results")
            out["finished"] = n("SELECT COUNT(*) FROM match_results WHERE status='finished'")
        except sqlite3.OperationalError:
            out["results"] = out["finished"] = 0
    finally:
        con.close()
    return out


def _matches_overview(config: cfg.Config) -> list[dict[str, Any]]:
    matches = load_matches(cfg.ROOT / config.matches_csv)
    path = _db_path()
    pred_ids: set[str] = set()
    finished_ids: set[str] = set()
    if path.exists():
        con = sqlite3.connect(path)
        try:
            pred_ids = {r[0] for r in con.execute("SELECT DISTINCT match_id FROM forecast_runs")}
            try:
                finished_ids = {
                    r[0] for r in con.execute(
                        "SELECT match_id FROM match_results WHERE status='finished'"
                    )
                }
            except sqlite3.OperationalError:
                pass
        finally:
            con.close()
    out = []
    for m in matches:
        out.append({
            "match_id": m.match_id, "team_1": m.team_1, "team_2": m.team_2,
            "local_date": m.local_date, "phase": m.phase,
            "has_predictions": m.match_id in pred_ids,
            "finished": m.match_id in finished_ids,
            "label": f"{m.match_id}. {m.team_1} vs {m.team_2} ({m.local_date})",
        })
    return out


def _results_rows() -> list[dict[str, Any]]:
    path = _db_path()
    if not path.exists():
        return []
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        try:
            rows = con.execute(
                "SELECT match_id, team_1, team_2, status, actual_score_team_1, "
                "actual_score_team_2, actual_winner, source FROM match_results "
                "ORDER BY CAST(match_id AS INTEGER)"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [dict(r) for r in rows]
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# JSON APIs
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": __version__})


@app.get("/api/bootstrap")
def api_bootstrap(_: None = Depends(require_view)) -> JSONResponse:
    config = cfg.load_config()
    models = [
        {"key": m["key"], "provider": m.get("provider"), "model_id": m.get("model_id")}
        for m in config.enabled_models()
    ]
    return JSONResponse({
        "version": __version__,
        "actions_enabled": bool(_DASHBOARD_PASSWORD),
        "summary": _summary(),
        "moments": list(prompt_lib.MATCH_MOMENTS),
        "default_moments": list(config.match_moments),
        "reps_default": config.runs_per_combination,
        "models": models,
        "prompts": list(config.prompt_ids),
        "matches": _matches_overview(config),
    })


@app.get("/api/job")
def api_job(_: None = Depends(require_view)) -> JSONResponse:
    return JSONResponse(_job_payload())


@app.get("/api/summary")
def api_summary(_: None = Depends(require_view)) -> JSONResponse:
    return JSONResponse(_summary())


@app.get("/api/results")
def api_results(_: None = Depends(require_view)) -> JSONResponse:
    return JSONResponse({"results": _results_rows()})


def _jsonable(o: Any) -> Any:
    """Recursively make a value strict-JSON safe (NaN/Inf -> None, numpy -> py)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            return _jsonable(o.item())
        except Exception:
            return o
    return o


def _records(df: Any) -> list[dict[str, Any]]:
    """DataFrame -> JSON-safe records (NaN/Inf -> None)."""
    if df is None or getattr(df, "empty", True):
        return []
    out = []
    for rec in df.to_dict(orient="records"):
        out.append({
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in rec.items()
        })
    return out


@app.get("/api/evaluate")
def api_evaluate(_: None = Depends(require_view)) -> JSONResponse:
    config = cfg.load_config()
    try:
        ev = build_evaluation(config)
    except FileNotFoundError:
        return JSONResponse({"leaderboard": [], "by_prompt": [], "notes": ["No database yet."]})
    return JSONResponse({
        "leaderboard": _records(ev.tables.get("Leaderboard (by model)")),
        "by_prompt": _records(ev.tables.get("By Model & Prompt")),
        "notes": ev.notes,
    })


@app.get("/api/compare")
def api_compare(match_id: str, _: None = Depends(require_view)) -> JSONResponse:
    return JSONResponse(_jsonable(per_match_comparison(cfg.load_config(), match_id)))


# --------------------------------------------------------------------------- #
# Actions (JSON body)
# --------------------------------------------------------------------------- #
class RunReq(BaseModel):
    match_id: str | None = None
    moments: list[str] = []
    models: list[str] = []
    reps: int | None = None
    dry_run: bool = False


class FetchReq(BaseModel):
    mode: str = "results"   # results | fixtures | both
    dry_run: bool = False


@app.post("/actions/run")
def action_run(req: RunReq, _: None = Depends(require_action)) -> JSONResponse:
    args = ["run", "--no-export"]
    if req.match_id:
        args += ["--match-id", req.match_id]
    for m in req.moments:
        if m in prompt_lib.MATCH_MOMENTS:
            args += ["--moment", m]
    for k in req.models:
        args += ["--model", k]
    if req.reps:
        args += ["--reps", str(req.reps)]
    if req.dry_run:
        args.append("--dry-run")
    ok, msg = start_job(args)
    return JSONResponse({"ok": ok, "message": msg, "command": " ".join(args)},
                        status_code=200 if ok else 409)


@app.post("/actions/fetch")
def action_fetch(req: FetchReq, _: None = Depends(require_action)) -> JSONResponse:
    args = ["fetch"]
    if req.mode in ("results", "both"):
        args.append("--results")
    if req.mode in ("fixtures", "both"):
        args.append("--fixtures")
    if req.dry_run:
        args.append("--dry-run")
    ok, msg = start_job(args)
    return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 409)


@app.post("/actions/report")
def action_report(_: None = Depends(require_action)) -> JSONResponse:
    ok, msg = start_job(["report"])
    return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 409)


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #
@app.get("/download/forecasts")
def download_forecasts(_: None = Depends(require_view)) -> FileResponse:
    path = export_workbook(cfg.load_config())
    return FileResponse(path, filename=Path(path).name)


@app.get("/download/evaluation")
def download_evaluation(_: None = Depends(require_view)) -> FileResponse:
    ev = build_evaluation(cfg.load_config())
    path = write_evaluation_excel(ev)
    return FileResponse(path, filename=Path(path).name)


@app.get("/download/report")
def download_report(_: None = Depends(require_view)) -> FileResponse:
    from .analysis import ReportOptions, build_report, write_report_excel

    rep = build_report(cfg.load_config(), ReportOptions())
    path = write_report_excel(rep)
    return FileResponse(path, filename=Path(path).name)


# --------------------------------------------------------------------------- #
# Single-page app
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def home(_: None = Depends(require_view)) -> HTMLResponse:
    return HTMLResponse(_SPA_HTML)


_SPA_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FIFA WC2026 — Forecast Console</title>
<style>
:root{
  --bg:#0b0f17; --panel:#121a28; --panel2:#0e1521; --border:#22304a; --border2:#2c3d5c;
  --txt:#e8ecf4; --muted:#8a99b5; --accent:#3b82f6; --accent2:#1e4ed8;
  --ok:#22c55e; --warn:#f59e0b; --err:#ef4444;
}
*{box-sizing:border-box} html,body{margin:0}
body{background:radial-gradient(1200px 600px at 80% -10%,#16233b 0%,var(--bg) 55%);
  color:var(--txt);font:14px/1.45 system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;min-height:100vh}
a{color:#7eb0ff;text-decoration:none} a:hover{text-decoration:underline}
.top{position:sticky;top:0;z-index:10;backdrop-filter:blur(8px);
  background:#0b0f17cc;border-bottom:1px solid var(--border);padding:12px 20px;
  display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.brand{font-weight:700;font-size:16px;letter-spacing:.2px}
.brand small{color:var(--muted);font-weight:400;margin-left:8px}
.kpis{display:flex;gap:14px;margin-left:auto;flex-wrap:wrap}
.kpi{display:flex;flex-direction:column;align-items:flex-end;line-height:1.1}
.kpi b{font-size:16px} .kpi span{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.4px}
.badge{padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;border:1px solid var(--border2)}
.b-idle{color:var(--muted)} .b-running{color:#0b0f17;background:var(--warn);border-color:var(--warn)}
.b-finished{color:#0b0f17;background:var(--ok);border-color:var(--ok)} .b-failed{color:#fff;background:var(--err);border-color:var(--err)}
.tabs{display:flex;gap:4px;padding:0 20px;border-bottom:1px solid var(--border);background:transparent}
.tab{padding:12px 16px;cursor:pointer;color:var(--muted);border-bottom:2px solid transparent;font-weight:600}
.tab:hover{color:var(--txt)} .tab.active{color:#fff;border-bottom-color:var(--accent)}
.wrap{max-width:1080px;margin:0 auto;padding:22px 20px 60px}
.panel{background:linear-gradient(180deg,var(--panel) 0%,var(--panel2) 100%);
  border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:18px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);margin:0 0 12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px}
.card{background:#0e1521;border:1px solid var(--border);border-radius:12px;padding:12px 14px}
.card b{font-size:24px;display:block} .card span{color:var(--muted);font-size:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px} @media(max-width:760px){.grid2{grid-template-columns:1fr}}
label{display:block;font-size:12px;color:var(--muted);margin:0 0 6px;text-transform:uppercase;letter-spacing:.4px}
select,input[type=number],input[type=text]{width:100%;background:#0b1220;border:1px solid var(--border2);
  color:var(--txt);border-radius:8px;padding:9px 10px;font-size:14px}
.checkset{display:flex;flex-wrap:wrap;gap:8px}
.chk{display:flex;align-items:center;gap:7px;background:#0b1220;border:1px solid var(--border2);
  border-radius:8px;padding:7px 11px;cursor:pointer;user-select:none;font-size:13px}
.chk input{accent-color:var(--accent)}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:end;margin-bottom:14px}
.row>div{flex:1;min-width:160px}
.btn{background:var(--accent);border:0;color:#fff;border-radius:9px;padding:10px 16px;font-weight:700;
  cursor:pointer;font-size:14px} .btn:hover{background:var(--accent2)} .btn:disabled{opacity:.45;cursor:not-allowed}
.btn.warn{background:var(--err)} .btn.ghost{background:#1c2840} .btn.ok{background:var(--ok);color:#06210f}
.btnrow{display:flex;gap:10px;flex-wrap:wrap}
.muted{color:var(--muted)} .small{font-size:12px}
.progress{height:8px;background:#0b1220;border-radius:999px;overflow:hidden;border:1px solid var(--border)}
.progress>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),#22d3ee);width:0%}
pre.log{background:#06090f;border:1px solid var(--border);border-radius:10px;padding:12px;
  font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;max-height:300px;overflow:auto;white-space:pre-wrap}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--border)}
th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px;position:sticky;top:0;background:var(--panel)}
tr:hover td{background:#0e1726}
.tablewrap{max-height:520px;overflow:auto;border:1px solid var(--border);border-radius:10px}
.pill{padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600;border:1px solid var(--border2)}
.pill.ok{color:#06210f;background:var(--ok);border-color:var(--ok)}
.pill.bad{color:#fff;background:var(--err);border-color:var(--err)}
.pill.na{color:var(--muted)}
.banner{background:#3a2a0a;border:1px solid #6b4e16;color:#f7d99a;padding:10px 14px;border-radius:10px;margin-bottom:16px}
.hide{display:none}
.toast{position:fixed;right:18px;bottom:18px;background:#0e1726;border:1px solid var(--border2);
  padding:12px 16px;border-radius:10px;max-width:360px;box-shadow:0 8px 30px #0008}
.section-actions{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}
</style></head>
<body>
<div class="top">
  <div class="brand">⚽ FIFA WC2026 <small>Forecast Console</small></div>
  <div class="kpis" id="kpis"></div>
  <span id="jobBadge" class="badge b-idle">idle</span>
</div>
<div class="tabs" id="tabs">
  <div class="tab active" data-tab="overview">Overview</div>
  <div class="tab" data-tab="run">Run</div>
  <div class="tab" data-tab="results">Results</div>
  <div class="tab" data-tab="compare">Compare</div>
  <div class="tab" data-tab="downloads">Downloads</div>
</div>
<div class="wrap">
  <div id="roBanner" class="banner hide">Read-only mode — set <b>DASHBOARD_PASSWORD</b> to enable the Run / Fetch buttons.</div>

  <!-- OVERVIEW -->
  <section id="t-overview">
    <div class="panel"><h2>Dataset</h2><div class="cards" id="ovCards"></div></div>
    <div class="panel"><h2>Accuracy leaderboard (finished matches)</h2><div id="ovLeader" class="tablewrap"></div>
      <p class="muted small" id="ovNotes"></p></div>
    <div class="panel"><h2>Current job</h2>
      <div id="ovJobMeta" class="small muted" style="margin-bottom:8px"></div>
      <div class="progress"><i id="ovBar"></i></div>
      <pre class="log" id="ovLog">(no job yet)</pre>
    </div>
  </section>

  <!-- RUN -->
  <section id="t-run" class="hide">
    <div class="panel">
      <h2>Run a collection</h2>
      <div class="row">
        <div><label>Match</label><select id="runMatch"></select></div>
        <div style="max-width:130px"><label>Repetitions</label><input type="number" id="runReps" min="1" max="50" value="10"></div>
      </div>
      <div style="margin-bottom:14px"><label>Moments</label><div class="checkset" id="runMoments"></div></div>
      <div style="margin-bottom:14px"><label>Models</label><div class="checkset" id="runModels"></div>
        <div class="small muted" style="margin-top:6px"><a href="#" id="modelsAll">select all</a> · <a href="#" id="modelsNone">none</a></div></div>
      <div class="section-actions">
        <label class="chk"><input type="checkbox" id="runDry"> dry-run (free, mock)</label>
        <button class="btn warn" id="btnRun">▶ Run</button>
        <span class="small muted" id="runEstimate"></span>
      </div>
    </div>
    <div class="panel">
      <h2>Maintenance</h2>
      <div class="btnrow">
        <button class="btn ghost" data-fetch="results">⬇ Fetch results</button>
        <button class="btn ghost" data-fetch="fixtures">⬇ Fetch fixtures</button>
        <button class="btn ghost" data-fetch="both">⬇ Fetch both</button>
        <button class="btn ghost" id="btnReport">📊 Rebuild report</button>
      </div>
    </div>
    <div class="panel"><h2>Live job</h2>
      <div id="runJobMeta" class="small muted" style="margin-bottom:8px"></div>
      <div class="progress"><i id="runBar"></i></div>
      <pre class="log" id="runLog">(no job yet)</pre>
    </div>
  </section>

  <!-- RESULTS -->
  <section id="t-results" class="hide">
    <div class="panel"><h2>Actual results</h2><div id="resTable" class="tablewrap"></div></div>
  </section>

  <!-- COMPARE -->
  <section id="t-compare" class="hide">
    <div class="panel">
      <h2>Model forecasts vs actual result</h2>
      <div class="row"><div><label>Match</label><select id="cmpMatch"></select></div></div>
      <div id="cmpInfo" class="small" style="margin-bottom:12px"></div>
      <div id="cmpTable" class="tablewrap"></div>
    </div>
  </section>

  <!-- DOWNLOADS -->
  <section id="t-downloads" class="hide">
    <div class="panel"><h2>Excel workbooks</h2>
      <p><a href="./download/evaluation">⬇ Evaluation report (forecast vs results)</a></p>
      <p><a href="./download/forecasts">⬇ Full forecasts workbook</a></p>
      <p><a href="./download/report">⬇ Quality / variability report</a></p>
    </div>
  </section>
</div>
<div id="toast" class="toast hide"></div>

<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
let BOOT=null, POLL=null, lastStatus=null;
const fmtPct=v=>v==null?'–':(v+'%');
const esc=s=>String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function toast(msg,ms=3500){const t=$('#toast');t.textContent=msg;t.classList.remove('hide');
  clearTimeout(t._t);t._t=setTimeout(()=>t.classList.add('hide'),ms);}

function tab(name){
  $$('.tab').forEach(t=>t.classList.toggle('active',t.dataset.tab===name));
  ['overview','run','results','compare','downloads'].forEach(n=>$('#t-'+n).classList.toggle('hide',n!==name));
  if(name==='results')loadResults();
  if(name==='compare')loadCompare();
  if(name==='overview')loadEvaluate();
}
$$('.tab').forEach(t=>t.onclick=()=>tab(t.dataset.tab));

function renderKpis(s){
  const k=$('#kpis');
  if(!s||!s.db_exists){k.innerHTML='<div class="kpi"><b>0</b><span>no data</span></div>';return;}
  const items=[['total','runs'],['success','ok'],['error','errors'],['matches','matches'],['finished','finished']];
  k.innerHTML=items.map(([key,lab])=>`<div class="kpi"><b>${s[key]??0}</b><span>${lab}</span></div>`).join('');
}
function renderOvCards(s){
  const c=$('#ovCards');
  if(!s||!s.db_exists){c.innerHTML='<div class="card"><b>0</b><span>no database yet — run a collection</span></div>';return;}
  const items=[['total','executions'],['success','success'],['error','errors'],['valid','valid JSON'],
    ['matches','matches'],['models','models'],['results','results'],['finished','finished']];
  c.innerHTML=items.map(([k,l])=>`<div class="card"><b>${s[k]??0}</b><span>${l}</span></div>`).join('');
}

function table(rows,cols){
  if(!rows||!rows.length)return '<p class="muted small" style="padding:10px">No data.</p>';
  const head='<tr>'+cols.map(c=>`<th>${esc(c.label)}</th>`).join('')+'</tr>';
  const body=rows.map(r=>'<tr>'+cols.map(c=>`<td>${c.render?c.render(r[c.key],r):esc(r[c.key])}</td>`).join('')+'</tr>').join('');
  return `<table>${head}${body}</table>`;
}

async function api(path,opts){const r=await fetch(path,opts);if(!r.ok){throw new Error((await r.text())||r.status);}return r.json();}

async function boot(){
  BOOT=await api('./api/bootstrap');
  $('#roBanner').classList.toggle('hide',BOOT.actions_enabled);
  renderKpis(BOOT.summary); renderOvCards(BOOT.summary);
  buildRunForm();
  // match dropdowns
  const opt=m=>`<option value="${m.match_id}">${esc(m.label)}${m.finished?' ✓':''}</option>`;
  $('#runMatch').innerHTML='<option value="">— All matches —</option>'+BOOT.matches.map(opt).join('');
  const withPred=BOOT.matches.filter(m=>m.has_predictions);
  $('#cmpMatch').innerHTML=(withPred.length?withPred:BOOT.matches).map(opt).join('');
  loadEvaluate();
}

function buildRunForm(){
  $('#runReps').value=BOOT.reps_default||10;
  $('#runMoments').innerHTML=BOOT.moments.map(m=>{
    const on=BOOT.default_moments.includes(m)?'checked':'';
    return `<label class="chk"><input type="checkbox" value="${m}" ${on}> ${m}</label>`;}).join('');
  $('#runModels').innerHTML=BOOT.models.map(m=>
    `<label class="chk"><input type="checkbox" value="${m.key}" checked> ${esc(m.key)} <span class="muted small">(${esc(m.provider)})</span></label>`).join('');
  const act=BOOT.actions_enabled;
  $('#btnRun').disabled=!act;
  $$('[data-fetch]').forEach(b=>b.disabled=!act); $('#btnReport').disabled=!act;
  updateEstimate();
  $('#runMatch').onchange=$('#runReps').oninput=updateEstimate;
  $('#runModels').onchange=$('#runMoments').onchange=updateEstimate;
}
function selModels(){return $$('#runModels input:checked').map(i=>i.value);}
function selMoments(){return $$('#runMoments input:checked').map(i=>i.value);}
function updateEstimate(){
  const matches=$('#runMatch').value?1:BOOT.matches.length;
  const calls=matches*Math.max(selMoments().length,1)*Math.max(selModels().length,1)*BOOT.prompts.length*(+$('#runReps').value||1);
  $('#runEstimate').textContent=`≈ ${calls.toLocaleString()} API calls`;
}
$('#modelsAll').onclick=e=>{e.preventDefault();$$('#runModels input').forEach(i=>i.checked=true);updateEstimate();};
$('#modelsNone').onclick=e=>{e.preventDefault();$$('#runModels input').forEach(i=>i.checked=false);updateEstimate();};

$('#btnRun').onclick=async()=>{
  const body={match_id:$('#runMatch').value||null,moments:selMoments(),models:selModels(),
    reps:+$('#runReps').value||null,dry_run:$('#runDry').checked};
  if(!body.models.length)return toast('Select at least one model.');
  if(!body.dry_run && !confirm('This calls the paid LLM APIs and may cost money. Continue?'))return;
  try{const r=await api('./actions/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    toast(r.ok?('Started: '+r.command):('Busy: '+r.message));startPolling();tab('run');}
  catch(e){toast('Error: '+e.message);}
};
$$('[data-fetch]').forEach(b=>b.onclick=async()=>{
  try{const r=await api('./actions/fetch',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({mode:b.dataset.fetch,dry_run:false})});
    toast(r.ok?'Fetch started':('Busy: '+r.message));startPolling();}catch(e){toast('Error: '+e.message);}});
$('#btnReport').onclick=async()=>{try{const r=await api('./actions/report',{method:'POST'});
  toast(r.ok?'Report rebuild started':('Busy: '+r.message));startPolling();}catch(e){toast('Error: '+e.message);}};

function renderJob(j){
  const badge=$('#jobBadge');badge.className='badge b-'+(j.status||'idle');badge.textContent=j.status||'idle';
  const meta=`${j.command?('<b>'+esc(j.command)+'</b> · '):''}${j.status||'idle'}${j.returncode!=null?(' · rc='+j.returncode):''}${j.started_at?(' · '+j.started_at):''}`;
  $('#ovJobMeta').innerHTML=meta; $('#runJobMeta').innerHTML=meta;
  const pct=j.progress?Math.round(100*j.progress.done/Math.max(j.progress.total,1)):(j.status==='finished'?100:0);
  $('#ovBar').style.width=pct+'%'; $('#runBar').style.width=pct+'%';
  const log=j.log||'(no job yet)'; $('#ovLog').textContent=log; $('#runLog').textContent=log;
  $('#ovLog').scrollTop=$('#ovLog').scrollHeight; $('#runLog').scrollTop=$('#runLog').scrollHeight;
}
async function pollJob(){
  try{const j=await api('./api/job');renderJob(j);
    if(j.status==='running'){/*keep polling*/}
    else{ if(lastStatus==='running'){ // just finished
        refreshSummary();loadEvaluate(); if(!$('#t-results').classList.contains('hide'))loadResults();}
      stopPolling();}
    lastStatus=j.status;
  }catch(e){/* ignore transient */}
}
function startPolling(){lastStatus='running';renderJob({status:'running',command:'(starting)'});if(POLL)clearInterval(POLL);POLL=setInterval(pollJob,2000);pollJob();}
function stopPolling(){if(POLL){clearInterval(POLL);POLL=null;}}

async function refreshSummary(){try{const s=await api('./api/summary');BOOT.summary=s;renderKpis(s);renderOvCards(s);}catch(e){}}

async function loadEvaluate(){
  try{const d=await api('./api/evaluate');
    $('#ovNotes').textContent=(d.notes||[]).join('  ·  ');
    $('#ovLeader').innerHTML=table(d.leaderboard,[
      {key:'model',label:'Model'},{key:'provider',label:'Provider'},{key:'n',label:'Preds'},
      {key:'outcome_acc',label:'Winner %',render:v=>fmtPct(v)},
      {key:'exact_acc',label:'Exact %',render:v=>fmtPct(v)},
      {key:'mean_abs_gd_err',label:'GD err'},{key:'mean_brier',label:'Brier'}]);
  }catch(e){$('#ovLeader').innerHTML='<p class="muted small" style="padding:10px">'+esc(e.message)+'</p>';}
}
async function loadResults(){
  try{const d=await api('./api/results');
    $('#resTable').innerHTML=table(d.results,[
      {key:'match_id',label:'#'},{key:'team_1',label:'Home'},{key:'team_2',label:'Away'},
      {key:'actual_score_team_1',label:'H'},{key:'actual_score_team_2',label:'A'},
      {key:'actual_winner',label:'Winner'},{key:'status',label:'Status'},{key:'source',label:'Source'}]);
  }catch(e){$('#resTable').innerHTML='<p class="muted small" style="padding:10px">'+esc(e.message)+'</p>';}
}
async function loadCompare(){
  const sel=$('#cmpMatch'); if(!sel.value && sel.options.length)sel.value=sel.options[0].value;
  if(!sel.value){$('#cmpTable').innerHTML='<p class="muted small" style="padding:10px">No matches with predictions yet.</p>';return;}
  sel.onchange=loadCompare;
  try{const d=await api('./api/compare?match_id='+encodeURIComponent(sel.value));
    const i=d.info||{};
    $('#cmpInfo').innerHTML=`<b>${esc(i.team_1)} vs ${esc(i.team_2)}</b> — `+
      (i.actual_score?`actual <b>${esc(i.actual_score)}</b> (winner: ${esc(i.actual_winner)}) <span class="pill ok">finished</span>`
        :'<span class="pill na">no result yet</span>');
    const win=v=>v==null?'<span class="pill na">–</span>':(v?'<span class="pill ok">✓</span>':'<span class="pill bad">✗</span>');
    $('#cmpTable').innerHTML=table(d.rows,[
      {key:'model',label:'Model'},{key:'prompt_id',label:'Prompt'},{key:'moment',label:'Moment'},
      {key:'reps',label:'n'},{key:'modal_score',label:'Predicted'},
      {key:'winner_agreement',label:'Agree',render:v=>Math.round(v*100)+'%'},
      {key:'modal_winner',label:'Pred winner'},
      {key:'outcome_correct',label:'Winner?',render:win},
      {key:'exact_correct',label:'Exact?',render:win}]);
  }catch(e){$('#cmpTable').innerHTML='<p class="muted small" style="padding:10px">'+esc(e.message)+'</p>';}
}

boot().then(()=>{ // resume polling if a job is already running
  fetch('./api/job').then(r=>r.json()).then(j=>{if(j.status==='running')startPolling();else renderJob(j);});
}).catch(e=>toast('Load error: '+e.message,8000));
</script>
</body></html>
"""
