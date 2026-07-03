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
import tempfile
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from starlette.background import BackgroundTask

from . import __version__
from . import config as cfg
from . import prompts as prompt_lib
from .evaluation import build_evaluation, per_match_comparison, write_evaluation_excel
from .export import export_workbook
from .matches import load_matches
from .storage import Database

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
    runs: dict[str, tuple[int, int]] = {}   # match_id -> (n_runs, n_models)
    finished_ids: set[str] = set()
    if path.exists():
        con = sqlite3.connect(path)
        try:
            for mid, n, nm in con.execute(
                "SELECT match_id, COUNT(*), COUNT(DISTINCT model) "
                "FROM forecast_runs GROUP BY match_id"
            ):
                runs[mid] = (int(n), int(nm))
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
        n_runs, n_models = runs.get(m.match_id, (0, 0))
        out.append({
            "match_id": m.match_id, "team_1": m.team_1, "team_2": m.team_2,
            "local_date": m.local_date, "phase": m.phase,
            "has_predictions": n_runs > 0,
            "n_runs": n_runs, "n_models": n_models,
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
        {"key": m["key"], "provider": m.get("provider"), "model_id": m.get("model_id"),
         "reps": config.reps_for(m["key"])}
        for m in config.enabled_models()
    ]
    return JSONResponse({
        "version": __version__,
        "actions_enabled": bool(_DASHBOARD_PASSWORD),
        "summary": _summary(),
        "moments": list(prompt_lib.MATCH_MOMENTS),
        "default_moments": list(config.match_moments),
        "reps_default": config.runs_per_combination,
        "orders": len(config.team_order_types),
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


def _eval_frame():
    """Canonical per-prediction DataFrame (finished matches only), reusing the
    evaluation module's join + canonicalization. Empty frame when nothing joins."""
    from .evaluation import load_joined, _canonicalize

    joined = load_joined(_db_path())
    if joined.empty:
        return joined
    for col, default in (("match_moment", None), ("team_order_type", "original")):
        if col not in joined.columns:
            joined[col] = default
    return _canonicalize(joined)


_BOX_METRICS = ["abs_goal_diff_error", "total_goals_error", "brier"]
_NO_EVAL = "No finished results joined to valid predictions yet — ingest results first."


@app.get("/api/metrics/box")
def api_metrics_box(moment: str = "all", _: None = Depends(require_view)) -> JSONResponse:
    """Five-number summaries (Tukey box + outliers) per model×prompt, per metric."""
    import numpy as np
    import pandas as pd

    df = _eval_frame()
    if df is None or df.empty:
        return JSONResponse({"groups": [], "metrics": _BOX_METRICS, "note": _NO_EVAL})
    if moment and moment != "all" and "match_moment" in df.columns:
        df = df[df["match_moment"] == moment]

    def box(vals) -> dict | None:
        v = pd.to_numeric(vals, errors="coerce").dropna().to_numpy()
        if v.size == 0:
            return None
        q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        inl = v[(v >= lo) & (v <= hi)]
        outl = sorted(float(x) for x in v[(v < lo) | (v > hi)])
        return {
            "n": int(v.size), "min": float(v.min()), "q1": q1, "med": med, "q3": q3,
            "max": float(v.max()), "mean": float(v.mean()),
            "whislo": float(inl.min()) if inl.size else float(v.min()),
            "whishi": float(inl.max()) if inl.size else float(v.max()),
            "outliers": outl[:60],
        }

    groups = []
    for (model, prompt), g in df.groupby(["model", "prompt_id"], dropna=False):
        entry = {"model": str(model), "prompt_id": str(prompt), "n": int(len(g))}
        for m in _BOX_METRICS:
            entry[m] = box(g[m]) if m in g.columns else None
        groups.append(entry)
    groups.sort(key=lambda e: (e["model"], e["prompt_id"]))
    return JSONResponse({"groups": groups, "metrics": _BOX_METRICS, "moment": moment, "note": ""})


@app.get("/api/race")
def api_race(moment: str = "pre_match", _: None = Depends(require_view)) -> JSONResponse:
    """Cumulative outcome accuracy per model across matches ordered by kickoff.

    For each match the model's modal predicted winner is scored against the actual
    outcome; the series is the running fraction correct (step-carried across matches
    the model didn't cover)."""
    from collections import Counter

    import pandas as pd

    df = _eval_frame()
    if df is None or df.empty:
        return JSONResponse({"labels": [], "series": {}, "note": _NO_EVAL})
    if moment and moment != "all" and "match_moment" in df.columns:
        sub = df[df["match_moment"] == moment]
        df = sub if not sub.empty else df

    order = (
        df.groupby("match_id")["kickoff_datetime"].min().sort_values().index.tolist()
    )
    labels, meta = [], {}
    for i, mid in enumerate(order, start=1):
        g = df[df["match_id"] == mid]
        r = g.iloc[0]
        a1 = pd.to_numeric(g["actual_c1"], errors="coerce").dropna()
        a2 = pd.to_numeric(g["actual_c2"], errors="coerce").dropna()
        actual = f"{int(a1.iloc[0])}-{int(a2.iloc[0])}" if len(a1) and len(a2) else None
        meta[mid] = {
            "name": f"{r['team_1']} v {r['team_2']}",
            "date": str(r.get("kickoff_datetime") or "")[:10],
            "actual": actual,
        }
        labels.append({"i": i, "match_id": str(mid), **meta[mid]})

    series: dict[str, list] = {}
    for model in sorted(x for x in df["model"].dropna().unique().tolist()):
        md = df[df["model"] == model]
        per_match: dict[str, int] = {}
        for mid, g in md.groupby("match_id"):
            outs = [o for o in g["pred_outcome"].tolist() if isinstance(o, str)]
            if not outs:
                continue
            modal = Counter(outs).most_common(1)[0][0]
            per_match[mid] = int(modal == g["actual_outcome"].iloc[0])
        pts, correct, total, last = [], 0, 0, None
        for mid in order:
            if mid in per_match:
                total += 1
                correct += per_match[mid]
                last = correct / total
                pts.append({"acc": round(last, 4), "c": per_match[mid]})
            else:
                pts.append({"acc": round(last, 4) if last is not None else None, "c": None})
        series[model] = pts
    return JSONResponse({"labels": labels, "series": series, "moment": moment, "note": ""})


_FORECAST_COLS = [
    "run_id", "match_id", "team_1", "team_2", "provider", "model", "prompt_id",
    "match_moment", "team_order_type", "repetition_number",
    "parsed_score_team_1", "parsed_score_team_2",
    "parsed_team1_win_probability", "parsed_draw_probability", "parsed_team2_win_probability",
    "json_valid", "execution_status", "latency_ms", "total_tokens", "api_cost",
]


@app.get("/api/forecasts")
def api_forecasts(
    match_id: str | None = None, model: str | None = None, prompt_id: str | None = None,
    moment: str | None = None, order: str | None = None, status: str | None = None,
    valid: str | None = None, q: str | None = None, limit: int = 50, offset: int = 0,
    _: None = Depends(require_view),
) -> JSONResponse:
    """Filtered, paginated view of the forecast_runs table."""
    path = _db_path()
    if not path.exists():
        return JSONResponse({"total": 0, "rows": [], "limit": limit, "offset": offset})
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(forecast_runs)")}
        where: list[str] = []
        params: list[Any] = []

        def add(cond: str, val: Any) -> None:
            where.append(cond)
            params.append(val)

        if match_id:
            add("match_id = ?", match_id)
        if model:
            add("model = ?", model)
        if prompt_id:
            add("prompt_id = ?", prompt_id)
        if moment and "match_moment" in cols:
            add("match_moment = ?", moment)
        if order:
            add("team_order_type = ?", order)
        if status:
            add("execution_status = ?", status)
        if valid in ("0", "1"):
            add("json_valid = ?", int(valid))
        if q:
            where.append("(team_1 LIKE ? OR team_2 LIKE ? OR model LIKE ? OR prompt_id LIKE ?)")
            like = f"%{q}%"
            params += [like, like, like, like]
        wsql = (" WHERE " + " AND ".join(where)) if where else ""

        total = con.execute(f"SELECT COUNT(*) FROM forecast_runs{wsql}", params).fetchone()[0]
        sel = [c for c in _FORECAST_COLS if c in cols]
        lim = max(1, min(int(limit), 500))
        off = max(0, int(offset))
        rows = con.execute(
            f"SELECT {', '.join(sel)} FROM forecast_runs{wsql} "
            "ORDER BY CAST(match_id AS INTEGER), run_id LIMIT ? OFFSET ?",
            params + [lim, off],
        ).fetchall()
        return JSONResponse(_jsonable({
            "total": int(total), "limit": lim, "offset": off,
            "rows": [dict(r) for r in rows],
        }))
    finally:
        con.close()


@app.get("/api/forecast/{run_id}")
def api_forecast_detail(run_id: str, _: None = Depends(require_view)) -> JSONResponse:
    """Full row (incl. prompt text, raw response, six hats) for one execution."""
    path = _db_path()
    if not path.exists():
        raise HTTPException(404, "no database")
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        r = con.execute("SELECT * FROM forecast_runs WHERE run_id = ?", (run_id,)).fetchone()
    finally:
        con.close()
    if not r:
        raise HTTPException(404, "not found")
    return JSONResponse(_jsonable(dict(r)))


# --------------------------------------------------------------------------- #
# Actions (JSON body)
# --------------------------------------------------------------------------- #
class RunReq(BaseModel):
    match_id: str | None = None     # single, kept for backward compatibility
    match_ids: list[str] = []       # run several matches at once ([] = all)
    moments: list[str] = []
    models: list[str] = []
    reps: int | None = None         # override for every model (None = per-model)
    retry_errors: bool = False      # re-run errored + missing; keep successes
    dry_run: bool = False


class FetchReq(BaseModel):
    mode: str = "results"   # results | fixtures | both
    dry_run: bool = False


@app.post("/actions/run")
def action_run(req: RunReq, _: None = Depends(require_action)) -> JSONResponse:
    args = ["run", "--no-export"]
    ids = list(req.match_ids)
    if req.match_id and req.match_id not in ids:
        ids.append(req.match_id)
    for mid in ids:
        args += ["--match-id", mid]
    for m in req.moments:
        if m in prompt_lib.MATCH_MOMENTS:
            args += ["--moment", m]
    for k in req.models:
        args += ["--model", k]
    if req.reps:
        args += ["--reps", str(req.reps)]
    if req.retry_errors:
        args.append("--retry-errors")
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


@app.post("/actions/merge-upload")
async def action_merge_upload(
    file: UploadFile = File(...),
    replace: bool = Form(False),
    _: None = Depends(require_action),
) -> JSONResponse:
    """Merge a forecast .db uploaded from another machine into the server DB.

    This is how locally-run models (Ollama on your PC) reach the Coolify DB:
    run the collection locally into a scratch .db, then upload it here. Rows are
    added by deterministic run_id (INSERT OR IGNORE), so cloud rows are never
    clobbered; ``replace=true`` overwrites same-id rows instead.
    """
    up_dir = cfg.DATA_DIR / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    tmp = up_dir / f"merge_{stamp}.db"
    data = await file.read()
    tmp.write_bytes(data)
    try:
        if data[:16] != b"SQLite format 3\x00":
            raise HTTPException(400, "Uploaded file is not a SQLite database.")
        db = Database(_db_path())
        try:
            stats = db.merge_from(tmp, replace=replace)
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the UI
        raise HTTPException(400, f"Merge failed: {exc}")
    finally:
        try:
            tmp.unlink()
        except OSError:  # pragma: no cover
            pass
    return JSONResponse({"ok": True, "filename": file.filename, **stats})


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


@app.get("/download/archive")
def download_archive(_: None = Depends(require_view)) -> FileResponse:
    """Zip the SQLite DB (consistent snapshot) plus every raw JSON artifact.

    The DB is copied via SQLite's online backup API so it's safe to download
    even while a collection job is writing. The zip is streamed and deleted
    afterwards. JSON lives under data/{raw_requests,raw_responses,traces,
    metadata}; the manifest sits at the repo root.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fd, tmp_zip = tempfile.mkstemp(suffix=".zip", dir=str(cfg.DATA_DIR))
    os.close(fd)
    zip_path = Path(tmp_zip)

    db_path = _db_path()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # DB: online-backup snapshot into a temp file, then add to the zip.
        if db_path.exists():
            fd2, tmp_db = tempfile.mkstemp(suffix=".db", dir=str(cfg.DATA_DIR))
            os.close(fd2)
            try:
                src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                dst = sqlite3.connect(tmp_db)
                with dst:
                    src.backup(dst)
                dst.close()
                src.close()
                zf.write(tmp_db, arcname=db_path.name)
            finally:
                try:
                    os.unlink(tmp_db)
                except OSError:  # pragma: no cover
                    pass
        # All raw JSON artifacts, preserving their folder names in the zip.
        for d in (
            cfg.RAW_REQUESTS_DIR,
            cfg.RAW_RESPONSES_DIR,
            cfg.TRACES_DIR,
            cfg.METADATA_DIR,
        ):
            if d.exists():
                for f in sorted(d.glob("*.json")):
                    zf.write(f, arcname=f"{d.name}/{f.name}")
        # Experiment manifest (JSON at the repo root).
        manifest = cfg.ROOT / "experiment_manifest.json"
        if manifest.exists():
            zf.write(manifest, arcname=manifest.name)

    return FileResponse(
        zip_path,
        filename=f"fifa_forecast_backup_{stamp}.zip",
        media_type="application/zip",
        background=BackgroundTask(lambda: zip_path.unlink(missing_ok=True)),
    )


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
.hide{display:none!important}
.toast{position:fixed;right:18px;bottom:18px;background:#0e1726;border:1px solid var(--border2);
  padding:12px 16px;border-radius:10px;max-width:360px;box-shadow:0 8px 30px #0008}
.section-actions{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}
.picklist{max-height:220px;overflow:auto;border:1px solid var(--border2);border-radius:8px;padding:8px;
  display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:4px}
.clickrow td{cursor:pointer} .clickrow:hover td{background:#16233b}
.modal{position:fixed;inset:0;background:#000a;display:flex;align-items:flex-start;justify-content:center;
  padding:40px 16px;z-index:50;overflow:auto}
.modalbox{background:var(--panel);border:1px solid var(--border2);border-radius:14px;max-width:860px;width:100%;padding:20px}
.kv{display:grid;grid-template-columns:170px 1fr;gap:5px 14px;font-size:13px;margin:8px 0}
.kv .k{color:var(--muted)} .x{float:right;cursor:pointer;color:var(--muted);font-size:18px}
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
  <div class="tab" data-tab="forecasts">Forecasts</div>
  <div class="tab" data-tab="analytics">Analytics</div>
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
      <div style="margin-bottom:14px">
        <label>Matches — check one or many (none = all)</label>
        <div class="small" id="matchCoverage" style="margin-bottom:8px"></div>
        <input type="text" id="runMatchSearch" placeholder="filter by team, date or phase…" style="margin-bottom:8px">
        <label class="chk" style="display:inline-flex;margin-bottom:8px"><input type="checkbox" id="hideRun"> show only not-run yet</label>
        <div id="runMatchList" class="picklist"></div>
        <div class="small muted" style="margin-top:6px">
          <a href="#" id="matchAll">all</a> · <a href="#" id="matchNone">none</a> ·
          <a href="#" id="matchNotRun">select not-run</a> · <a href="#" id="matchShown">select shown</a> · <b id="matchCount"></b></div>
      </div>
      <div style="margin-bottom:14px"><label>Moments</label><div class="checkset" id="runMoments"></div></div>
      <div style="margin-bottom:14px"><label>Models</label><div class="checkset" id="runModels"></div>
        <div class="small muted" style="margin-top:6px"><a href="#" id="modelsAll">select all</a> · <a href="#" id="modelsNone">none</a></div></div>
      <div class="row"><div style="max-width:220px"><label>Reps override (blank = per-model)</label><input type="number" id="runReps" min="1" max="50" placeholder="per model"></div></div>
      <label class="chk" style="display:inline-flex;margin-bottom:10px"><input type="checkbox" id="runRetry"> only re-run failed &amp; missing (keep what already succeeded)</label>
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

  <!-- FORECASTS -->
  <section id="t-forecasts" class="hide">
    <div class="panel">
      <h2>Forecast runs — filterable table</h2>
      <div class="row">
        <div><label>Match</label><select id="fxMatch"></select></div>
        <div><label>Model</label><select id="fxModel"></select></div>
        <div><label>Prompt</label><select id="fxPrompt"></select></div>
        <div><label>Moment</label><select id="fxMoment"></select></div>
      </div>
      <div class="row">
        <div><label>Order</label><select id="fxOrder"><option value="">any</option><option>original</option><option>reversed</option></select></div>
        <div><label>Status</label><select id="fxStatus"><option value="">any</option><option>success</option><option>error</option></select></div>
        <div><label>JSON</label><select id="fxValid"><option value="">any</option><option value="1">valid</option><option value="0">invalid</option></select></div>
        <div><label>Search</label><input type="text" id="fxQ" placeholder="team / model…"></div>
      </div>
      <div class="section-actions" style="margin-bottom:10px">
        <button class="btn ghost" id="fxApply">Apply</button>
        <button class="btn ghost" id="fxReset">Reset</button>
        <span class="small muted" id="fxCount"></span>
        <span class="small muted" style="margin-left:auto">click a row → prompt + raw response</span>
      </div>
      <div id="fxTable" class="tablewrap"></div>
      <div class="section-actions">
        <button class="btn ghost" id="fxPrev">‹ Prev</button>
        <span class="small muted" id="fxPage"></span>
        <button class="btn ghost" id="fxNext">Next ›</button>
      </div>
    </div>
  </section>

  <!-- ANALYTICS -->
  <section id="t-analytics" class="hide">
    <div class="panel">
      <h2>Metric distributions — boxplots by model &amp; prompt</h2>
      <div class="row">
        <div><label>Metric</label><select id="boxMetric">
          <option value="abs_goal_diff_error">Goal-difference error</option>
          <option value="total_goals_error">Total-goals error</option>
          <option value="brier">Brier (probability prompts)</option>
        </select></div>
        <div><label>Moment</label><select id="anMoment">
          <option value="all">all</option><option value="pre_match">pre_match</option>
          <option value="halftime">halftime</option><option value="post_match">post_match</option>
        </select></div>
        <div style="align-self:end"><button class="btn ghost" id="boxReload">Reload</button></div>
      </div>
      <p class="small muted">One box per model×prompt over finished matches (all reps/orders, canonical order). Lower is better; red dots are outliers.</p>
      <div id="boxWrap" class="tablewrap"></div>
    </div>
    <div class="panel">
      <h2>Accuracy race — cumulative outcome accuracy over time</h2>
      <div class="row">
        <div><label>Moment</label><select id="raceMoment">
          <option value="pre_match">pre_match</option><option value="all">all</option>
          <option value="halftime">halftime</option><option value="post_match">post_match</option>
        </select></div>
        <div style="align-self:end"><button class="btn ghost" id="raceReload">Reload</button></div>
      </div>
      <p class="small muted">Matches ordered by kickoff. Each match scores the model's modal predicted winner; the line is the running % correct. Hit play to watch models rise and fall.</p>
      <div id="raceWrap"></div>
    </div>
  </section>

  <!-- DOWNLOADS -->
  <section id="t-downloads" class="hide">
    <div class="panel"><h2>Excel workbooks</h2>
      <p><a href="./download/evaluation">⬇ Evaluation report (forecast vs results)</a></p>
      <p><a href="./download/forecasts">⬇ Full forecasts workbook</a></p>
      <p><a href="./download/report">⬇ Quality / variability report</a></p>
    </div>
    <div class="panel"><h2>Full backup (.zip)</h2>
      <p class="muted small">The complete raw dataset in one archive: the SQLite
        database plus every JSON artifact (requests, responses, traces, metadata)
        and the manifest. The DB is a consistent snapshot — safe to grab mid-run.</p>
      <p><a href="./download/archive">⬇ Download database + all JSON (zip)</a></p>
    </div>
    <div class="panel"><h2>Import local runs</h2>
      <p class="muted small">Ran open models (Qwen / Mistral / Gemma) on your own machine?
        Upload the resulting <code>.db</code> file to merge those forecasts into this
        server's database. Existing rows are kept (matched by run_id); tick replace to
        overwrite same-id rows.</p>
      <p><input type="file" id="mergeFile" accept=".db,.sqlite,application/octet-stream"></p>
      <label class="small"><input type="checkbox" id="mergeReplace"> replace rows with the same run_id</label>
      <p><button class="btn" id="btnMerge">⬆ Upload &amp; merge</button></p>
      <div id="mergeMsg" class="muted small"></div>
    </div>
  </section>
</div>
<div id="fxModal" class="modal hide"><div class="modalbox" id="fxModalBox"></div></div>
<div id="toast" class="toast hide"></div>

<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
let BOOT=null, POLL=null, lastStatus=null;
let ANALYTICS={box:null,race:null}, RACE=null;
const PALETTE=['#3b82f6','#22c55e','#f59e0b','#ef4444','#a855f7','#06b6d4','#ec4899','#84cc16','#f97316','#14b8a6','#eab308','#8b5cf6','#64748b','#f43f5e'];
const pcolor=p=>({'simple-prediction':'#06b6d4','probability-prediction':'#22c55e','six-hats-prediction':'#f59e0b'}[p]||'#3b82f6');
const fmtPct=v=>v==null?'–':(v+'%');
const esc=s=>String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function toast(msg,ms=3500){const t=$('#toast');t.textContent=msg;t.classList.remove('hide');
  clearTimeout(t._t);t._t=setTimeout(()=>t.classList.add('hide'),ms);}

function tab(name){
  $$('.tab').forEach(t=>t.classList.toggle('active',t.dataset.tab===name));
  ['overview','run','results','compare','forecasts','analytics','downloads'].forEach(n=>$('#t-'+n).classList.toggle('hide',n!==name));
  if(name==='results')loadResults();
  if(name==='compare')loadCompare();
  if(name==='forecasts')loadForecasts();
  if(name==='overview')loadEvaluate();
  if(name==='analytics')loadAnalytics();
}
$$('.tab').forEach(t=>t.onclick=()=>tab(t.dataset.tab));

// ---- Analytics: boxplots + accuracy race -------------------------------
async function loadAnalyticsBox(){
  try{const m=$('#anMoment').value||'all';
    ANALYTICS.box=await api('./api/metrics/box?moment='+encodeURIComponent(m));renderBox();}
  catch(e){$('#boxWrap').innerHTML='<div class="small muted">Error: '+esc(e.message)+'</div>';}
}
async function loadAnalyticsRace(){
  try{const m=$('#raceMoment').value||'pre_match';
    ANALYTICS.race=await api('./api/race?moment='+encodeURIComponent(m));renderRace();}
  catch(e){$('#raceWrap').innerHTML='<div class="small muted">Error: '+esc(e.message)+'</div>';}
}
function loadAnalytics(){loadAnalyticsBox();loadAnalyticsRace();}

function renderBox(){
  const data=ANALYTICS.box, wrap=$('#boxWrap'); if(!data){return;}
  const metric=$('#boxMetric').value;
  const groups=data.groups.filter(g=>g[metric]).map(g=>({model:g.model,prompt:g.prompt_id,b:g[metric]}));
  if(!groups.length){wrap.innerHTML='<div class="small muted">'+esc(data.note||'No data for this metric yet (needs finished results; Brier needs probability prompts).')+'</div>';return;}
  let lo=Infinity,hi=-Infinity;
  groups.forEach(g=>{const b=g.b;lo=Math.min(lo,b.whislo,...(b.outliers||[]));hi=Math.max(hi,b.whishi,...(b.outliers||[]));});
  if(!(hi>lo))hi=lo+1;
  groups.sort((a,b)=>a.b.med-b.b.med);
  const rowH=24,padL=230,padR=24,padT=24,W=Math.min(920,(wrap.clientWidth||840)),H=padT+groups.length*rowH+22;
  const x=v=>padL+(v-lo)/(hi-lo)*(W-padL-padR);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%" style="font:11px system-ui">`;
  for(let t=0;t<=4;t++){const v=lo+(hi-lo)*t/4,xx=x(v);
    s+=`<line x1="${xx}" y1="${padT-6}" x2="${xx}" y2="${H-18}" stroke="var(--border)"/>`;
    s+=`<text x="${xx}" y="${H-5}" fill="var(--muted)" text-anchor="middle">${v.toFixed(2)}</text>`;}
  groups.forEach((g,i)=>{const cy=padT+i*rowH+rowH/2,b=g.b,col=pcolor(g.prompt);
    s+=`<line x1="${x(b.whislo)}" y1="${cy}" x2="${x(b.whishi)}" y2="${cy}" stroke="var(--muted)"/>`;
    s+=`<line x1="${x(b.whislo)}" y1="${cy-5}" x2="${x(b.whislo)}" y2="${cy+5}" stroke="var(--muted)"/>`;
    s+=`<line x1="${x(b.whishi)}" y1="${cy-5}" x2="${x(b.whishi)}" y2="${cy+5}" stroke="var(--muted)"/>`;
    s+=`<rect x="${x(b.q1)}" y="${cy-8}" width="${Math.max(1,x(b.q3)-x(b.q1))}" height="16" fill="${col}22" stroke="${col}" stroke-width="1.5"/>`;
    s+=`<line x1="${x(b.med)}" y1="${cy-8}" x2="${x(b.med)}" y2="${cy+8}" stroke="${col}" stroke-width="2"/>`;
    (b.outliers||[]).forEach(o=>{s+=`<circle cx="${x(o)}" cy="${cy}" r="2" fill="var(--err)" opacity="0.55"/>`;});
    s+=`<title></title>`;
    s+=`<text x="${padL-8}" y="${cy+3}" fill="var(--txt)" text-anchor="end">${esc(g.model)} · <tspan fill="${col}">${esc(g.prompt.replace('-prediction',''))}</tspan> <tspan fill="var(--muted)">n=${b.n}</tspan></text>`;});
  s+='</svg>';wrap.innerHTML=s;
}

function renderRace(){
  const data=ANALYTICS.race, wrap=$('#raceWrap');
  if(!data||!data.labels.length){wrap.innerHTML='<div class="small muted">'+esc((data&&data.note)||'No finished results yet.')+'</div>';return;}
  const models=Object.keys(data.series), N=data.labels.length;
  wrap.innerHTML=`<div class="section-actions" style="margin-bottom:8px">
      <button class="btn ghost" id="racePlay">▶ Play</button>
      <input type="range" id="raceScrub" min="1" max="${N}" value="${N}" style="flex:1;min-width:160px">
      <span class="small muted" id="raceAt"></span></div>
    <div style="display:flex;gap:14px;flex-wrap:wrap">
      <div style="flex:1;min-width:320px" id="raceSvg"></div>
      <div id="raceLegend" class="small" style="min-width:190px"></div></div>`;
  RACE={data,models,N,k:N,timer:null};
  $('#raceScrub').oninput=e=>{RACE.k=+e.target.value;drawRace();};
  $('#racePlay').onclick=toggleRacePlay;
  drawRace();
}
function drawRace(){
  const {data,models,N,k}=RACE;
  const W=760,H=340,padL=42,padR=12,padT=14,padB=24;
  const x=i=>padL+(N<=1?0:(i-1)/(N-1))*(W-padL-padR), y=a=>padT+(1-a)*(H-padT-padB);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%" style="font:11px system-ui">`;
  for(let t=0;t<=4;t++){const a=t/4,yy=y(a);
    s+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="var(--border)"/>`;
    s+=`<text x="${padL-6}" y="${yy+3}" fill="var(--muted)" text-anchor="end">${(a*100)|0}%</text>`;}
  s+=`<line x1="${x(k)}" y1="${padT}" x2="${x(k)}" y2="${H-padB}" stroke="var(--accent)" stroke-dasharray="3 3" opacity="0.6"/>`;
  const rank=[];
  models.forEach((m,mi)=>{const pts=data.series[m],col=PALETTE[mi%PALETTE.length];
    let d='',on=false,ly=null,la=null;
    for(let i=1;i<=k;i++){const p=pts[i-1];if(!p||p.acc==null)continue;
      const px=x(i),py=y(p.acc);d+=(on?'L':'M')+px.toFixed(1)+' '+py.toFixed(1)+' ';on=true;ly=py;la=p.acc;}
    if(d){s+=`<path d="${d}" fill="none" stroke="${col}" stroke-width="2" opacity="0.9"/>`;
      s+=`<circle cx="${x(k)}" cy="${ly}" r="3" fill="${col}"/>`;rank.push({m,acc:la,col});}});
  s+='</svg>';$('#raceSvg').innerHTML=s;
  const lab=data.labels[k-1];
  $('#raceAt').textContent=`#${k}/${N} · ${lab.date} · ${lab.name}`+(lab.actual?` (${lab.actual})`:'');
  rank.sort((a,b)=>b.acc-a.acc);
  $('#raceLegend').innerHTML='<b>Ranking @ '+k+'</b>'+rank.map((r,ix)=>
    `<div style="display:flex;align-items:center;gap:6px;margin-top:3px">
      <span style="width:14px;color:var(--muted)">${ix+1}</span>
      <span style="width:10px;height:10px;border-radius:2px;background:${r.col};display:inline-block"></span>
      <span style="flex:1">${esc(r.m)}</span><b>${(r.acc*100).toFixed(0)}%</b></div>`).join('');
}
function toggleRacePlay(){const btn=$('#racePlay');
  if(RACE.timer){clearInterval(RACE.timer);RACE.timer=null;btn.textContent='▶ Play';return;}
  if(RACE.k>=RACE.N)RACE.k=1;
  btn.textContent='⏸ Pause';
  RACE.timer=setInterval(()=>{RACE.k++;$('#raceScrub').value=RACE.k;drawRace();
    if(RACE.k>=RACE.N){clearInterval(RACE.timer);RACE.timer=null;btn.textContent='▶ Play';}},260);
}
$('#boxMetric').onchange=renderBox;
$('#anMoment').onchange=loadAnalyticsBox;
$('#boxReload').onclick=loadAnalyticsBox;
$('#raceMoment').onchange=loadAnalyticsRace;
$('#raceReload').onclick=loadAnalyticsRace;

function renderKpis(s){
  const k=$('#kpis');
  if(!s||!s.db_exists){k.innerHTML='<div class="kpi"><b>0</b><span>no data</span></div>';return;}
  const tot=(BOOT&&BOOT.matches)?BOOT.matches.length:0;
  let html=[['total','runs'],['success','ok'],['error','errors']]
    .map(([key,lab])=>`<div class="kpi"><b>${s[key]??0}</b><span>${lab}</span></div>`).join('');
  if(tot)html+=`<div class="kpi"><b style="color:var(--ok)">${s.matches??0}</b><span>matches run</span></div>`
            +`<div class="kpi"><b style="color:var(--warn)">${tot-(s.matches||0)}</b><span>to run</span></div>`;
  k.innerHTML=html;
}
function renderOvCards(s){
  const c=$('#ovCards');
  if(!s||!s.db_exists){c.innerHTML='<div class="card"><b>0</b><span>no database yet — run a collection</span></div>';return;}
  const tot=(BOOT&&BOOT.matches)?BOOT.matches.length:0;
  const items=[['total','executions'],['success','success'],['error','errors'],['valid','valid JSON'],
    ['models','models'],['results','results'],['finished','finished']];
  let html=items.map(([k,l])=>`<div class="card"><b>${s[k]??0}</b><span>${l}</span></div>`).join('');
  if(tot)html=`<div class="card"><b>${s.matches??0} / ${tot}</b><span>matches run</span></div>`
              +`<div class="card"><b style="color:var(--warn)">${tot-(s.matches||0)}</b><span>matches to run</span></div>`+html;
  c.innerHTML=html;
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
  buildMatchPicker();
  buildForecastFilters();
  const opt=m=>`<option value="${m.match_id}">${esc(m.label)}${m.finished?' ✓':''}</option>`;
  const withPred=BOOT.matches.filter(m=>m.has_predictions);
  $('#cmpMatch').innerHTML=(withPred.length?withPred:BOOT.matches).map(opt).join('');
  loadEvaluate();
}

function buildRunForm(){
  $('#runReps').value='';
  $('#runMoments').innerHTML=BOOT.moments.map(m=>{
    const on=BOOT.default_moments.includes(m)?'checked':'';
    return `<label class="chk"><input type="checkbox" value="${m}" ${on}> ${m}</label>`;}).join('');
  $('#runModels').innerHTML=BOOT.models.map(m=>
    `<label class="chk"><input type="checkbox" value="${m.key}" checked> ${esc(m.key)} <span class="muted small">(${esc(m.provider)}) · ${m.reps}×</span></label>`).join('');
  const act=BOOT.actions_enabled;
  $('#btnRun').disabled=!act;
  $$('[data-fetch]').forEach(b=>b.disabled=!act); $('#btnReport').disabled=!act;
  updateEstimate();
  $('#runReps').oninput=updateEstimate;
  $('#runModels').onchange=$('#runMoments').onchange=updateEstimate;
}
function selModels(){return $$('#runModels input:checked').map(i=>i.value);}
function selMoments(){return $$('#runMoments input:checked').map(i=>i.value);}
function selMatches(){return $$('#runMatchList input:checked').map(i=>i.value);}
function updateEstimate(){
  const c=selMatches().length, matches=c||BOOT.matches.length;
  const override=+$('#runReps').value||null, orders=BOOT.orders||2;
  const repsSum=selModels().reduce((a,k)=>{const m=BOOT.models.find(x=>x.key===k);return a+(override||(m?m.reps:1));},0);
  const calls=matches*Math.max(selMoments().length,1)*BOOT.prompts.length*orders*repsSum;
  $('#runEstimate').textContent=`≈ ${calls.toLocaleString()} API calls (${orders} orderings)`;
  const mc=$('#matchCount'); if(mc)mc.textContent=c?`${c} selected`:'none → all matches';
}
$('#modelsAll').onclick=e=>{e.preventDefault();$$('#runModels input').forEach(i=>i.checked=true);updateEstimate();};
$('#modelsNone').onclick=e=>{e.preventDefault();$$('#runModels input').forEach(i=>i.checked=false);updateEstimate();};

$('#btnRun').onclick=async()=>{
  const body={match_ids:selMatches(),moments:selMoments(),models:selModels(),
    reps:+$('#runReps').value||null,retry_errors:$('#runRetry').checked,dry_run:$('#runDry').checked};
  if(!body.models.length)return toast('Select at least one model.');
  if(!body.moments.length)return toast('Select at least one moment.');
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
$('#btnMerge').onclick=async()=>{
  const f=$('#mergeFile').files[0];
  if(!f)return toast('Choose a .db file first.');
  const fd=new FormData();fd.append('file',f);fd.append('replace',$('#mergeReplace').checked?'true':'false');
  const btn=$('#btnMerge');btn.disabled=true;$('#mergeMsg').textContent='Uploading & merging…';
  try{const r=await api('./actions/merge-upload',{method:'POST',body:fd});
    $('#mergeMsg').innerHTML=`Merged <b>${esc(r.filename||'')}</b>: +${r.runs_added} forecast row(s), +${r.results_added} result row(s). `+
      `Totals: ${r.runs_total} forecasts, ${r.results_total} results.`;
    toast(`Merged +${r.runs_added} rows`);refreshSummary();}
  catch(e){$('#mergeMsg').textContent='Error: '+e.message;toast('Merge failed');}
  finally{btn.disabled=false;}
};

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

function applyMatchFilter(){
  const q=$('#runMatchSearch').value.toLowerCase(), onlyNot=$('#hideRun').checked;
  [...$('#runMatchList').children].forEach(el=>{
    el.style.display=(el.dataset.text.includes(q) && (!onlyNot || el.dataset.run==='0'))?'':'none';});
}
function buildMatchPicker(){
  const list=$('#runMatchList');
  list.innerHTML=BOOT.matches.map(m=>{
    const badge=m.has_predictions
      ? `<span class="pill ok" title="${m.n_runs} runs · ${m.n_models} models">run · ${m.n_runs}</span>`
      : `<span class="pill na">not run</span>`;
    const fin=m.finished?` <span class="pill" style="color:#7eb0ff">finished</span>`:'';
    return `<label class="chk" style="justify-content:flex-start" data-run="${m.has_predictions?1:0}" data-text="${esc((m.label+' '+(m.phase||'')).toLowerCase())}">
       <input type="checkbox" value="${esc(m.match_id)}"> ${esc(m.match_id)}. ${esc(m.team_1)} v ${esc(m.team_2)}
       <span class="muted small">${esc(m.local_date||'')}</span> ${badge}${fin}</label>`;}).join('');
  list.onchange=updateEstimate;
  $('#runMatchSearch').oninput=applyMatchFilter;
  $('#hideRun').onchange=applyMatchFilter;
  const setAll=v=>{[...list.querySelectorAll('input')].forEach(i=>i.checked=v);updateEstimate();};
  $('#matchAll').onclick=e=>{e.preventDefault();setAll(true);};
  $('#matchNone').onclick=e=>{e.preventDefault();setAll(false);};
  $('#matchNotRun').onclick=e=>{e.preventDefault();[...list.children].forEach(el=>el.querySelector('input').checked=(el.dataset.run==='0'));updateEstimate();};
  $('#matchShown').onclick=e=>{e.preventDefault();[...list.children].forEach(el=>{if(el.style.display!=='none')el.querySelector('input').checked=true;});updateEstimate();};
  const run=BOOT.matches.filter(m=>m.has_predictions).length, tot=BOOT.matches.length;
  $('#matchCoverage').innerHTML=`<b style="color:var(--ok)">${run}</b> already run · <b style="color:var(--warn)">${tot-run}</b> not run yet · ${tot} total`;
  updateEstimate();
}

let fxOffset=0; const fxLimit=50;
function buildForecastFilters(){
  const opts=(arr,fn)=>'<option value="">any</option>'+arr.map(fn).join('');
  $('#fxMatch').innerHTML=opts(BOOT.matches,m=>`<option value="${esc(m.match_id)}">${esc(m.match_id)}. ${esc(m.team_1)} v ${esc(m.team_2)}</option>`);
  $('#fxModel').innerHTML=opts(BOOT.models,m=>`<option value="${esc(m.key)}">${esc(m.key)}</option>`);
  $('#fxPrompt').innerHTML=opts(BOOT.prompts,p=>`<option value="${esc(p)}">${esc(p)}</option>`);
  $('#fxMoment').innerHTML=opts(BOOT.moments,m=>`<option value="${esc(m)}">${esc(m)}</option>`);
  $('#fxApply').onclick=()=>{fxOffset=0;loadForecasts();};
  $('#fxReset').onclick=()=>{['fxMatch','fxModel','fxPrompt','fxMoment','fxOrder','fxStatus','fxValid'].forEach(id=>$('#'+id).value='');$('#fxQ').value='';fxOffset=0;loadForecasts();};
  $('#fxQ').onkeydown=e=>{if(e.key==='Enter'){fxOffset=0;loadForecasts();}};
  $('#fxPrev').onclick=()=>{if(fxOffset>0){fxOffset=Math.max(0,fxOffset-fxLimit);loadForecasts();}};
  $('#fxNext').onclick=()=>{fxOffset+=fxLimit;loadForecasts();};
}
function fxQuery(){
  const p=new URLSearchParams(), g=id=>$('#'+id).value;
  if(g('fxMatch'))p.set('match_id',g('fxMatch'));
  if(g('fxModel'))p.set('model',g('fxModel'));
  if(g('fxPrompt'))p.set('prompt_id',g('fxPrompt'));
  if(g('fxMoment'))p.set('moment',g('fxMoment'));
  if(g('fxOrder'))p.set('order',g('fxOrder'));
  if(g('fxStatus'))p.set('status',g('fxStatus'));
  if(g('fxValid'))p.set('valid',g('fxValid'));
  if(g('fxQ'))p.set('q',g('fxQ'));
  p.set('limit',fxLimit);p.set('offset',fxOffset);
  return p.toString();
}
async function loadForecasts(){
  try{const d=await api('./api/forecasts?'+fxQuery());
    const score=r=>(r.parsed_score_team_1!=null&&r.parsed_score_team_2!=null)?`${r.parsed_score_team_1}-${r.parsed_score_team_2}`:'–';
    const vp=v=>v?'<span class="pill ok">ok</span>':'<span class="pill bad">no</span>';
    const body=d.rows.map(r=>`<tr class="clickrow" data-id="${esc(r.run_id)}">
      <td>${esc(r.match_id)}</td><td>${esc(r.team_1)} v ${esc(r.team_2)}</td><td>${esc(r.model)}</td>
      <td>${esc(r.prompt_id)}</td><td>${esc(r.match_moment||'')}</td><td>${esc(r.team_order_type||'')}</td>
      <td>${esc(r.repetition_number)}</td><td>${score(r)}</td><td>${vp(r.json_valid)}</td>
      <td>${esc(r.execution_status)}</td><td>${r.total_tokens==null?'':r.total_tokens}</td></tr>`).join('');
    $('#fxTable').innerHTML=d.rows.length
      ? `<table><tr><th>#</th><th>Match</th><th>Model</th><th>Prompt</th><th>Moment</th><th>Order</th><th>Rep</th><th>Score</th><th>JSON</th><th>Status</th><th>Tok</th></tr>${body}</table>`
      : '<p class="muted small" style="padding:10px">No rows match these filters.</p>';
    $('#fxCount').textContent=`${d.total.toLocaleString()} rows`;
    $('#fxPage').textContent=d.total?`${d.offset+1}–${Math.min(d.offset+d.limit,d.total)} of ${d.total}`:'0';
    $('#fxPrev').disabled=d.offset<=0; $('#fxNext').disabled=d.offset+d.limit>=d.total;
    $$('#fxTable .clickrow').forEach(tr=>tr.onclick=()=>openDetail(tr.dataset.id));
  }catch(e){$('#fxTable').innerHTML='<p class="muted small" style="padding:10px">'+esc(e.message)+'</p>';}
}
async function openDetail(runId){
  try{const r=await api('./api/forecast/'+encodeURIComponent(runId));
    const row=(k,v)=>(v==null||v==='')?'':`<div class="k">${esc(k)}</div><div>${esc(v)}</div>`;
    let hats='';['white_hat','red_hat','black_hat','yellow_hat','green_hat','blue_hat'].forEach(h=>{if(r[h])hats+=`<div class="k">${h}</div><div>${esc(r[h])}</div>`;});
    $('#fxModalBox').innerHTML=`<span class="x" id="fxClose">✕</span>
      <h2 style="margin-top:0">${esc(r.model)} · ${esc(r.prompt_id)} · ${esc(r.match_moment||'')}</h2>
      <div class="kv">
        ${row('match',r.match_id+'. '+r.team_1+' v '+r.team_2)}${row('order',r.team_order_type)}${row('rep',r.repetition_number)}
        ${row('predicted',(r.parsed_score_team_1!=null?r.parsed_score_team_1+'-'+r.parsed_score_team_2:'–'))}
        ${row('probs (1/X/2)',[r.parsed_team1_win_probability,r.parsed_draw_probability,r.parsed_team2_win_probability].join(' / '))}
        ${row('json valid',r.json_valid?'yes':'no')}${row('status',r.execution_status)}${row('latency ms',r.latency_ms)}
        ${row('tokens p/c/total',[r.prompt_tokens,r.completion_tokens,r.total_tokens].join(' / '))}${row('cost usd',r.api_cost)}
        ${row('error',r.error_message)}</div>
      ${hats?`<h2>Six hats</h2><div class="kv">${hats}</div>`:''}
      <h2>Raw response</h2><pre class="log" style="max-height:260px">${esc(r.raw_response||'(none)')}</pre>
      <h2>Prompt sent</h2><pre class="log" style="max-height:200px">${esc(r.prompt_text||'')}</pre>`;
    $('#fxModal').classList.remove('hide');
    $('#fxClose').onclick=()=>$('#fxModal').classList.add('hide');
  }catch(e){toast('Error: '+e.message);}
}
$('#fxModal').onclick=e=>{if(e.target.id==='fxModal')$('#fxModal').classList.add('hide');};

boot().then(()=>{ // resume polling if a job is already running
  fetch('./api/job').then(r=>r.json()).then(j=>{if(j.status==='running')startPolling();else renderJob(j);});
}).catch(e=>toast('Load error: '+e.message,8000));
</script>
</body></html>
"""
